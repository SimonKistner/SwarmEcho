"""Independent validation with frozen cases, bounded batches, and statistics.

Host-side planning/reporting has no JAX dependency. Device imports are confined
to preparation/execution so configuration and persistence can be checked on CPU.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np

from swarmecho.training.adaptive_spawn import atomic_json
from swarmecho.training.information_metrics import information_percentages


def case_plan(seed, num_maps, num_episodes, stage, generation=0):
    """IID uniform maps and per-case keys, independent of lane/chunk counts."""
    maps = np.random.default_rng(np.random.SeedSequence([seed, generation, 701]))
    keys = np.random.default_rng(np.random.SeedSequence([seed, stage, generation, 709]))
    return (maps.integers(num_maps, size=num_episodes, dtype=np.int32),
            keys.integers(0, 2**32, size=(num_episodes, 2), dtype=np.uint32))


def summarize_outcomes(map_ids, outcomes, map_names, max_steps):
    """Pool IID episode results; also report equally weighted observed-map rates."""
    success = np.asarray(outcomes["success"], dtype=bool)
    count = len(success)
    if not count:
        raise ValueError("Validation produced no episodes.")
    counts = np.bincount(map_ids, minlength=len(map_names))
    successes = np.bincount(map_ids, weights=success, minlength=len(map_names))
    observed = counts > 0
    rates = successes[observed] / counts[observed]
    mean = float(success.mean())
    z = 1.959963984540054
    denominator = 1 + z*z/count
    centre = (mean + z*z/(2*count)) / denominator
    radius = z * math.sqrt(mean*(1-mean)/count + z*z/(4*count*count)) / denominator
    successful_length = float(np.mean(outcomes["length"][success])) if np.any(success) else float(max_steps)
    metrics = {
        "success_rate": mean, "success_count": int(success.sum()), "episodes": count,
        "success_ci_low": max(0., centre-radius), "success_ci_high": min(1., centre+radius),
        "maps_evaluated": int(observed.sum()), "maps_total": len(map_names),
        "map_coverage": float(observed.mean()), "map_balanced_success_rate": float(rates.mean()),
        "worst_map_success_rate": float(rates.min()), "best_map_success_rate": float(rates.max()),
        "median_map_success_rate": float(np.median(rates)), "p10_map_success_rate": float(np.quantile(rates, .1)),
        "success_episode_length": successful_length,
        "success_length_reduction": 1 - successful_length/max_steps,
    }
    for field in ("return", "length", "population", "target_found", "visually_found"):
        metrics[f"mean_{field}"] = float(np.mean(outcomes[field]))
    if "useless_agents" in outcomes and np.any(success):
        metrics["useless_agents_at_success"] = float(np.mean(outcomes["useless_agents"][success]))
    rows = []
    for index, name in enumerate(map_names):
        selected = map_ids == index
        row = {"map_id": name, "episodes": int(counts[index]), "successes": int(successes[index]),
               "success_rate": float(successes[index]/counts[index]) if counts[index] else None}
        for field in ("return", "length", "population"):
            row[f"mean_{field}"] = float(np.mean(outcomes[field][selected])) if counts[index] else None
        rows.append(row)
    return metrics, rows


def prepare_generated_validation_maps(level, destination, log, training_buildings=None):
    from swarmecho.env.random_buildings import (
        GENERATOR_VERSION, generate_building, map_seed, save_generated_map, target_exclusion_parameters,
    )
    from swarmecho.training.map_selection import building_identity
    folder = Path(destination) / "validation" / "generated_maps"
    # Use the generator's manifest rather than assuming a filename convention.
    manifest = folder / "selection.json"
    settings = {"seed": level.validation.seed, "count": level.validation.num_maps,
                "generation": asdict(level.random_buildings), "version": GENERATOR_VERSION}
    if level.random_buildings.advanced_exclustion_base_LOS:
        settings["target_exclusion_parameters"] = target_exclusion_parameters(level.random_buildings, level.env)
    train_hashes = (sorted({building_identity(building, "")["sha256"] for building in training_buildings})
                    if training_buildings is not None else None)
    if manifest.is_file():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved["settings"] != settings:
            raise ValueError("Generated validation pool settings changed; choose a new run.")
        if train_hashes is not None and saved["training_geometry"] != train_hashes:
            raise ValueError("Generated training split changed relative to the frozen validation pool.")
        paths = [folder / name for name in saved["files"]]
        if not all(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == sha
                   for path, sha in zip(paths, saved["sha256"])):
            raise ValueError("Frozen generated validation map is missing.")
        return paths
    if training_buildings is None:
        raise ValueError("Frozen generated validation selection is missing; create it during training startup.")
    folder.mkdir(parents=True, exist_ok=True)
    paths, seen = [], set(train_hashes)
    for draw in range(max(1000, level.validation.num_maps*100)):
        name = f"eval_{len(paths):04d}"
        data, building = generate_building(level.random_buildings,
            map_seed(level.validation.seed, draw, evaluation=True), name=name,
            drone_clearance=level.env.drone_radius + level.env.obstacle_planning_clearance_m,
            env_cfg=level.env)
        sha = building_identity(building, "")["sha256"]
        if sha in seen:
            continue
        seen.add(sha)
        paths.append(save_generated_map(data, folder / f"{name}.yaml"))
        if len(paths) == 1 or len(paths) % 100 == 0:
            log(f"Generated {len(paths)}/{level.validation.num_maps} distinct validation maps")
        if len(paths) == level.validation.num_maps:
            break
    if len(paths) != level.validation.num_maps:
        raise ValueError("Could not generate enough validation geometry distinct from training.")
    atomic_json(manifest, {"settings": settings, "training_geometry": train_hashes,
        "files": [path.name for path in paths],
        "sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]})
    return paths


class ValidationRunner:
    """One immutable validation task per run/agent stage; no visual artifacts."""
    def __init__(self, level, destination, paths, *, checkpoint_path=None, checkpoint_dir=None, log=lambda text: None):
        from swarmecho.core.config import resolve_validation_level
        level = resolve_validation_level(level)
        self.training_target_budget = level.env.num_agents
        if level.validation.mode == "uniform_full_task":
            paths = level.map_paths
            # A virtual training budget must never restrict the full-task cases.
            # Reset uses uniform legal-volume proposals at the full population.
        self.level, self.config = level, level.validation
        self.directory = Path(destination) / "validation"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.paths = tuple(Path(path) for path in paths or ())
        self.geo_labels = None
        self.probe_grid_shape = None
        if len(self.paths) != self.config.num_maps:
            raise ValueError("Validation map selection differs from validation.num_maps.")
        self.map_names = [path.stem for path in self.paths]
        identities = [{"name": path.stem, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                      for path in self.paths]
        # Keep the saved validation identity for G03 contracts predating these fixed choices.
        env_identity = {
            **asdict(level.env), "wall_sensing_layout": "fibonacci", "base_keeps_informing": False,
            "observe_base_vector": False, "observe_target_vector": False, "observe_seeing_target": False,
            "observe_base_vote_ratio": False, "observe_current_timestep": False, "observe_coverage_probe": False,
            "no_movement_termination_steps": 1000, "relay_agent_budget": None, "roadmap_approach": "full",
            "num_obstacles": 0, "obstacle_size_min_m": 2.0, "obstacle_size_max_m": 4.0,
            "obstacle_spawn_layer_min": 2, "obstacle_spawn_layer_max": 5, "obstacle_boundary_buffer_m": 0.5,
            "obstacle_target_buffer_m": 0.5, "roadmap_corner_bonus_m": 0.0,
        }
        env_identity["adapt_size"] = {
            **env_identity["adapt_size"], "initial_agents": level.env.num_agents,
            "vote_mode": "hold", "vote_protocol": "latest_decision", "vote_holding": 1,
            "quorum_fraction": 1.0, "quorum_hold": 3, "recruitment_enabled": False,
            "reward_by_reference_enabled": False, "reward_by_reference_fraction": 0.2,
            "reward_by_reference_steps": 500,
        }
        reward_identity = {
            **asdict(level.reward), "no_movement_termination_penalty": 0.0,
            "success_reward_as_hold_record": False, "peer_informing_reward_enabled": False,
            "peer_informing_reward": 50, "gap_reward_uses_change": False, "gap_change_reward_per_meter": 3.0,
            "delivery_gap_vesting_enabled": False, "delivery_gap_vesting_steps": 10,
            "reinforcement_reward": 0, "rejected_call_reward": 0,
            "chain_reward_system": "obstacle_geodesic", "allow_redundancy_reward": False,
            "progress_based_cluster_tiebreaker_soft": True, "target_found_requires_delivery": True,
            "enable_chain_efficiency_reward": level.reward.chain_efficiency_reward != 0,
            "priveleged_vote_reward": {"enabled": False, "withheld_reinforcement_vote_reward": -1.0,
                                       "unnecessary_reinforcement_call_reward": -500},
        }
        task = {"maps": identities, "env": env_identity, "reward": reward_identity,
                "dynamic_max_geo_distance": {"enabled": True, "mode": "global_candidates", **asdict(level.training.dynamic_max_geo_distance)},
                "seed": self.config.seed, "num_episodes": self.config.num_episodes,
                "fixed_targets": self.config.fixed_targets,
                "gate": [level.evaluation.early_exit_success_rate, level.evaluation.early_exit_hold_evals,
                         level.evaluation.early_exit_min_success_length_reduction]}
        if self.config.mode is not None:
            task["mode"] = self.config.mode
        self.identity = hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()
        metadata = self.directory / "metadata.json"
        if metadata.is_file() and json.loads(metadata.read_text(encoding="utf-8"))["identity"] != self.identity:
            raise ValueError("Validation maps, stage or case settings changed; use a new run.")
        atomic_json(metadata, {"format": "swarmecho-validation/v1", "identity": self.identity,
            "maps": identities, "config": asdict(self.config), "stage": level.env.num_agents,
            "success_condition": level.env.success_condition})
        self.best_success_rate = None
        self.hold_count, self.last_update = 0, -1
        self.last_metrics = {}
        self.case_targets = None
        self._functions = None
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        resume = level.training.ckpt_loading_mode == "resume"
        history = self.directory / "history.jsonl"
        if history.is_file() and history.stat().st_size and not resume and os.environ.get("SWARMECHO_RECOVERY_RETRY") != "1":
            raise ValueError("Validation history exists; resume explicitly or choose a new run.")
        if self.checkpoint_path and (self.checkpoint_path / "validation_state.json").is_file():
            state = json.loads((self.checkpoint_path / "validation_state.json").read_text(encoding="utf-8"))
            if state["identity"] == self.identity:
                if resume:
                    self._restore_state(state)
                source = self.checkpoint_path / "validation_cases.npz"
                if self.config.fixed_targets and source.is_file() and not self.cases_path.exists():
                    shutil.copy2(source, self.cases_path)
                if (resume and state.get("last_update", -1) >= 0 and self.config.fixed_targets
                        and not self.cases_path.is_file()):
                    raise ValueError("Checkpoint has validation outcomes but its frozen cases are missing.")
            elif resume:
                raise ValueError("Checkpoint validation identity differs from this stage/configuration.")
        if self.config.fixed_targets and self.cases_path.is_file():
            with np.load(self.cases_path, allow_pickle=False) as data:
                identity = bytes(data["identity"]).decode()
                if identity != self.identity:
                    raise ValueError("Frozen validation case identity changed.")
                self.map_ids = data["map_ids"].copy()
                self.case_keys = data["keys"].copy()
                self.case_targets = data["targets"].copy()
            self._check_cases()
        else:
            self.map_ids, self.case_keys = case_plan(self.config.seed, len(self.paths), self.config.num_episodes,
                                                   level.env.num_agents)
        # A newer best model remains valid when explicitly rewinding a run to an older checkpoint.
        best = self._best_state_path(destination)
        if (resume or os.environ.get("SWARMECHO_RECOVERY_RETRY") == "1") and best.is_file():
            state = json.loads(best.read_text(encoding="utf-8"))
            if state.get("identity") == self.identity and state.get("best_success_rate") is not None:
                self.best_success_rate = max(self.best_success_rate or -1., state["best_success_rate"])
        with history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"kind": "session", "update": self._resume_update() if resume else 0}) + "\n")
        scope = "uniform full-task authored maps" if self.config.mode == "uniform_full_task" else "validation maps"
        log(f"Validation: {len(self.paths)} {scope}, {self.config.num_episodes} episodes, "
            f"batches of at most {self.config.parallel_envs}; success={level.env.success_condition}; no heatmaps/replays.")

    @property
    def cases_path(self):
        return self.directory / "cases.npz"

    def _resume_update(self):
        if self.checkpoint_path and (self.checkpoint_path / "step_history.json").is_file():
            return int(json.loads((self.checkpoint_path / "step_history.json").read_text())["update"])
        return self.last_update

    def _best_state_path(self, destination):
        directory = self.level.evaluation.checkpoint_dir
        root = self.checkpoint_dir or (Path(directory) if directory else Path(destination) / "checkpoints")
        return root / "best_val_checkpoint" / "validation_state.json"

    def _restore_state(self, state):
        self.best_success_rate = state.get("best_success_rate")
        self.hold_count = int(state.get("hold_count", 0))
        self.last_update = int(state.get("last_update", -1))
        self.last_metrics = state.get("last_metrics", {})

    def _check_cases(self):
        n = self.config.num_episodes
        if (self.map_ids.shape != (n,) or self.map_ids.dtype.kind not in "iu"
                or np.any(self.map_ids < 0) or np.any(self.map_ids >= len(self.paths))
                or self.case_keys.shape != (n, 2) or self.case_keys.dtype != np.uint32
                or self.case_targets.shape != (n, 3) or not np.all(np.isfinite(self.case_targets))):
            raise ValueError("Invalid frozen validation cases.")

    def _prepare(self):
        if self._functions is not None:
            return self._functions
        from swarmecho.env.buildings import load_building
        from swarmecho.env.building_bank import prepare_bank, prepare_bank_from_records
        from swarmecho.env.environment import make_env_fns
        level, cfg = self.level, self.level.env
        buildings = [load_building(path) for path in self.paths]
        plan = True
        options = dict(dynamic_max_geo_distance=True,
            relay_options=level.training.dynamic_max_geo_distance, randomize_base=False)
        pool = self.paths[0].parent.parent
        records = None
        if all(path.parent == self.paths[0].parent and (pool / "records" / f"{path.stem}.npz").is_file()
               for path in self.paths):
            from swarmecho.curriculum_config.maps.random_building_bank.generate_bank import load_records
            records = load_records(pool, [{"id": path.stem} for path in self.paths])
            bank = prepare_bank_from_records(records, buildings[0], cfg,
                interiors=[b.interior_cells for b in buildings], relay_cache_dir=pool / "relay_cache", **options)
        else:
            bank = prepare_bank(buildings, cfg, plan=plan, **options)
        if self.probe_grid_shape is not None:
            from swarmecho.training.geo_auxiliary import PathLabels
            self.geo_labels = PathLabels(self.paths,cfg,buildings=buildings,records=records,
                directory=self.directory/"geo_probe_auxiliary",
                grid_shape=self.probe_grid_shape)
        self._functions = make_env_fns(buildings[0], cfg, building_bank=bank,
            plan_geodesic=plan,
            spawn_pair_max_attempts=level.training.spawn_pair_max_attempts,
            **options)
        return self._functions

    def run(self, model, *, update, steps, spawn_sampler=None):
        import jax
        import jax.numpy as jnp
        from swarmecho.training.train import _run_parallel_evaluation_jit
        from swarmecho.core.terminal import terminal_print
        self.probe_grid_shape = getattr(getattr(model, "geo_probe_head", None), "grid_shape", None)
        reset, step, observations, env_metrics = self._prepare()
        score_spawn = getattr(spawn_sampler, "curriculum_source", None) == "validation"
        return_required = True
        self.training_target_budget = getattr(spawn_sampler, "curriculum_budget", None) or self.level.env.num_agents
        if not self.config.fixed_targets:
            self.map_ids, self.case_keys = case_plan(self.config.seed, len(self.paths), self.config.num_episodes,
                self.level.env.num_agents, generation=update)
            self.case_targets = None
        fields = {name: [] for name in ("return", "length", "success", "target_found", "visually_found", "population")}
        if self.level.env.adapt_size.enabled:
            fields["useless_agents"] = []
        if return_required:
            fields["required_agents"] = []
        targets = []
        geo_scores = []
        information_counts = np.zeros(3, dtype=np.int64)
        return_geo = hasattr(model,"geo_probe_head")
        batch_size = min(self.config.parallel_envs, self.config.num_episodes)
        for start in range(0, self.config.num_episodes, batch_size):
            end = min(start+batch_size, self.config.num_episodes)
            indices = np.minimum(np.arange(start, start+batch_size), self.config.num_episodes-1)
            batch = jax.device_get(_run_parallel_evaluation_jit(model,
                jax.random.PRNGKey(self.config.seed),
                reset=reset, step=step, observations=observations, env_metrics=env_metrics,
                reward_cfg=self.level.reward, cfg=self.level.env, num_envs=batch_size,
                horizon=self.level.env.max_steps,
                case_map_ids=jnp.asarray(self.map_ids[indices]), case_keys=jnp.asarray(self.case_keys[indices]),
                case_targets=jnp.asarray(self.case_targets[indices]) if self.case_targets is not None else None,
                return_population=True, return_required_agents=return_required,
                return_useless_agents=self.level.env.adapt_size.enabled,
                return_information=True, return_geo=return_geo))
            size = end-start
            failed = np.flatnonzero(np.asarray(batch[12])[:size] == 0)
            if len(failed):
                names = [self.map_names[self.map_ids[start+int(i)]] for i in failed[:8]]
                raise ValueError(f"No certified validation target for map(s) {names}; no unfiltered fallback.")
            if not np.all(np.asarray(batch[14])[:size]):
                raise ValueError("A validation map has no valid decommission zone.")
            for name, column in (("return", 0), ("length", 1), ("success", 2),
                                 ("target_found", 3), ("visually_found", 6), ("population", 15)):
                fields[name].append(np.asarray(batch[column])[:size])
            if return_required:
                fields["required_agents"].append(np.asarray(batch[16])[:size])
            if self.level.env.adapt_size.enabled:
                fields["useless_agents"].append(np.asarray(batch[16+int(return_required)])[:size])
            information_index = 16 + int(return_required) + int(self.level.env.adapt_size.enabled)
            information_counts += np.asarray(batch[information_index])[:size].sum(axis=0)
            if return_geo:
                from swarmecho.training.geo_auxiliary import score_head
                features = np.asarray(batch[-2])[:size]
                valid = np.array(batch[-1][:size],copy=True)
                grid_width = self.geo_labels.grid_width
                labels = np.zeros((size, grid_width), bool)
                for lane in range(size):
                    if not valid[lane].any():
                        continue
                    target = np.asarray(batch[8])[lane]
                    label = self.geo_labels.get(self.map_ids[start+lane],np.asarray(batch[9])[lane],target)
                    if label is None:
                        valid[lane] = False
                    else:
                        labels[lane] = label
                scored = np.asarray(jax.device_get(score_head(model.geo_probe_head,jnp.asarray(features),
                    jnp.asarray(np.broadcast_to(labels[:, None, :], (*valid.shape, grid_width))),
                    model.geo_probe_head.threshold)))
                for lane in range(size):
                    if valid[lane].any():
                        geo_scores.append(scored[lane,valid[lane]].mean(axis=0))
            targets.append(np.asarray(batch[8])[:size])
            terminal_print(f"[VALIDATION] {end}/{self.config.num_episodes} episodes", flush=True)
        outcomes = {name: np.concatenate(values) for name, values in fields.items()}
        sampled_targets = np.concatenate(targets)
        if self.case_targets is not None and not np.array_equal(sampled_targets, self.case_targets):
            raise ValueError("Validation reset did not retain the frozen target positions.")
        self.case_targets = sampled_targets
        if self.config.fixed_targets and not self.cases_path.is_file():
            temporary = self.cases_path.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, identity=np.frombuffer(self.identity.encode(), dtype=np.uint8),
                    map_ids=self.map_ids, keys=self.case_keys, targets=self.case_targets)
            os.replace(temporary, self.cases_path)
        metrics, per_map = summarize_outcomes(self.map_ids, outcomes, self.map_names, self.level.env.max_steps)
        metrics.update(information_percentages(information_counts))
        if return_required:
            for group in range(1, self.level.env.num_agents + 1):
                selected = outcomes["required_agents"] == group
                metrics[f"required_agents_{group}/episodes"] = int(selected.sum())
                if selected.any():
                    metrics[f"required_agents_{group}/success_rate"] = float(outcomes["success"][selected].mean())
        if geo_scores:
            score = np.mean(geo_scores,axis=0)*100
            metrics.update({"geo_probe_auxiliary/validation/exact_path_accuracy_pct":float(score[0]),
                "geo_probe_auxiliary/validation/path_similarity_pct":float(score[1]),
                "geo_probe_auxiliary/validation/soft_path_similarity_pct":float(score[2])})
        if score_spawn:
            scores = spawn_sampler.observe_curriculum("validation", outcomes["required_agents"],
                outcomes["success"], outcomes["length"], self.level.env.max_steps)
            metrics.update({"spawn_curriculum_" + key: value for key, value in scores.items()})
        self.last_update, self.last_metrics = int(update), metrics
        row = {"format": "swarmecho-validation-result/v1", "kind": "validation", "identity": self.identity,
               "update": int(update), "global_step": int(steps), "stage": self.level.env.num_agents,
               "agents": self.level.env.num_agents,
               "success_condition": self.level.env.success_condition,
               "metrics": metrics, "information_counts": information_counts.tolist()}
        if self.config.mode == "uniform_full_task":
            row.update(mode=self.config.mode, training_target_budget=self.training_target_budget)
        if self.config.save_per_map_stats:
            row["maps"] = per_map
        if return_geo:
            row["geo_probe_auxiliary"] = dict(episodes_with_scores=len(geo_scores),
                label_requests=self.geo_labels.requested,missing_references=self.geo_labels.missing)
        atomic_json(self.directory / f"results_u{update:06d}.json", row)
        with (self.directory / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({key: value for key, value in row.items() if key != "maps"}, allow_nan=False) + "\n")
        return metrics

    def observe_gate(self, metrics):
        ev = self.level.evaluation
        if not ev.early_exit:
            return False
        threshold = (ev.final_early_exit_success_rate
                     if (ev.target_sampling == "curriculum" or self.level.training.forward_target_curriculum.enabled)
                     and ev.final_early_exit_success_rate is not None
                     else ev.early_exit_success_rate)
        qualifies = metrics["success_rate"] >= threshold
        if ev.early_exit_min_success_length_reduction is not None:
            qualifies &= metrics["success_length_reduction"] >= ev.early_exit_min_success_length_reduction
        self.hold_count = self.hold_count+1 if qualifies else 0
        return self.hold_count >= max(1, ev.early_exit_hold_evals)

    def save(self, checkpoint):
        atomic_json(Path(checkpoint) / "validation_state.json", {
            "identity": self.identity, "best_success_rate": self.best_success_rate,
            "hold_count": self.hold_count, "last_update": self.last_update, "last_metrics": self.last_metrics})
        if self.config.fixed_targets and self.cases_path.is_file():
            shutil.copy2(self.cases_path, Path(checkpoint) / "validation_cases.npz")
