"""Capture matched discovery/communication events without policy optimization."""
from __future__ import annotations

import hashlib
import time

import numpy as np

from geo_experiments.common import ROOT, SOURCES, fingerprint, resolve_path, sample_count
from geo_experiments.artifacts import add_group, archive, group, save_data, update_run
from geo_experiments.console import terminal_print as print


def prepare_policy(config, run_dir=None, *, allow_suite=False):
    # Import the existing runtime only here: plotting saved results needs no JAX.
    from swarmecho.core.config import LEVEL_DIR, MAP_DIR, load_level, resolve_evaluation_level
    from swarmecho.core.compatibility import resolve_level_file, resolve_map_file
    from swarmecho.training.checkpoints import restore_model_checkpoint
    from swarmecho.training.train import build_model

    if not config["policy"]["checkpoint"]:
        raise ValueError("Set policy.checkpoint to a checkpoint trained with the configured level (G01_static by default).")
    checkpoint = resolve_path(config["policy"]["checkpoint"])
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    from geo_experiments.policy import load_probe_level
    configured_level, source, original, ignored = load_probe_level(config, checkpoint, run_dir)
    level = resolve_evaluation_level(configured_level)
    if allow_suite and configured_level.evaluation.eval_differes_from_training_map:
        raise ValueError("Suite probes use the authored training map list; separate evaluation-map substitution is unsupported.")
    if (configured_level.random_buildings.enabled or (configured_level.static_map_suite and not allow_suite)
            or configured_level.evaluation.random_eval or level.env.num_obstacles
            or (level.env.adapt_size.enabled and not allow_suite) or level.training.randomize_base):
        raise ValueError("Phase 0 supports one authored static map, fixed base, fixed population, no random obstacles.")
    if level.env.success_condition == "coverage" or not level.network.memory_comm_enabled:
        raise ValueError("Target probes require a target mission with learned communication enabled.")
    map_path = (resolve_map_file(configured_level.evaluation.eval_map, MAP_DIR).resolve()
                if configured_level.evaluation.eval_differes_from_training_map
                and configured_level.evaluation.eval_map else configured_level.map_paths[0])
    print(f"[policy] level={level.name}; agents={level.env.num_agents}; max_steps={level.env.max_steps}", flush=True)
    model = build_model(level)
    restore_model_checkpoint(model, checkpoint)
    checkpoint_hash = hashlib.sha256()
    for file in sorted(p for p in checkpoint.rglob("*") if p.is_file()):
        checkpoint_hash.update(file.relative_to(checkpoint).as_posix().encode())
        checkpoint_hash.update(fingerprint(file).encode())
    runtime_files = {file.relative_to(ROOT).as_posix(): fingerprint(file)
                     for file in sorted((ROOT / "src" / "swarmecho").rglob("*.py"))}
    runtime_hash = hashlib.sha256()
    for name, digest in runtime_files.items():
        runtime_hash.update(name.encode())
        runtime_hash.update(digest.encode())
    # Identity is portable: copying a checkpoint/config to another host is OK.
    identity = {
        "checkpoint_sha256": checkpoint_hash.hexdigest(), "map_sha256": fingerprint(map_path),
        "level_sha256": fingerprint(source),
        "runtime_sha256": runtime_hash.hexdigest(), "collector_sha256": fingerprint(__file__),
        "seed": config["dataset"]["seed"], "max_steps": level.env.max_steps,
        "success_condition": level.env.success_condition, "num_agents": level.env.num_agents,
        "communication_every_k_steps": model.memory_comm_every_k_steps,
    }
    if allow_suite:
        identity["maps"] = [{"name": name, "sha256": fingerprint(path)}
                            for name, path in zip(configured_level.map_names, configured_level.map_paths, strict=True)]
        identity["probe_collection_files"] = {name: fingerprint(ROOT / "geo_experiments" / name)
            for name in ("collection.py", "target_collection.py", "labels.py", "auditing.py", "policy.py", "common.py", "geometry.py")}
    if run_dir is not None:
        update_run(run_dir, provenance={"identity": identity, "checkpoint": str(checkpoint),
                   "level": level.name, "level_source": str(source), "map_source": str(map_path),
                   "runtime_files": runtime_files,
                   "experiment_files": {p.name: fingerprint(p) for p in (ROOT / "geo_experiments").glob("*.py")}},
                   definitions={"level.yaml": original, "ignored_legacy_reward_payouts": ignored,
                                **{f"maps/{i}.yaml": path.read_text(encoding="utf-8")
                                   for i, path in enumerate(configured_level.map_paths)},
                                "map.yaml": map_path.read_text(encoding="utf-8")})
    return model, level, identity


def make_collector(level, *, history_stride=None, own_stride=None, stop_after_finder=False):
    """Shared evaluation rollout; finder prefixes or full-team own-cell snapshots."""
    if history_stride is not None and (type(history_stride) is not int or history_stride < 1):
        raise ValueError("history_stride must be a positive integer or None.")
    if own_stride is not None:
        if type(own_stride) is not int or own_stride < 1:
            raise ValueError("own_stride must be a positive integer.")
        history_stride = 1
    import jax
    import jax.numpy as jnp
    from flax import nnx

    from swarmecho.env.adapt_size import policy_resets
    from swarmecho.env.obstacles import segments_blocked
    from swarmecho.models.actions import physical_actions, deterministic_actions
    from swarmecho.training.train import _evaluation_env_fns, communication_due

    reset, step, observations, _ = _evaluation_env_fns(level)
    cfg = level.env

    @nnx.jit
    def collect_batch(model, keys):
        def one(key):
            state = reset(key)
            n = cfg.num_agents
            initial = {
                "state": state, "hidden": model.initial_actor_hidden((), num_agents=n),
                "signature": model.initial_actor_signature((), num_agents=n),
                "value": model.initial_actor_value((), num_agents=n),
                "base_valid": jnp.bool_(False),
                "base_sig": jnp.zeros(model.tarmac_sig_dim),
                "base_val": jnp.zeros(model.tarmac_val_dim),
                "finder_id": jnp.int32(-1), "discovery_step": jnp.int32(-1),
                "finder_step": jnp.int32(-1), "base_step": jnp.int32(-1),
                "reporter_id": jnp.int32(-1),
                "finder_gru": jnp.zeros(model.hidden_dim),
                "finder_message": jnp.zeros(model.tarmac_sig_dim + model.tarmac_val_dim),
                "finder_position": jnp.zeros(3), "reporter_position": jnp.zeros(3),
                "finder_active_agents": jnp.int32(-1), "base_active_agents": jnp.int32(-1),
            }
            if history_stride is not None:
                slots = (cfg.max_steps + history_stride - 1) // history_stride + 1
                # The finder is unknown until sighting, so temporarily record the team.
                # Only the selected finder's prefix is returned to the host.
                initial.update({
                    "history_steps": jnp.full(slots, -1, dtype=jnp.int32),
                    "history_obs": jnp.zeros((slots, n, model.obs_dim)),
                    "history_encoded": jnp.zeros((slots, n, model.hidden_dim)),
                    "history_context": jnp.zeros((slots, n, model.tarmac_val_dim)),
                    "history_flags": jnp.zeros((slots, n, 2), dtype=bool),
                })
            if own_stride is not None:
                sample_slots = (cfg.max_steps + own_stride - 1) // own_stride
                initial.update({
                    "sample_valid": jnp.zeros((sample_slots, n), dtype=bool),
                    "sample_position": jnp.zeros((sample_slots, n, 3)),
                    "sample_gru": jnp.zeros((sample_slots, n, model.hidden_dim)),
                    "sample_message": jnp.zeros((sample_slots, n, model.tarmac_sig_dim + model.tarmac_val_dim)),
                    "sample_found": jnp.zeros(sample_slots, dtype=bool),
                })

            def advance(c):
                state = c["state"]
                sees = state.directly_sees_target & state.active
                discovered = (c["finder_id"] < 0) & jnp.any(sees)
                finder = jnp.where(discovered, jnp.argmax(sees.astype(jnp.int32)), c["finder_id"])
                discovery_step = jnp.where(discovered, state.step, c["discovery_step"])
                delta = state.pos[:, None, :] - state.pos[None, :, :]
                comm = ((jnp.linalg.norm(delta, axis=-1) <= cfg.comm_radius)
                        & state.active[:, None] & state.active[None, :]
                        & ~jnp.eye(n, dtype=bool))
                base_range = ((jnp.linalg.norm(state.pos - state.base_pos, axis=-1) <= cfg.comm_radius_base)
                              & state.active)
                if state.solid_min.shape[0]:
                    comm &= ~segments_blocked(state.pos[:, None, :], state.pos[None, :, :],
                                              state.solid_min, state.solid_max)
                    base_range &= ~segments_blocked(state.pos, state.base_pos,
                                                    state.solid_min, state.solid_max)
                share = communication_due(state.step, model.memory_comm_every_k_steps)
                receivers = jnp.ones(n, dtype=bool) if cfg.base_keeps_informing else ~state.target_known
                obs = observations(state)
                resets = policy_resets(state)
                base_mask = c["base_valid"] & base_range & receivers & share
                hidden, signature, value, means, _ = model.actor.__call_team__(
                    obs, c["hidden"], c["signature"], c["value"],
                    reset=resets, active=state.active, comm_mask=comm & share,
                    base_signature=c["base_sig"], base_value=c["base_val"],
                    base_memory_mask=base_mask,
                    deterministic=True,
                )
                # Match evaluation: current tokens are emitted from the updated
                # GRU; the base saves these tokens, not an attention aggregate.
                take_finder = (finder >= 0) & (c["finder_step"] < 0) & share
                safe_finder = jnp.maximum(finder, 0)
                reporters = state.target_known & base_range
                reporter = jnp.argmax(reporters.astype(jnp.int32))
                take_base = ~c["base_valid"] & jnp.any(reporters) & share
                actions = deterministic_actions(model.actor, hidden, means)
                next_state = step(state, jnp.where(state.active[:, None], physical_actions(actions), 0.0))
                updated = {
                    "state": next_state, "hidden": hidden, "signature": signature, "value": value,
                    "base_valid": c["base_valid"] | take_base,
                    "base_sig": jnp.where(take_base, signature[reporter], c["base_sig"]),
                    "base_val": jnp.where(take_base, value[reporter], c["base_val"]),
                    "finder_id": finder, "discovery_step": discovery_step,
                    "finder_step": jnp.where(take_finder, state.step, c["finder_step"]),
                    "base_step": jnp.where(take_base, state.step, c["base_step"]),
                    "reporter_id": jnp.where(take_base, reporter, c["reporter_id"]),
                    "finder_gru": jnp.where(take_finder, hidden[safe_finder], c["finder_gru"]),
                    "finder_message": jnp.where(take_finder, jnp.concatenate((signature[safe_finder], value[safe_finder])), c["finder_message"]),
                    "finder_position": jnp.where(take_finder, state.pos[safe_finder], c["finder_position"]),
                    "reporter_position": jnp.where(take_base, state.pos[reporter], c["reporter_position"]),
                    "finder_active_agents": jnp.where(take_finder, jnp.sum(state.active, dtype=jnp.int32), c["finder_active_agents"]),
                    "base_active_agents": jnp.where(take_base, jnp.sum(state.active, dtype=jnp.int32), c["base_active_agents"]),
                }
                if history_stride is not None:
                    # Use the actor's existing context helper with its exact reset and
                    # inactive masking semantics; no reconstruction of attention rules.
                    prev_h = jnp.where(resets[:, None], 0.0, c["hidden"])
                    prev_sig = jnp.where(resets[:, None], 0.0, c["signature"])
                    prev_val = jnp.where(resets[:, None], 0.0, c["value"])
                    if getattr(model.actor, "mask_inactive", False):
                        prev_h = jnp.where(state.active[:, None], prev_h, 0.0)
                        prev_sig = jnp.where(state.active[:, None], prev_sig, 0.0)
                        prev_val = jnp.where(state.active[:, None], prev_val, 0.0)
                        obs = jnp.where(state.active[:, None], obs, 0.0)
                    context = model.actor._tarmac_context(
                        prev_h, prev_sig, prev_val, comm & share, state.active,
                        c["base_sig"], c["base_val"], base_mask,
                    )
                    def record(_):
                        # Always include capture even when it is off the sampling grid.
                        slot = (state.step + history_stride - 1) // history_stride
                        return {
                            "history_steps": c["history_steps"].at[slot].set(state.step),
                            "history_obs": c["history_obs"].at[slot].set(obs),
                            "history_encoded": c["history_encoded"].at[slot].set(model.actor.encoder(obs)),
                            "history_context": c["history_context"].at[slot].set(context),
                            "history_flags": c["history_flags"].at[slot].set(
                                jnp.stack((state.active, resets), axis=-1)),
                        }
                    names = ("history_steps", "history_obs", "history_encoded", "history_context", "history_flags")
                    record_now = (jnp.bool_(True) if own_stride is not None else
                                  (c["finder_step"] < 0) & ((state.step % history_stride == 0) | take_finder))
                    updated.update(jax.lax.cond(record_now, record, lambda _: {k: c[k] for k in names}, None))
                if own_stride is not None:
                    names = ("sample_valid", "sample_position", "sample_gru", "sample_message", "sample_found")
                    def record_sample(_):
                        slot = state.step // own_stride
                        return {
                            "sample_valid": c["sample_valid"].at[slot].set(state.active),
                            "sample_position": c["sample_position"].at[slot].set(state.pos),
                            "sample_gru": c["sample_gru"].at[slot].set(hidden),
                            "sample_message": c["sample_message"].at[slot].set(jnp.concatenate((signature, value), axis=-1)),
                            "sample_found": c["sample_found"].at[slot].set(finder >= 0),
                        }
                    updated.update(jax.lax.cond(state.step % own_stride == 0, record_sample,
                                               lambda _: {k: c[k] for k in names}, None))
                return updated

            result = jax.lax.while_loop(lambda c: ~c["state"].done & (c["state"].step < cfg.max_steps)
                & ((c["finder_step"] < 0) if stop_after_finder else jnp.bool_(True)), advance, initial)
            state = result.pop("state")
            output = {name: result[name] for name in (
                "finder_gru", "finder_message", "finder_id", "discovery_step", "finder_step",
                "base_step", "reporter_id", "finder_position", "reporter_position",
                "finder_active_agents", "base_active_agents",
            )} | {
                "base_message": jnp.concatenate((result["base_sig"], result["base_val"])),
                "target_position": state.target_pos, "base_position": state.base_pos,
                "success": state.success, "episode_length": state.step,
                "delivered": state.base_target_known,
                "target_required_agents": state.target_required_agents,
                "spawn_pair_attempts": state.spawn_pair_attempts,
                "complete_sample": (result["finder_step"] >= 0) & result["base_valid"],
            }
            if history_stride is not None:
                finder = jnp.maximum(result["finder_id"], 0)
                output["history_steps"] = result["history_steps"]
                for name in ("history_obs", "history_encoded", "history_context", "history_flags"):
                    output[name] = result[name] if own_stride is not None else result[name][:, finder]
            if own_stride is not None:
                output.update({k: v for k, v in result.items() if k.startswith("sample_")})
            return output

        return jax.vmap(one)(keys)

    return collect_batch


def collect_dataset(model, level, config, run_dir, identity):
    import jax
    import jax.numpy as jnp
    from swarmecho.training.train import _validate_evaluation_pairs
    from geo_experiments.audit_observations import Audit

    d = config["dataset"]
    collect = make_collector(level)
    audit_config = config["audit"]
    audit = Audit() if audit_config["enabled"] and audit_config.get("mode", "full") == "full" else None
    audit_target = min(audit_config["samples"], sample_count(d)) if audit is not None else 0
    collect_history = make_collector(level, history_stride=audit_config["every"]) if audit is not None else None
    chunks, evaluation_chunks = [], []
    requested = sample_count(d)
    attempted = accepted = succeeded = missing = delivered = 0
    started = time.monotonic()
    while accepted < requested and attempted < d["max_episodes"]:
        auditing = audit is not None and audit.samples < audit_target
        count = min(d["eval_batch_size"], audit_config["batch_size"] if auditing else d["eval_batch_size"],
                    d["max_episodes"] - attempted)
        ids = np.arange(attempted, attempted + count, dtype=np.uint32)
        keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(d["seed"]), i))(jnp.asarray(ids))
        print(f"[collect] episodes {attempted + 1}–{attempted + count}; matched successes {accepted}/{requested}", flush=True)
        batch = jax.device_get((collect_history if auditing else collect)(model, keys))
        _validate_evaluation_pairs(level, batch["spawn_pair_attempts"])
        eligible = batch["delivered"] if config.get("protocol", {}).get("eligibility") == "delivered" else batch["success"]
        mask = eligible & batch["complete_sample"]
        needed = requested - accepted
        selected = np.flatnonzero(mask)[:needed]
        if auditing:
            for row in selected[:audit_target - audit.samples]:
                audit.add(batch, int(row), int(ids[row]), level.building.cell_size_m)
        chunks.append({**{name: np.asarray(value)[selected] for name, value in batch.items()
                          if not name.startswith("history_")},
                       "episode_id": ids[selected]})
        evaluation_chunks.append({"episode_id": ids, **{name: np.asarray(batch[name]) for name in
                                  ("success", "delivered", "target_required_agents", "complete_sample", "episode_length", "target_position")}})
        succeeded += int(batch["success"].sum())
        delivered += int(batch["delivered"].sum())
        missing += int((batch["success"] & ~batch["complete_sample"]).sum())
        attempted += count
        accepted += len(selected)
        update_run(run_dir, progress={"stage": "collection", "attempted": attempted, "accepted": accepted,
                   "successful_episodes": succeeded, "successful_missing_snapshot": missing,
                   "requested": requested, "elapsed_seconds": time.monotonic() - started})
    dataset = {name: np.concatenate([c[name] for c in chunks]) for name in chunks[0]}
    episodes = {name: np.concatenate([c[name] for c in evaluation_chunks]) for name in evaluation_chunks[0]}
    metadata = {"num_samples": accepted,
                "attempted_episodes": attempted, "successful_episodes": succeeded,
                "delivered_episodes": delivered,
                "eligibility": config.get("protocol", {}).get("eligibility", "success"),
                "success_rate": succeeded / attempted, "successful_missing_snapshot": missing,
                "collection_seconds": time.monotonic() - started,
                "input_dimensions": {s: int(dataset[s].shape[1]) for s in SOURCES},
                "message_layout": "signature followed by value",
                "finder_rule": "first direct visual sighting; smallest agent index breaks simultaneous ties",
                "snapshot_rule": "updated actor GRU/current emitted token on first communication tick after discovery",
                "base_rule": "first current emitted token stored by base, potentially from a relay"}
    audit_result = None
    if audit_config["enabled"] and audit_config.get("mode") == "conditional" and accepted == requested:
        from geo_experiments.auditing import audit_targets
        temporary = {}
        temporary["geometry/cell_size_m"] = np.asarray(level.building.cell_size_m)
        add_group(temporary, "dataset", {**dataset, "map_id": np.zeros(accepted, dtype=np.int32)})
        summary = audit_targets(model, [level], config, temporary)
        audit_result = (summary, {**group(temporary, "audit"),
                        **{"pretest/" + k: v for k, v in group(temporary, "audit_pretest").items()},
                        **{"collisions/" + k: v for k, v in group(temporary, "audit_collisions").items()}})
    if audit is not None:
        summary = audit.summary()
        summary.update(settings={"samples": audit_target, "every": audit_config["every"],
                                 "batch_size": min(d["eval_batch_size"], audit_config["batch_size"])},
                       source={"kind": "integrated", "identity": identity,
                               "scope": "first matched episodes of this dataset"})
        audit_result = (summary, audit.arrays())
        print(f"[audit] {audit.samples} samples; {summary['distinct_grus']} exact GRUs; "
              f"{summary['mixed_cell_groups']} mixed-cell groups", flush=True)
    if accepted != requested:
        arrays = {}
        add_group(arrays, "dataset", dataset)
        add_group(arrays, "episodes", episodes)
        if audit_result is not None:
            add_group(arrays, "audit", audit_result[1])
        save_data(run_dir, arrays)
        update_run(run_dir, collection=metadata, ambiguity_audit=audit_result[0] if audit_result else None)
        raise RuntimeError(f"Collected only {accepted}/{requested} matched successful episodes in {attempted} attempts. "
                           "Partial data saved in data.npz; increase max_episodes or inspect policy success.")
    return dataset, metadata, episodes, audit_result


def reuse_dataset(path, config, run_dir, identity):
    import json

    path = resolve_path(path)
    manifest = json.loads((path.parent / "run.json").read_text(encoding="utf-8"))
    if manifest["provenance"]["identity"] != identity:
        raise ValueError("Cached dataset checkpoint/level/map/collection identity does not match this run.")
    arrays = archive(path)
    dataset, episodes = group(arrays, "dataset"), group(arrays, "episodes")
    if len(dataset["episode_id"]) != sample_count(config["dataset"]):
        raise ValueError("Cached dataset size must match train + validation + test samples.")
    eligible = dataset["delivered"] if config.get("protocol", {}).get("eligibility") == "delivered" else dataset["success"]
    if not np.all(eligible & dataset["complete_sample"]):
        raise ValueError("Cached dataset contains failed episodes or missing event snapshots.")
    metadata = dict(manifest["collection"])
    metadata["reused_from"] = str(path)
    audit_result = None
    if config["audit"]["enabled"]:
        summary = manifest.get("ambiguity_audit")
        wanted = min(config["audit"]["samples"], sample_count(config["dataset"]))
        if config["audit"].get("mode") == "conditional":
            if summary and summary.get("format") == "swarmecho-target-audit/v2":
                from geo_experiments.auditing import audit_is_current
                if not audit_is_current(summary, config, ("map_specific",)):
                    raise ValueError("Cached collection has a different source-specific audit pipeline/settings.")
                return dataset, metadata, episodes, (summary, group(arrays, "audit"))
            if (not summary or summary.get("format") != "swarmecho-conditional-audit/v1"
                    or summary["pretest"]["samples"] != wanted
                    or manifest["config"]["audit"] != config["audit"]):
                raise ValueError("Cached collection has no matching conditional latent screen.")
            return dataset, metadata, episodes, (summary, group(arrays, "audit"))
        if (not summary or summary.get("source", {}).get("kind") != "integrated"
                or summary.get("samples") != wanted
                or summary.get("settings", {}).get("every") != config["audit"]["every"]
                or not group(arrays, "audit")):
            raise ValueError("Cached data has no matching integrated ambiguity audit. "
                             "Collect fresh data with dataset.path=null, or set audit.enabled=false.")
        audit_result = (summary, group(arrays, "audit"))
    return dataset, metadata, episodes, audit_result
