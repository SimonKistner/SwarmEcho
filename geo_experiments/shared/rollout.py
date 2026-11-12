"""Capture matched discovery/communication events without policy optimization."""
from __future__ import annotations

import hashlib
import time

import numpy as np

from geo_experiments.shared.common import ROOT, fingerprint, resolve_path, sample_count, uses_bank_maps
from geo_experiments.shared.artifacts import add_group, archive, group, save_data, update_run
from geo_experiments.shared.console import terminal_print as print


COLLECTION_FILES = {
    "collection.py": "shared/rollout.py",
    "target_collection.py": "shared/collector.py",
    "labels.py": "shared/cell_labels.py",
    "auditing.py": "shared/diagnostics/auditing.py",
    "policy.py": "shared/policy.py",
    "common.py": "shared/definitions.py",
    "geometry.py": "shared/geometry.py",
    "validation_maps.py": "shared/validation_maps.py",
}


def collection_fingerprint(name):
    path = ROOT / "geo_experiments" / COLLECTION_FILES[name]
    if name == "common.py":
        # The old common module is now two files; both affect sample selection.
        common = ROOT / "geo_experiments/shared/common.py"
        return hashlib.sha256(path.read_bytes() + common.read_bytes()).hexdigest()
    return fingerprint(path)

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
    from geo_experiments.shared.policy import load_probe_level
    configured_level, source, original, ignored = load_probe_level(config, checkpoint, run_dir)
    validation_maps = uses_bank_maps(config)
    if validation_maps and not allow_suite:
        raise ValueError("Validation maps require a modern target probe.")
    level = configured_level if validation_maps else resolve_evaluation_level(configured_level)
    if allow_suite and not validation_maps and configured_level.evaluation.eval_differes_from_training_map:
        raise ValueError("Suite probes use the authored training map list; separate evaluation-map substitution is unsupported.")
    if ((configured_level.random_buildings.enabled and not validation_maps) or (configured_level.static_map_suite and not allow_suite)
            or (configured_level.evaluation.random_eval and not validation_maps) or level.env.num_obstacles
            or (level.env.adapt_size.enabled and not allow_suite)
            or (level.training.randomize_base and config.get("experiment") != "path")):
        raise ValueError("Phase 0 supports one authored static map, fixed base, fixed population, no random obstacles.")
    if level.env.success_condition == "coverage" or not level.network.memory_comm_enabled:
        raise ValueError("Target probes require a target mission with learned communication enabled.")
    map_path = (resolve_map_file(configured_level.evaluation.eval_map, MAP_DIR).resolve()
                if not validation_maps and configured_level.evaluation.eval_differes_from_training_map
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
        if validation_maps:
            identity["map_source"] = config["policy"]["map_source"]
        identity["maps"] = [{"name": name, "sha256": fingerprint(path)}
                            for name, path in zip(configured_level.map_names, configured_level.map_paths, strict=True)]
        identity["probe_collection_files"] = {name: collection_fingerprint(name)
            for name in ("collection.py", "target_collection.py", "labels.py", "auditing.py", "policy.py", "common.py", "geometry.py",
                         *(("validation_maps.py",) if validation_maps else ())) }
    if "definition" in config:
        request = {key: config["definition"][key] for key in ("target", "inputs", "output", "sampling")}
        import json
        identity["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        identity["definition_sources"] = {name: fingerprint(ROOT / "geo_experiments/shared" / name)
            for name in ("definitions.py", "inputs.py", "targets.py", "collector.py", "cell_labels.py",
                         "path_geometry.py", "path_collection.py", "sampling.py")}
    if run_dir is not None:
        update_run(run_dir, provenance={"identity": identity, "checkpoint": str(checkpoint),
                   "level": level.name, "level_source": str(source), "map_source": str(map_path),
                   "runtime_files": runtime_files,
                   "experiment_files": {p.relative_to(ROOT / "geo_experiments").as_posix(): fingerprint(p)
                                        for p in (ROOT / "geo_experiments").rglob("*.py") if "outputs" not in p.parts}},
                   definitions={"level.yaml": original, "ignored_legacy_reward_payouts": ignored,
                                **{f"maps/{i}.yaml": path.read_text(encoding="utf-8")
                                   for i, path in enumerate(configured_level.map_paths)},
                                "map.yaml": map_path.read_text(encoding="utf-8")})
    return model, level, identity


def make_collector(level, *, requested_inputs=None, history_stride=None, own_stride=None, stop_after_finder=False, environment=None, agent_capture=None, knowledge_capture=False):
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

    reset, step, observations, _ = _evaluation_env_fns(level) if environment is None else environment
    cfg = level.env
    if agent_capture is not None:
        candidate_stride = agent_capture["every_steps"]
        candidate_slots = (cfg.max_steps + candidate_stride - 1) // candidate_stride + 1

    @nnx.jit
    def collect_batch(model, keys, map_ids=None):
        def one(key, map_id=None):
            state = reset(key) if map_id is None else reset(key, map_id=map_id)
            n = cfg.num_agents
            initial = {
                "state": state, "hidden": model.initial_actor_hidden((), num_agents=n),
                "signature": model.initial_actor_signature((), num_agents=n),
                "value": model.initial_actor_value((), num_agents=n),
                "prev_path_probe": model.initial_path_probe(num_agents=n),
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
            if knowledge_capture:
                initial.update({
                    # First-acquisition source bits: vision=1, base=2, peer=4.
                    # Keep overlaps; receiving a learned base token is a separate
                    # observation from the environment's target-known flag.
                    "knowledge_origin": jnp.where(state.active & state.target_known, 1, 0),
                    "knowledge_counted": jnp.zeros(n, dtype=bool),
                    "knowledge_had_base_token": jnp.zeros(n, dtype=bool),
                    "knowledge_first_counts": jnp.zeros(8, dtype=jnp.int32),
                    "knowledge_first_base_token_counts": jnp.zeros(8, dtype=jnp.int32),
                    "knowledge_informed_steps": jnp.zeros(8, dtype=jnp.int32),
                    "knowledge_base_history_steps": jnp.int32(0),
                    "knowledge_unattributed_events": jnp.int32(0),
                })
            if agent_capture is not None:
                initial.update({
                    "informed_first": jnp.full(n, -1, dtype=jnp.int32),
                    "informed_closed": jnp.zeros(n, dtype=bool),
                    "candidate_gru": jnp.zeros((candidate_slots, n, model.hidden_dim)),
                    "candidate_steps": jnp.full((candidate_slots, n), -1, dtype=jnp.int32),
                })
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
                resets = policy_resets(state)
                obs = model.actor_observations(
                    observations(state), c["prev_path_probe"], state.active, state.target_known, resets,
                )
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
                next_path_probe = model.next_path_probe(
                    hidden, state.active & next_state.active & ~next_state.done, state.target_known,
                )
                updated = {
                    "state": next_state, "hidden": hidden, "signature": signature, "value": value,
                    "prev_path_probe": next_path_probe,
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
                if knowledge_capture:
                    eligible = state.active & state.target_known
                    first = eligible & ~c["knowledge_counted"]
                    had_base = c["knowledge_had_base_token"] | base_mask
                    histogram = lambda mask: jnp.bincount(c["knowledge_origin"],
                        weights=mask.astype(jnp.int32), length=8).astype(jnp.int32)
                    # The environment sends knowledge synchronously using these
                    # same pre-action links; direct sight uses post-motion state.
                    peer_source = jnp.any(comm & share & state.target_known[None, :], axis=1)
                    base_source = base_range & share & state.base_target_known
                    new = next_state.active & next_state.target_known & ~state.target_known
                    bits = (next_state.directly_sees_target.astype(jnp.int32)
                            | (base_source.astype(jnp.int32) << 1)
                            | (peer_source.astype(jnp.int32) << 2))
                    origin = jnp.where(new, bits, c["knowledge_origin"])
                    reset_lifetime = ~next_state.active
                    if next_state.adaptive is not None:
                        reset_lifetime |= next_state.adaptive.spawned
                    updated.update({
                        "knowledge_origin": jnp.where(reset_lifetime, 0, origin),
                        "knowledge_counted": (c["knowledge_counted"] | eligible) & ~reset_lifetime,
                        "knowledge_had_base_token": had_base & ~reset_lifetime,
                        "knowledge_first_counts": c["knowledge_first_counts"] + histogram(first),
                        "knowledge_first_base_token_counts": c["knowledge_first_base_token_counts"] + histogram(first & had_base),
                        "knowledge_informed_steps": c["knowledge_informed_steps"] + histogram(eligible),
                        "knowledge_base_history_steps": c["knowledge_base_history_steps"] + jnp.sum(eligible & had_base, dtype=jnp.int32),
                        "knowledge_unattributed_events": c["knowledge_unattributed_events"] + jnp.sum(new & (bits == 0), dtype=jnp.int32),
                    })
                if agent_capture is not None:
                    # state.target_known is observed BEFORE this actor update;
                    # hidden has now processed that information. Never pair a
                    # post-transition knowledge flag with the preceding GRU.
                    informed = state.target_known & state.active & ~c["informed_closed"]
                    first = jnp.where(informed & (c["informed_first"] < 0), state.step, c["informed_first"])
                    age = state.step - first
                    take = informed & (first >= 0) & (age % candidate_stride == 0)
                    slots = jnp.clip(age // candidate_stride, 0, candidate_slots - 1)
                    agents = jnp.arange(n)
                    updated.update({
                        "informed_first": first,
                        "informed_closed": c["informed_closed"] | ((first >= 0) & ~state.active),
                        "candidate_gru": c["candidate_gru"].at[slots, agents].set(
                            jnp.where(take[:, None], hidden, c["candidate_gru"][slots, agents])),
                        "candidate_steps": c["candidate_steps"].at[slots, agents].set(
                            jnp.where(take, state.step, c["candidate_steps"][slots, agents])),
                    })
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
            if agent_capture is not None:
                output.update({name: result[name] for name in ("informed_first", "candidate_gru", "candidate_steps")})
            if history_stride is not None:
                finder = jnp.maximum(result["finder_id"], 0)
                output["history_steps"] = result["history_steps"]
                for name in ("history_obs", "history_encoded", "history_context", "history_flags"):
                    output[name] = result[name] if own_stride is not None else result[name][:, finder]
            if own_stride is not None:
                output.update({k: v for k, v in result.items() if k.startswith("sample_")})
            if knowledge_capture:
                output.update({key: result[key] for key in (
                    "knowledge_first_counts", "knowledge_first_base_token_counts", "knowledge_informed_steps",
                    "knowledge_base_history_steps", "knowledge_unattributed_events")})
            if requested_inputs is not None:
                from geo_experiments.shared.inputs import project_event_inputs
                output = project_event_inputs(output, requested_inputs)
            return output

        return jax.vmap(one)(keys) if map_ids is None else jax.vmap(one)(keys, map_ids)

    return collect_batch


