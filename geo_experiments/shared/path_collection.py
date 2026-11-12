"""Mixed episode/agent event acquisition through the one shared rollout engine."""
from dataclasses import replace
import gc
import json
import time

import numpy as np

from geo_experiments.shared.artifacts import add_group, save_data, update_run
from geo_experiments.shared.collector import map_budgets
from geo_experiments.shared.common import split_episodes, uses_bank_maps
from geo_experiments.shared.console import terminal_print
from geo_experiments.shared.path_geometry import OFFSETS, ReferencePlanner, NoReferenceRoute
from geo_experiments.shared.rollout import make_collector
from geo_experiments.shared.sampling import retained_candidates


def candidate_buffer_bytes(model, level, config, batch_size):
    """Report an estimate, never a cap: rollout/model memory is additional."""
    stride = config["definition"]["inputs"]["all_knowing_till_end"]["every_steps"]
    slots = (level.env.max_steps + stride - 1) // stride + 1
    bytes_per_lane = slots * level.env.num_agents * (model.hidden_dim * 4 + 4)
    return int(batch_size * bytes_per_lane)


def is_allocation_failure(error):
    """Only retry memory failures; unrelated JAX/runtime errors must propagate."""
    if isinstance(error, MemoryError):
        return True
    message = str(error).lower()
    return any(marker in message for marker in (
        "out of memory", "cuda_error_out_of_memory", "cudaerrormemoryallocation",
        "cublas_status_alloc_failed", "cudnn_status_alloc_failed", "hiperroroutofmemory",
        "std::bad_alloc")) or (
            ("resource_exhausted" in message or "resource exhausted" in message)
            and ("allocat" in message or "memory" in message))


def input_rows(batch, lane, config, episode):
    """Return independent readout rows; an absent base never removes GRU rows."""
    rows = {name: [] for name in config["definition"]["inputs"]}

    def add(name, feature, agent, step, first):
        if step >= 0:
            rows[name].append(dict(features=np.array(feature, copy=True), episode=episode, agent=agent,
                                   step=step, informed_step=first))

    finder = int(batch["finder_id"][lane])
    firsts = batch["informed_first"][lane]
    first_finder = int(firsts[finder]) if finder >= 0 else -1
    for name in ("finder_gru", "finder_message"):
        add(name, batch[name][lane], finder, int(batch["finder_step"][lane]), first_finder)
    add("base", batch["base_message"][lane], -1, int(batch["base_step"][lane]), -1)
    cap = config["definition"]["sampling"]["repeated_snapshot_cap_per_agent"]
    steps = batch["candidate_steps"][lane]
    for agent in range(steps.shape[1]):
        chosen = retained_candidates(steps[:, agent], cap)
        if not len(chosen):
            continue
        first = int(firsts[agent])
        add("all_knowing_after_info", batch["candidate_gru"][lane, 0, agent], agent,
            int(steps[0, agent]), first)
        for slot in chosen:
            add("all_knowing_till_end", batch["candidate_gru"][lane, slot, agent], agent,
                int(steps[slot, agent]), first)
    return rows


def collect_path_samples(model, levels, config, run_dir, _graph=None):
    import jax
    import jax.numpy as jnp
    from swarmecho.training.train import _validate_evaluation_pairs

    started = time.monotonic()
    budgets = map_budgets(config, len(levels))
    quotas = np.array([sum(b[f"{s}_samples"] for s in ("train", "validation", "test")) for b in budgets])
    planners = [ReferencePlanner(level.building, level.env,
        (level.map_paths[0].parent.parent / "records" / f"{level.map_paths[0].stem}.npz")
        if uses_bank_maps(config) else None) for level in levels]
    names = tuple(config["definition"]["inputs"])
    inputs = {name: [] for name in names}
    row_counts = {name: 0 for name in names}
    episodes, routes = [], []
    attempted = np.zeros(len(levels), dtype=int)
    accepted = np.zeros(len(levels), dtype=int)
    rejected_routes = np.zeros(len(levels), dtype=int)
    pass_id = time.time_ns()
    stride = config["definition"]["inputs"]["all_knowing_till_end"]["every_steps"]
    requested_size = config["dataset"]["eval_batch_size"]
    # Never allocate lanes beyond the entire collection's possible work. After
    # this, keep one stable JIT shape until OOM requires a smaller one.
    available = int(quotas.sum()) if uses_bank_maps(config) else int(quotas.max())
    size = initial_size = min(requested_size, available, config["dataset"]["max_episodes"])
    oom_reductions = []
    terminal_print(f"[collect] requested capture batch={requested_size}; starting batch={size}; "
                   f"candidate buffer estimate={candidate_buffer_bytes(model, levels[0], config, size) / 1024**2:.1f} MiB "
                   "(other rollout memory additional); allocation failure halves the batch", flush=True)
    if "capture_memory_mib" in config["dataset"]:
        terminal_print("[collect] legacy capture_memory_mib is ignored; batch size now adapts on allocation failure", flush=True)
    if uses_bank_maps(config):
        from geo_experiments.shared.validation_maps import make_validation_environment
        bank_level = replace(levels[0], map_paths=tuple(l.map_paths[0] for l in levels),
                             map_names=[l.building_name for l in levels])
        bank = make_collector(bank_level, environment=make_validation_environment(bank_level, levels),
                              agent_capture={"every_steps": stride})
    else:
        collectors = [make_collector(l, agent_capture={"every_steps": stride}) for l in levels]
    root = jax.random.PRNGKey(config["dataset"]["seed"])
    keys_for = jax.jit(jax.vmap(lambda m, i: jax.random.fold_in(jax.random.fold_in(root, m), i)))
    timings = dict(rollout_and_transfer=0., reference_routes=0., row_assembly=0., oom_recovery=0.)
    while np.any(accepted < quotas) and attempted.sum() < config["dataset"]["max_episodes"]:
        count = min(size, config["dataset"]["max_episodes"] - int(attempted.sum()))
        maps = np.repeat(np.arange(len(levels)), quotas - accepted)[:count].astype(np.int32)
        if not uses_bank_maps(config):
            maps = maps[maps == maps[0]]
        count = len(maps)
        ids = np.empty(count, dtype=np.uint32)
        for m in np.unique(maps):
            slots = np.flatnonzero(maps == m)
            ids[slots] = np.arange(attempted[m], attempted[m] + len(slots))
        rollout_started = time.monotonic()
        raw = batch = keys = runtime_maps = runtime_ids = None
        allocation_error = None
        try:
            runtime_maps = jnp.asarray(np.pad(maps, (0, size - count), mode="edge"))
            runtime_ids = jnp.asarray(np.pad(ids, (0, size - count), mode="edge"))
            keys = keys_for(runtime_maps, runtime_ids)
            raw = bank(model, keys, runtime_maps) if uses_bank_maps(config) else collectors[int(maps[0])](model, keys)
            # Synchronize here: asynchronous device allocation errors must be
            # caught before we increment attempts or append any accepted rows.
            batch = {key: np.asarray(value)[:count] for key, value in jax.device_get(raw).items()}
        except (RuntimeError, MemoryError) as error:
            if not is_allocation_failure(error) or size == 1:
                raise
            allocation_error = str(error)
        if allocation_error is not None:
            # Leave the exception handler first so its traceback no longer pins
            # failed device buffers. Drop outputs and oversized compiled shapes;
            # the frozen model, map bank and previously accepted CPU rows survive.
            raw = batch = keys = runtime_maps = runtime_ids = None
            gc.collect()
            jax.clear_caches()
            previous_size, size = size, max(1, size // 2)
            elapsed = time.monotonic() - rollout_started
            timings["oom_recovery"] += elapsed
            oom_reductions.append(dict(from_batch=previous_size, to_batch=size,
                attempted_episodes=int(attempted.sum()), error=allocation_error, seconds=elapsed))
            terminal_print(f"[collect] allocation failure at capture batch={previous_size}; retrying batch={size}; "
                           "episode counters unchanged", flush=True)
            update_run(run_dir, progress={"stage": "collection", "accepted": int(accepted.sum()),
                "attempted": int(attempted.sum()), "capture_batch_size": int(size), "oom_reductions": oom_reductions})
            continue
        # GPU outputs are no longer needed once transfer has succeeded.
        raw = keys = runtime_maps = runtime_ids = None
        rollout_seconds = time.monotonic() - rollout_started
        timings["rollout_and_transfer"] += rollout_seconds
        route_seconds = row_seconds = 0.
        for lane, m in enumerate(maps):
            attempted[m] += 1
            _validate_evaluation_pairs(levels[m], batch["spawn_pair_attempts"][lane:lane + 1])
            if not np.any(batch["informed_first"][lane] >= 0):
                continue
            # Every accepted episode has at least one informed-agent sample;
            # successful delivery is deliberately not required.
            episode = len(episodes)
            route_started = time.monotonic()
            try:
                route = planners[m].route(batch["base_position"][lane], batch["target_position"][lane])
            except ValueError as error:
                diagnostic = dict(collection_pass=pass_id, map_id=int(m), map_name=levels[m].building_name,
                    episode_id=int(ids[lane]), seed=config["dataset"]["seed"],
                    base_position=batch["base_position"][lane].tolist(), target_position=batch["target_position"][lane].tolist(),
                    required_agents=int(batch["target_required_agents"][lane]), error=str(error),
                    reason="no_finite_reference" if isinstance(error, NoReferenceRoute) else "route_construction_error")
                if isinstance(error, NoReferenceRoute):
                    diagnostic.update(base_attachments=error.base_attachments, target_attachments=error.target_attachments)
                with (run_dir / "reference_route_failures.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(diagnostic) + "\n")
                route_seconds += time.monotonic() - route_started
                if not isinstance(error, NoReferenceRoute):
                    raise ValueError(f"Route construction failed on map {int(m)} ({levels[m].building_name}), "
                                     f"episode {int(ids[lane])}; coordinates saved in reference_route_failures.jsonl") from error
                rejected_routes[m] += 1
                terminal_print(f"[collect] replacing unlabelable episode: map={int(m)} episode={int(ids[lane])}; "
                               f"no finite reference path ({rejected_routes.sum()} total)", flush=True)
                continue
            route_seconds += time.monotonic() - route_started
            rows_started = time.monotonic()
            rows = input_rows(batch, lane, config, episode)
            for name in names:
                # Compact each episode immediately. Keeping millions of row
                # dictionaries or views into whole candidate buffers would
                # dwarf the retained features' actual storage requirements.
                if rows[name]:
                    inputs[name].append({"features": np.stack([r["features"] for r in rows[name]]),
                        **{key: np.array([r[key] for r in rows[name]], dtype=np.int32)
                           for key in ("episode", "agent", "step", "informed_step")}})
                    row_counts[name] += len(rows[name])
            episodes.append(dict(map_id=m, episode_id=ids[lane], target_position=batch["target_position"][lane],
                                 base_position=batch["base_position"][lane], required_agents=batch["target_required_agents"][lane],
                                 success=batch["success"][lane], delivered=batch["delivered"][lane],
                                 length=batch["episode_length"][lane], rotation=route["rotation"], anchor=route["anchor"],
                                 reference_length=route["length"], candidate_count=int((batch["candidate_steps"][lane] >= 0).sum())))
            routes.append(route)
            accepted[m] += 1
            row_seconds += time.monotonic() - rows_started
        timings["reference_routes"] += route_seconds
        timings["row_assembly"] += row_seconds
        terminal_print(f"[collect] path episodes {accepted.sum():,}/{quotas.sum():,}; attempts {attempted.sum():,}; capture batch={size}; "
                       f"rollout+copy={rollout_seconds:.2f}s; routes={route_seconds:.2f}s; rows={row_seconds:.2f}s", flush=True)
        update_run(run_dir, progress={"stage": "collection", "accepted": int(accepted.sum()), "attempted": int(attempted.sum()),
                                     "reference_route_rejections": int(rejected_routes.sum()),
                                     "capture_batch_size": int(size), "oom_reductions": oom_reductions})
        # Do not keep the preceding batch's candidate buffers alive while the
        # next one is allocated. Retained features were copied into CPU chunks.
        del batch
    arrays = {}
    if not episodes:
        raise RuntimeError("No informed-agent episodes with finite reward-reference routes were collected within max_episodes. "
                           "See reference_route_failures.jsonl if route labels were unavailable.")
    add_group(arrays, "episodes", {key: np.asarray([e[key] for e in episodes]) for key in episodes[0]})
    add_group(arrays, "labels", {"path": np.stack([r["mask"] for r in routes])})
    add_group(arrays, "paths", {"points": np.concatenate([r["points"] for r in routes]),
        "offsets": np.cumsum([0, *[len(r["points"]) for r in routes]]).astype(np.int32)})
    add_group(arrays, "geometry", {"cell_offsets": OFFSETS, "map_names": np.array([l.building_name for l in levels]),
                                   "cell_size_m": np.array([p.cell_size for p in planners])})
    for m, planner in enumerate(planners):
        add_group(arrays, f"maps/{m}", {"solid_min": planner.lower, "solid_max": planner.upper, "cells": planner.cells})
        from swarmecho.env.buildings import BuildingArrays, building_snapshot
        if isinstance(levels[m].building, BuildingArrays):
            add_group(arrays, f"maps/{m}", {"building_snapshot": np.array(json.dumps(building_snapshot(levels[m].building)))})
    dimensions = {"finder_gru": model.hidden_dim, "finder_message": model.tarmac_sig_dim + model.tarmac_val_dim,
                  "base": model.tarmac_sig_dim + model.tarmac_val_dim,
                  "all_knowing_after_info": model.hidden_dim, "all_knowing_till_end": model.hidden_dim}
    for name, chunks in inputs.items():
        values = {"features": np.concatenate([r["features"] for r in chunks]) if chunks else np.empty((0, dimensions[name]), np.float32)}
        values.update({key: np.concatenate([r[key] for r in chunks]) if chunks else np.empty(0, dtype=np.int32)
                       for key in ("episode", "agent", "step", "informed_step")})
        add_group(arrays, f"inputs/{name}", values)
        chunks.clear()
    metadata = dict(num_samples=len(episodes), attempted_episodes=int(attempted.sum()),
        input_rows=row_counts,
        per_map=[dict(name=l.building_name, accepted=int(accepted[m]), attempted=int(attempted[m]),
                      reference_route_rejections=int(rejected_routes[m])) for m, l in enumerate(levels)],
        reference_route_rejections=int(rejected_routes.sum()),
        reference_route_failure_log="reference_route_failures.jsonl" if rejected_routes.sum() else None,
        collection_seconds=time.monotonic() - started, capture_batch_size=int(size),
        requested_capture_batch_size=int(requested_size), initial_capture_batch_size=int(initial_size),
        oom_reductions=oom_reductions,
        timing_seconds=timings,
        split_rule="Exact per-map episode budgets; all agents/steps/inputs from an episode share one split",
        eligibility="At least one informed active agent and a finite reward-reference route; eventual failure/delivery do not filter GRUs")
    save_data(run_dir, arrays)
    update_run(run_dir, collection=metadata)
    if np.any(accepted != quotas):
        raise RuntimeError("Path collection hit max_episodes before all map quotas were filled; partial arrays saved.")
    splits = {s: [] for s in ("train", "validation", "test")}
    for m, budget in enumerate(budgets):
        rows = np.flatnonzero(arrays["episodes/map_id"] == m)
        for split, indices in split_episodes(arrays["episodes/required_agents"][rows], budget, require_class_coverage=False).items():
            splits[split].extend(rows[indices])
    add_group(arrays, "splits", {key: np.sort(rows).astype(np.int32) for key, rows in splits.items()})
    return arrays, metadata
