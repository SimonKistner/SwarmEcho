"""One delivered-target collection and common split, reusable by every readout."""
from __future__ import annotations

import time

import numpy as np

from geo_experiments.artifacts import add_group, save_data, update_run
from geo_experiments.collection import make_collector
from geo_experiments.common import SOURCES, split_episodes
from geo_experiments.labels import position_labels
from geo_experiments.console import terminal_print as print


def map_budgets(config, count):
    """Split total budgets evenly across maps, preserving exact split totals."""
    result = []
    for m in range(count):
        budget = {f"{s}_samples": config["dataset"][f"{s}_samples"] // count
                  + int(m < config["dataset"][f"{s}_samples"] % count)
                  for s in ("train", "validation", "test")}
        if min(budget.values()) < 1:
            raise ValueError("Each split budget must provide at least one sample per map.")
        budget["split_seed"] = config["dataset"]["split_seed"] + m
        result.append(budget)
    return result


def episode_keys(seed, map_id, ids):
    import jax
    import jax.numpy as jnp
    key = jax.random.fold_in(jax.random.PRNGKey(seed), map_id)
    return jax.vmap(lambda i: jax.random.fold_in(key, i))(jnp.asarray(ids, dtype=jnp.uint32))


def collect_targets(model, levels, config, run_dir, graph):
    if config["policy"].get("map_source") == "validation":
        return collect_validation_targets(model, levels, config, run_dir, graph)
    import jax
    from swarmecho.training.train import _validate_evaluation_pairs

    budgets = map_budgets(config, len(levels))
    chunks, attempts, per_map = [], [], []
    total_attempted = total_kept = 0
    started = time.monotonic()
    for m, (level, budget) in enumerate(zip(levels, budgets, strict=True)):
        requested = sum(budget[f"{s}_samples"] for s in ("train", "validation", "test"))
        collect = make_collector(level)
        attempted = accepted = delivered = successes = missing = 0
        while accepted < requested and total_attempted < config["dataset"]["max_episodes"]:
            size = min(config["dataset"]["eval_batch_size"], config["dataset"]["max_episodes"] - total_attempted)
            ids = np.arange(attempted, attempted + size, dtype=np.uint32)
            print(f"[collect] map {m + 1}/{len(levels)} {level.building_name}: {accepted}/{requested}; attempts {attempted:,}", flush=True)
            batch = jax.device_get(collect(model, episode_keys(config["dataset"]["seed"], m, ids)))
            _validate_evaluation_pairs(level, batch["spawn_pair_attempts"])
            eligible = batch["delivered"] if config["protocol"]["eligibility"] == "delivered" else batch["success"]
            selected = np.flatnonzero(eligible & batch["complete_sample"])[:requested - accepted]
            chunks.append({**{k: np.asarray(v)[selected] for k, v in batch.items()},
                           "episode_id": ids[selected], "map_id": np.full(len(selected), m, dtype=np.int32)})
            attempts.append({"episode_id": ids, "map_id": np.full(size, m, dtype=np.int32),
                             **{k: np.asarray(batch[k]) for k in ("success", "delivered", "complete_sample",
                                "episode_length", "target_position", "target_required_agents")}})
            delivered += int(batch["delivered"].sum())
            successes += int(batch["success"].sum())
            missing += int((eligible & ~batch["complete_sample"]).sum())
            attempted += size
            total_attempted += size
            accepted += len(selected)
            total_kept += len(selected)
            update_run(run_dir, progress={"stage": "collection", "map_id": m, "accepted": total_kept,
                       "attempted": total_attempted, "map_accepted": accepted, "map_requested": requested})
        per_map.append({"name": level.building_name, "attempted": attempted, "accepted": accepted,
                        "delivered": delivered, "successful": successes, "eligible_missing_snapshot": missing})
        if accepted != requested:
            break
    return finish_collection(chunks, attempts, per_map, budgets, config, run_dir, graph, started)


def collect_validation_targets(model, levels, config, run_dir, graph):
    """Mixed-map batches with exact quotas; no separate JIT per building."""
    import jax
    import jax.numpy as jnp
    from dataclasses import replace
    from geo_experiments.validation_maps import make_validation_environment
    from swarmecho.training.train import _validate_evaluation_pairs

    budgets = map_budgets(config, len(levels))
    quotas = np.asarray([sum(b[f"{s}_samples"] for s in ("train", "validation", "test")) for b in budgets])
    per_map = [dict(name=level.building_name, attempted=0, accepted=0, delivered=0,
                    successful=0, eligible_missing_snapshot=0) for level in levels]
    bank_level = replace(levels[0], map_paths=tuple(item.map_paths[0] for item in levels),
                         map_names=[item.building_name for item in levels])
    collect = make_collector(bank_level, environment=make_validation_environment(bank_level, levels))
    attempted = np.zeros(len(levels), dtype=np.int64)
    accepted = np.zeros(len(levels), dtype=np.int64)
    chunks, attempts = [], []
    started = time.monotonic()
    root = jax.random.PRNGKey(config["dataset"]["seed"])
    keys_for = jax.jit(jax.vmap(lambda m, i: jax.random.fold_in(jax.random.fold_in(root, m), i)))
    batch_size = config["dataset"]["eval_batch_size"]
    while np.any(accepted < quotas) and attempted.sum() < config["dataset"]["max_episodes"]:
        # At most the outstanding quota from any map is attempted in this batch.
        # Small quotas share a batch; a failed episode receives a fresh per-map ID.
        count = min(batch_size, config["dataset"]["max_episodes"] - int(attempted.sum()))
        maps = np.repeat(np.arange(len(levels), dtype=np.int32), quotas - accepted)[:count]
        size = len(maps)
        ids = np.empty(size, dtype=np.uint32)
        for m in np.unique(maps):
            slots = np.flatnonzero(maps == m)
            ids[slots] = np.arange(attempted[m], attempted[m] + len(slots), dtype=np.uint32)
        # Pad only execution lanes; padded cases never enter counters or archives.
        runtime_maps = jnp.asarray(np.pad(maps, (0, batch_size - size), mode="edge"))
        runtime_ids = jnp.asarray(np.pad(ids, (0, batch_size - size), mode="edge"))
        print(f"[collect] validation bank: {accepted.sum():,}/{quotas.sum():,}; "
              f"{np.count_nonzero(accepted == quotas)}/{len(levels)} maps complete; "
              f"attempts {attempted.sum():,}", flush=True)
        batch = {k: np.asarray(v)[:size] for k, v in jax.device_get(
                 collect(model, keys_for(runtime_maps, runtime_ids), runtime_maps)).items()}
        eligible = batch["delivered"] if config["protocol"]["eligibility"] == "delivered" else batch["success"]
        selected = np.flatnonzero(eligible & batch["complete_sample"])
        chunks.append({**{k: v[selected] for k, v in batch.items()},
                       "episode_id": ids[selected], "map_id": maps[selected]})
        attempts.append({"episode_id": ids, "map_id": maps,
                         **{k: batch[k] for k in ("success", "delivered", "complete_sample",
                            "episode_length", "target_position", "target_required_agents")}})
        for m in np.unique(maps):
            mask = maps == m
            _validate_evaluation_pairs(levels[m], batch["spawn_pair_attempts"][mask])
            attempted[m] += int(mask.sum())
            accepted[m] += int((maps[selected] == m).sum())
            row = per_map[m]
            row.update(attempted=int(attempted[m]), accepted=int(accepted[m]))
            row["delivered"] += int(batch["delivered"][mask].sum())
            row["successful"] += int(batch["success"][mask].sum())
            row["eligible_missing_snapshot"] += int((eligible[mask] & ~batch["complete_sample"][mask]).sum())
        update_run(run_dir, progress={"stage": "collection", "accepted": int(accepted.sum()),
                   "attempted": int(attempted.sum()), "maps_complete": int((accepted == quotas).sum()),
                   "total_maps": len(levels)})
    return finish_collection(chunks, attempts, per_map, budgets, config, run_dir, graph, started)


def finish_collection(chunks, attempts, per_map, budgets, config, run_dir, graph, started):
    """The same archive/label contract for authored maps and validation banks."""
    total_kept = sum(row["accepted"] for row in per_map)
    total_attempted = sum(row["attempted"] for row in per_map)
    dataset = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    episodes = {k: np.concatenate([c[k] for c in attempts]) for k in attempts[0]}
    arrays = {}
    for prefix, values in (("dataset", dataset), ("episodes", episodes), ("geometry", graph)):
        add_group(arrays, prefix, values)
    metadata = {"num_samples": total_kept, "attempted_episodes": total_attempted,
                "delivered_episodes": sum(row["delivered"] for row in per_map),
                "successful_episodes": sum(row["successful"] for row in per_map),
                "eligibility": config["protocol"]["eligibility"], "per_map": per_map,
                "collection_seconds": time.monotonic() - started,
                "input_dimensions": {s: int(dataset[s].shape[1]) for s in SOURCES},
                "split_rule": "Exact budgets per map; cell-stratified episode split, identical for every readout",
                "snapshot_rule": "Updated finder GRU/current token at first communication tick after discovery",
                "base_rule": "First current token stored by the base; its sender may be a relay"}
    update_run(run_dir, collection=metadata)
    if config["policy"].get("map_source") == "validation":
        metadata["map_source"] = "validation"
    if len(per_map) != len(budgets) or any(r["accepted"] != sum(b[f"{s}_samples"] for s in
          ("train", "validation", "test")) for r, b in zip(per_map, budgets)):
        save_data(run_dir, arrays)
        raise RuntimeError("Target collection reached dataset.max_episodes before every map quota was filled. Partial data saved.")
    labels = position_labels(dataset, graph)
    splits = {s: [] for s in ("train", "validation", "test")}
    for m, budget in enumerate(budgets):
        rows = np.flatnonzero(dataset["map_id"] == m)
        pooled = config["policy"].get("map_source") == "validation" and config["protocol"]["decoder"] == "shared_cell"
        for split, indices in split_episodes(labels["target_cell"][rows], budget,
                                             require_class_coverage=not pooled).items():
            splits[split].extend(rows[indices])
    if config["policy"].get("map_source") == "validation" and config["protocol"]["decoder"] == "shared_cell":
        if set(labels["coordinate_cell"]) != set(labels["coordinate_cell"][splits["train"]]):
            raise ValueError("Pooled training split is missing observed coordinate classes; increase training samples.")
        metadata["split_rule"] = ("Exact per-map budgets; cell-stratified episode splits with pooled "
                                  "coordinate training coverage. Maps occur in every split; this is not unseen-map transfer.")
        update_run(run_dir, collection=metadata)
    add_group(arrays, "labels", labels)
    add_group(arrays, "splits", {k: np.sort(v).astype(np.int32) for k, v in splits.items()})
    return arrays, metadata
