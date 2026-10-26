"""Whole-episode own-cell snapshots, exact-prefix filtering and episode-pool splits."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
import time

import numpy as np

from geo_experiments.artifacts import add_group, save_data, update_run
from geo_experiments.audit_observations import STAGES, cross_cell_pairs, digest
from geo_experiments.collection import make_collector
from geo_experiments.geometry import target_labels
from geo_experiments.console import terminal_print as print


def episode_samples(batch, row, episode_id, stride, cell_size, lookup):
    """Hash every active input step; return only regular current-position snapshots."""
    n = batch["sample_valid"].shape[-1]
    histories = [[hashlib.sha256() for _ in STAGES] for _ in range(n)]
    valid_steps = batch["history_steps"][row]
    candidates, outside = [], 0
    for t in np.flatnonzero(valid_steps >= 0):
        step = int(valid_steps[t])
        for agent in range(n):
            flags = batch["history_flags"][row, t, agent]
            active, reset = map(bool, flags)
            if reset or not active:
                histories[agent] = [hashlib.sha256() for _ in STAGES]
            if not active:
                continue
            raw = batch["history_obs"][row, t, agent]
            encoded = batch["history_encoded"][row, t, agent]
            context = batch["history_context"][row, t, agent]
            values = (raw, encoded, np.concatenate((encoded, context)))
            for history, value in zip(histories[agent], values, strict=True):
                history.update(digest(value, flags))
            if step % stride:
                continue
            slot = step // stride
            if not batch["sample_valid"][row, slot, agent]:
                raise ValueError("Snapshot and actor-input active masks differ.")
            position = batch["sample_position"][row, slot, agent]
            cell = tuple(np.floor(position / cell_size).astype(int))
            if cell not in lookup:
                outside += 1
                continue
            hashes = np.stack([np.frombuffer(h.digest(), dtype=np.uint8) for h in histories[agent]])
            candidates.append({
                "agent_gru": batch["sample_gru"][row, slot, agent],
                "agent_message": batch["sample_message"][row, slot, agent],
                "own_position": position, "base_position": batch["base_position"][row],
                "episode_id": np.uint32(episode_id), "agent_id": np.int32(agent),
                "step": np.int32(step), "target_found": batch["sample_found"][row, slot],
                "history_sha256": hashes, "cell_id": np.int32(lookup[cell]),
            })
    return candidates, outside


class PrefixFilter:
    """A history belongs to one split; keep different labels within that split."""
    def __init__(self):
        self.owner = {}
        self.seen = set()
        self.duplicates = 0
        self.cross_split = 0

    def accept(self, sample, split):
        history = sample["history_sha256"][2].tobytes()
        owner = self.owner.get(history)
        if owner is not None and owner != split:
            self.cross_split += 1
            return False
        key = (history, int(sample["cell_id"]))
        if key in self.seen:
            self.duplicates += 1
            return False
        self.owner[history] = split
        self.seen.add(key)
        return True


def uniform_cell_limits(budgets, num_cells, seed):
    """Exact split totals, with at most one count difference per split and combined."""
    if num_cells < 1:
        raise ValueError("Uniform buckets require at least one interior cell.")
    order = np.random.default_rng(seed).permutation(num_cells)
    cursor = 0
    limits = {}
    for split, total in budgets.items():
        base, remainder = divmod(total, num_cells)
        row = np.full(num_cells, base, dtype=np.int64)
        row[order[(cursor + np.arange(remainder)) % num_cells]] += 1
        limits[split] = row
        cursor = (cursor + remainder) % num_cells
    return limits


def eligible_cells(cells, excluded):
    lookup = {tuple(map(int, cell)): i for i, cell in enumerate(cells)}
    missing = [tuple(cell) for cell in excluded if tuple(cell) not in lookup]
    if missing:
        raise ValueError(f"sampling.exclude_cells contains cells outside this map's interior: {missing}")
    mask = np.ones(len(cells), dtype=bool)
    for cell in excluded:
        mask[lookup[tuple(cell)]] = False
    if not mask.any():
        raise ValueError("sampling.exclude_cells must leave at least one eligible interior cell.")
    return mask


def eligible_cell_limits(budgets, eligible, seed):
    small = uniform_cell_limits(budgets, int(eligible.sum()), seed)
    limits = {}
    for name, values in small.items():
        limits[name] = np.zeros(len(eligible), dtype=np.int64)
        limits[name][eligible] = values
    return limits


class CellBuckets:
    def __init__(self, num_cells, limits=None):
        self.limits = limits
        self.counts = np.zeros(num_cells, dtype=np.int64)
        self.rejected = np.zeros(num_cells, dtype=np.int64)

    def full(self, cell):
        if self.limits is not None and self.counts[cell] >= self.limits[cell]:
            self.rejected[cell] += 1
            return True
        return False

    def add(self, cell):
        if self.limits is not None and self.counts[cell] >= self.limits[cell]:
            raise ValueError("Cannot add a sample to a full cell bucket.")
        self.counts[cell] += 1


def bucket_source_data(candidate_counts, buckets):
    values = {"candidate_cell_counts": candidate_counts}
    for name, bucket in buckets.items():
        values[f"{name}_retained_cell_counts"] = bucket.counts
        values[f"{name}_full_bucket_rejections"] = bucket.rejected
        if bucket.limits is not None:
            values[f"{name}_cell_limits"] = bucket.limits
    return values


def print_bucket_shortfalls(buckets, cells):
    for name, bucket in buckets.items():
        if bucket.limits is None:
            continue
        for cell in np.flatnonzero(bucket.counts < bucket.limits):
            print(f"[buckets] missing {name} cell {tuple(map(int, cells[cell]))}: "
                  f"{bucket.counts[cell]}/{bucket.limits[cell]}", flush=True)


def audit_snapshots(dataset, config):
    if not config["audit"]["enabled"]:
        return None
    count = min(config["audit"]["samples"], len(dataset["episode_id"]))
    rng = np.random.default_rng(config["dataset"]["split_seed"])
    indices = np.sort(rng.choice(len(dataset["episode_id"]), count, replace=False))
    rows = {}
    arrays = {"sample_indices": indices}
    for source in ("agent_gru", "agent_message"):
        groups = defaultdict(list)
        ids = np.empty(count, dtype=np.int32)
        for i, index in enumerate(indices):
            groups[digest(dataset[source][index])].append(i)
        conflicts = samples = pairs = 0
        matching = np.zeros(len(STAGES), dtype=np.int64)
        for group_id, members in enumerate(groups.values()):
            ids[members] = group_id
            labels = dataset["cell_id"][indices[members]]
            hist = dataset["history_sha256"][indices[members]]
            counts = Counter(map(int, labels))
            if len(counts) < 2:
                continue
            conflicts += 1
            samples += len(members)
            pairs += cross_cell_pairs(counts)
            for stage in range(len(STAGES)):
                histories = defaultdict(Counter)
                for label, fingerprint in zip(labels, hist[:, stage], strict=True):
                    histories[fingerprint.tobytes()][int(label)] += 1
                matching[stage] += sum(cross_cell_pairs(c) for c in histories.values())
        rows[source] = {"distinct_states": len(groups), "mixed_cell_groups": conflicts,
                        "mixed_cell_samples": samples, "cross_cell_pairs": pairs,
                        "history_matches": dict(zip(STAGES, map(int, matching), strict=True))}
        arrays[f"{source}_group"] = ids
    return ({"format": "swarmecho-own-cell-audit/v1", "samples": count,
             "source": {"kind": "own_snapshots", "scope": "seeded subset across retained episode pools"},
             "inputs": rows}, arrays)


def collect_own_dataset(model, level, config, run_dir, identity, graph):
    import jax
    import jax.numpy as jnp
    from swarmecho.training.train import _validate_evaluation_pairs

    d = config["dataset"]
    cell_size = float(graph["cell_size_m"])
    if not all(np.isfinite(x) and x > 0 for x in (cell_size, level.env.max_speed, level.env.dt)):
        raise ValueError("Own-cell sampling requires positive finite cell size, max speed and dt.")
    seconds = cell_size / level.env.max_speed + config["sampling"]["buffer_seconds"]
    stride = max(1, int(math.ceil(seconds / level.env.dt - 1e-10)))
    collect = make_collector(level, own_stride=stride)
    lookup = {tuple(cell): i for i, cell in enumerate(graph["cells"])}
    kept, episodes = [], []
    splits = {name: [] for name in ("train", "validation", "test")}
    uniform = config["sampling"].get("uniform_cells", True)
    eligible = eligible_cells(graph["cells"], config["sampling"].get("exclude_cells", []))
    limits = eligible_cell_limits({name: d[f"{name}_samples"] for name in splits}, eligible, d["split_seed"]) if uniform else {}
    buckets = {name: CellBuckets(len(lookup), limits.get(name)) for name in splits}
    filtering = PrefixFilter()
    coverage = np.zeros(len(lookup), dtype=np.int64)
    excluded_counts = np.zeros(len(lookup), dtype=np.int64)
    rng = np.random.default_rng(d["split_seed"])
    attempted = outside = total_candidates = quota_unused = 0
    before_found = after_found = 0
    started = time.monotonic()
    print(f"[cells] {int(eligible.sum())}/{len(eligible)} eligible cells; excluded "
          f"{graph['cells'][~eligible].tolist()} (absolute grid XYZ)", flush=True)
    for split, indices in splits.items():
        quota = d[f"{split}_samples"]
        bucket = buckets[split]
        while len(indices) < quota and attempted < d["max_episodes"]:
            count = min(d["eval_batch_size"], d["max_episodes"] - attempted)
            ids = np.arange(attempted, attempted + count, dtype=np.uint32)
            keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(d["seed"]), i))(jnp.asarray(ids))
            print(f"[collect-own] {split}: {len(indices)}/{quota}; episodes {attempted + 1}–{attempted + count}; every {stride} steps", flush=True)
            batch = jax.device_get(collect(model, keys))
            _validate_evaluation_pairs(level, batch["spawn_pair_attempts"])
            for row, episode_id in enumerate(ids):
                # A batch never crosses episode pools; surplus final-batch episodes
                # are provenance only and cannot leak into the next pool.
                samples, excluded = episode_samples(batch, row, episode_id, stride, cell_size, lookup)
                outside += excluded
                total_candidates += len(samples)
                for sample in samples:
                    coverage[int(sample["cell_id"])] += 1
                    if sample["target_found"]:
                        after_found += 1
                    else:
                        before_found += 1
                episodes.append({"episode_id": episode_id, "pool": np.int8(tuple(splits).index(split)),
                                 "success": batch["success"][row], "episode_length": batch["episode_length"][row],
                                 "target_position": batch["target_position"][row]})
                considered = 0
                for i in rng.permutation(len(samples)):
                    if len(indices) == quota:
                        break
                    sample = samples[i]
                    considered += 1
                    cell = int(sample["cell_id"])
                    if not eligible[cell]:
                        excluded_counts[cell] += 1
                        continue
                    if bucket.full(cell):
                        continue
                    if filtering.accept(sample, split):
                        bucket.add(cell)
                        indices.append(len(kept))
                        kept.append(sample)
                quota_unused += len(samples) - considered
            attempted += count
            progress = {"stage": "collection", "pool": split, "attempted": attempted,
                       "accepted": len(kept), "pool_accepted": len(indices), "pool_requested": quota,
                       "duplicate_samples": filtering.duplicates, "cross_split_duplicates": filtering.cross_split,
                       "excluded_cell_samples": int(excluded_counts.sum()),
                       "eligible_cells": int(eligible.sum()),
                       "full_bucket_rejections": int(sum(b.rejected.sum() for b in buckets.values()))}
            if uniform:
                deficits = bucket.limits - bucket.counts
                progress.update(cell_bucket_counts=bucket.counts, cell_bucket_limits=bucket.limits,
                                unfilled_cells=int((deficits > 0).sum()))
                underfilled = np.flatnonzero(deficits > 0)
                # The largest deficits identify cells delaying this split.
                slowest = underfilled[np.argsort(-deficits[underfilled], kind="stable")[:5]]
                note = ", ".join(f"{tuple(map(int, graph['cells'][cell]))}:{bucket.counts[cell]}/{bucket.limits[cell]}" for cell in slowest)
                print(f"[buckets] {split}: {len(underfilled)} unfilled cells" + (f"; {note}" if note else ""), flush=True)
            update_run(run_dir, progress=progress)
            del batch
        if len(indices) < quota:
            break
    coverage_data = bucket_source_data(coverage, buckets)
    coverage_data.update(eligible_cell_mask=eligible, excluded_cell_counts=excluded_counts)
    if not kept:
        arrays = {}
        add_group(arrays, "coverage", coverage_data)
        add_group(arrays, "geometry", graph)
        add_group(arrays, "episodes", {key: np.stack([row[key] for row in episodes]) for key in episodes[0]})
        save_data(run_dir, arrays)
        update_run(run_dir, collection={"num_samples": 0, "attempted_episodes": attempted,
                   "candidate_samples": total_candidates, "outside_interior_samples": outside,
                   "uniform_cell_buckets": uniform,
                   "eligible_cell_count": int(eligible.sum()), "total_cell_count": len(eligible),
                   "excluded_cells": graph["cells"][~eligible].tolist(),
                   "excluded_cell_samples": int(excluded_counts.sum()),
                   "full_bucket_rejections": int(sum(b.rejected.sum() for b in buckets.values()))})
        print_bucket_shortfalls(buckets, graph["cells"])
        raise RuntimeError("No eligible own-cell snapshots were retained. Partial visit/bucket counts saved; "
                           "inspect cell quotas, active agents and visited cells.")
    dataset = {key: np.stack([row[key] for row in kept]) for key in kept[0]}
    episode_data = {key: np.stack([row[key] for row in episodes]) for key in episodes[0]}
    splits = {name: np.asarray(indices, dtype=np.int32) for name, indices in splits.items()}
    labels = target_labels(dataset, graph, position_key="own_position", cell_key="own_cell")
    # Do not save target-labelled aliases in the own-cell experiment.
    labels.pop("target_storey")
    labels["own_cells"] = labels.pop("target_cells")
    audit = audit_snapshots(dataset, config)
    metadata = {"num_samples": len(kept), "attempted_episodes": attempted,
                "successful_episodes": int(episode_data["success"].sum()),
                "success_rate": float(episode_data["success"].mean()),
                "collection_seconds": time.monotonic() - started,
                "sampling_steps": stride, "sampling_seconds": stride * level.env.dt,
                "sampling_formula": "ceil((cell_size_m / max_speed + buffer_seconds) / dt)",
                "candidate_samples": total_candidates, "outside_interior_samples": outside,
                "duplicate_samples": filtering.duplicates, "cross_split_duplicates": filtering.cross_split,
                "quota_unused_samples": quota_unused,
                "uniform_cell_buckets": uniform,
                "eligible_cell_count": int(eligible.sum()), "total_cell_count": len(eligible),
                "excluded_cells": graph["cells"][~eligible].tolist(),
                "excluded_cell_samples": int(excluded_counts.sum()),
                "full_bucket_rejections": int(sum(b.rejected.sum() for b in buckets.values())),
                "cell_bucket_rule": "floor/ceil split totals over eligible interior cells; seeded rotating remainder" if uniform else "none; eligible-cell visitation weighted",
                "candidate_before_found": before_found, "candidate_after_found": after_found,
                "retained_before_found": int((~dataset["target_found"]).sum()),
                "retained_after_found": int(dataset["target_found"].sum()),
                "pool_episode_counts": {name: int((episode_data["pool"] == i).sum()) for i, name in enumerate(splits)},
                "input_dimensions": {source: int(dataset[source].shape[1]) for source in ("agent_gru", "agent_message")},
                "identity": identity,
                "split_rule": "separate episode pools: train, validation, test; earlier retained input histories excluded",
                "deduplication_rule": "one exact complete combined-input history/cell pair per pool; conflicting labels retained",
                "snapshot_rule": "updated GRU/current signature+value at regular pre-motion active-agent observations"}
    if any(len(splits[name]) != d[f"{name}_samples"] for name in splits):
        arrays = {}
        for prefix, values in (("dataset", dataset), ("episodes", episode_data), ("labels", labels),
                               ("geometry", graph), ("splits", splits), ("coverage", coverage_data)):
            add_group(arrays, prefix, values)
        if audit:
            add_group(arrays, "audit", audit[1])
        save_data(run_dir, arrays)
        update_run(run_dir, collection=metadata, ambiguity_audit=audit[0] if audit else None)
        print_bucket_shortfalls(buckets, graph["cells"])
        raise RuntimeError(f"Own-cell collection fell short after {attempted} episodes: "
                           f"{ {name: len(values) for name, values in splits.items()} }. Partial source data and bucket counts saved; "
                           "increase dataset.max_episodes or inspect visits to underfilled cells.")
    return dataset, metadata, episode_data, audit, labels, splits, coverage_data
