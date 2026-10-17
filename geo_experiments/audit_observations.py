"""Shared streaming variability audit and standalone console runner."""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from datetime import datetime
import hashlib
import os
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np

STAGES = ("Raw observation", "Encoded observation", "GRU input (encoding + received context)")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def digest(*arrays):
    """Exact float32 fingerprints, treating signed zero as equal like np.unique."""
    result = hashlib.sha256()
    for array in arrays:
        array = np.array(array, copy=True, order="C")
        if array.dtype.kind == "f":
            if not np.isfinite(array).all():
                raise ValueError("Nonfinite features found; history comparisons would be invalid.")
            array[array == 0] = 0
        result.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        result.update(array.tobytes())
    return result.digest()


def cross_cell_pairs(counts):
    total = sum(counts.values())
    return (total * (total - 1) - sum(n * (n - 1) for n in counts.values())) // 2


def differences(a, b):
    delta = np.abs(a - b)
    return (float(delta.max()), float(np.sqrt(np.mean(delta.astype(np.float64) ** 2))),
            float((delta == 0).mean()), float(np.all(delta == 0, axis=1).mean()))


class Audit:
    def __init__(self):
        self.groups = {}
        self.comparisons = []
        self.mismatched_times = Counter()
        self.samples = 0
        self.source_rows = []
        self.reference_rows = []
        self.unavailable_references = Counter()
        self.reference_cache = OrderedDict()
        self.reference_bytes = 0

    def add(self, batch, row, episode_id, cell_size):
        valid = batch["history_steps"][row] >= 0
        steps = batch["history_steps"][row, valid]
        flags = batch["history_flags"][row, valid]
        raw = batch["history_obs"][row, valid]
        encoded = batch["history_encoded"][row, valid]
        context = batch["history_context"][row, valid]
        gru_input = np.concatenate((encoded, context), axis=1)
        position = np.array(batch["target_position"][row], copy=True)
        base_cell = np.floor(batch["base_position"][row] / cell_size).astype(int)
        cell = tuple((np.floor(position / cell_size).astype(int) - base_cell).tolist())
        key = digest(batch["finder_gru"][row])
        fingerprints = [digest(steps, flags, values) for values in (raw, encoded, gru_input)]
        if key not in self.groups:
            self.groups[key] = {
                "cells": Counter(), "histories": [{}, {}, {}],
                "id": len(self.groups), "reference_episode": episode_id,
                "reference_steps": digest(steps), "reference_hashes": fingerprints,
                "reference_cell": cell, "reference_position": position,
            }
            reference = (steps, flags, raw, encoded, gru_input)
            size = sum(a.nbytes for a in reference)
            self.reference_cache[key] = reference
            self.reference_bytes += size
            while self.reference_bytes > 128 * 2**20 and self.reference_cache:
                _, removed = self.reference_cache.popitem(last=False)
                self.reference_bytes -= sum(a.nbytes for a in removed)
        group = self.groups[key]
        group["cells"][cell] += 1
        for stage, hist_key in enumerate(fingerprints):
            cluster = group["histories"][stage].setdefault(hist_key, {"cells": Counter(), "targets": []})
            cluster["cells"][cell] += 1
            cluster["targets"].append((episode_id, position, cell))
        is_cross_cell = cell != group["reference_cell"]
        self.source_rows.append((episode_id, group["id"], fingerprints))
        if digest(steps) != group["reference_steps"]:
            self.mismatched_times["all"] += 1
            self.mismatched_times["cross_cell"] += int(is_cross_cell)
        elif sum(group["cells"].values()) > 1:
            reference = self.reference_cache.get(key)
            if reference is None and fingerprints != group["reference_hashes"]:
                self.unavailable_references["all"] += 1
                self.unavailable_references["cross_cell"] += int(is_cross_cell)
                self.samples += 1
                return
            if reference is None:
                stats, same_flags = [(0., 0., 1., 1.)] * 3, True
            else:
                self.reference_cache.move_to_end(key)
                _, ref_flags, *ref_values = reference
                stats = [differences(values, ref) for values, ref in zip((raw, encoded, gru_input), ref_values)]
                same_flags = np.array_equal(flags, ref_flags)
            self.comparisons.append((is_cross_cell, same_flags, stats,
                                     float(np.linalg.norm(position - group["reference_position"]))))
            self.reference_rows.append((episode_id, group["reference_episode"]))
        self.samples += 1

    def arrays(self):
        """Compact source data; full floating histories are not copied into runs."""
        rows = self.source_rows
        comparisons = self.comparisons
        return {
            "episode_id": np.asarray([r[0] for r in rows], dtype=np.uint32),
            "hidden_group": np.asarray([r[1] for r in rows], dtype=np.int32),
            "history_sha256": np.asarray([[list(h) for h in r[2]] for r in rows], dtype=np.uint8).reshape(-1, 3, 32),
            "reference_episode_pairs": np.asarray(self.reference_rows, dtype=np.uint32).reshape(-1, 2),
            "reference_cross_cell": np.asarray([r[0] for r in comparisons], dtype=bool),
            "reference_flags_equal": np.asarray([r[1] for r in comparisons], dtype=bool),
            "reference_statistics": np.asarray([r[2] for r in comparisons], dtype=np.float64).reshape(-1, 3, 4),
            "reference_target_separation_m": np.asarray([r[3] for r in comparisons], dtype=np.float64),
        }

    def summary(self):
        mixed = [g for g in self.groups.values() if len(g["cells"]) > 1]
        pairs = sum(cross_cell_pairs(g["cells"]) for g in self.groups.values())
        result = {"format": "swarmecho-ambiguity-audit/v1", "status": "complete",
                  "samples": self.samples, "distinct_grus": len(self.groups),
                  "mixed_cell_groups": len(mixed),
                  "mixed_cell_samples": sum(sum(g["cells"].values()) for g in mixed),
                  "cross_cell_pairs": pairs, "exact": {}, "references": {}}
        for stage, title in enumerate(STAGES):
            matched = sum(cross_cell_pairs(c["cells"]) for g in self.groups.values()
                          for c in g["histories"][stage].values())
            result["exact"][title] = {"matched_pairs": matched, "total_pairs": pairs,
                                      "rate": matched / pairs if pairs else None}
        for kind in ("all", "cross_cell"):
            rows = [r for r in self.comparisons if kind == "all" or r[0]]
            reference = {"count": len(rows), "excluded_times": self.mismatched_times[kind],
                         "excluded_memory": self.unavailable_references[kind], "stages": {}}
            result["references"][kind] = reference
            if not rows:
                continue
            flags = np.asarray([r[1] for r in rows])
            statistics = np.asarray([r[2] for r in rows])
            separation = np.asarray([r[3] for r in rows])
            reference["flags_equal_rate"] = float(flags.mean())
            for stage, title in enumerate(STAGES):
                maximum, rms, elements, steps = statistics[:, stage].T
                record = {"equal_components_rate": float(elements.mean()), "equal_steps_rate": float(steps.mean()),
                          "max_abs_median": float(np.median(maximum)), "max_abs_p90": float(np.quantile(maximum, .9)),
                          "max_abs_max": float(maximum.max()), "rms_median": float(np.median(rms)), "thresholds": []}
                for tolerance in (0., 1e-6, 1e-3):
                    close = (maximum <= tolerance) & flags
                    record["thresholds"].append({"atol": tolerance, "count": int(close.sum()),
                        "rate": float(close.mean()),
                        "target_separation_median_m": float(np.median(separation[close])) if close.any() else None,
                        "target_separation_max_m": float(separation[close].max()) if close.any() else None})
                reference["stages"][title] = record
        examples, max_any, max_cross = [], None, None
        for group in self.groups.values():
            for cluster in group["histories"][2].values():
                targets = cluster["targets"]
                if len(targets) < 2:
                    continue
                positions = np.asarray([p for _, p, _ in targets])
                cells = np.asarray([c for _, _, c in targets])
                best = (-1., 0, 0)
                for start in range(0, len(targets), 128):
                    delta = positions[start:start + 128, None] - positions[None]
                    distance = np.sum(delta.astype(np.float64) ** 2, axis=-1)
                    max_any = max(max_any or 0., float(np.sqrt(distance.max())))
                    distance[~np.any(cells[start:start + 128, None] != cells[None], axis=-1)] = -1
                    i, j = np.unravel_index(np.argmax(distance), distance.shape)
                    if distance[i, j] > best[0]:
                        best = (float(distance[i, j]), start + i, j)
                if best[0] < 0:
                    continue
                separation = float(np.sqrt(best[0]))
                max_cross = max(max_cross or 0., separation)
                a, b = targets[best[1]], targets[best[2]]
                examples.append({"samples": len(targets), "separation_m": separation,
                                 "episode_ids": [a[0], b[0]], "cells": [list(a[2]), list(b[2])],
                                 "positions_m": [a[1].tolist(), b[1].tolist()]})
        result["target_separation"] = {"max_any_m": max_any, "max_cross_cell_m": max_cross,
            "mixed_cell_clusters": len(examples), "examples": sorted(examples, key=lambda r: r["separation_m"], reverse=True)[:5]}
        return result

    def report(self):
        result = self.summary()
        print("\n=== OBSERVATION VARIABILITY AUDIT ===")
        print(f"Samples: {result['samples']}; exact distinct finder GRUs: {result['distinct_grus']}")
        print(f"Mixed-target-cell GRU groups: {result['mixed_cell_groups']}; episodes in them: {result['mixed_cell_samples']}")
        print(f"Different-cell episode pairs within identical GRUs: {result['cross_cell_pairs']}")
        for title, row in result["exact"].items():
            share = f"{row['rate']:.2%}" if row['rate'] is not None else "N/A"
            print(f"  {title}: {row['matched_pairs']}/{row['total_pairs']} EXACT history matches ({share})")
        print("\nSimilarity to FIRST episode in each GRU group (reference comparisons, not all pairs):")
        for kind, reference in result["references"].items():
            print(f"  {kind}: {reference['count']} comparisons; excluded times={reference['excluded_times']}, unavailable reference={reference['excluded_memory']}")
            for title, row in reference["stages"].items():
                print(f"    {title}: equal components {row['equal_components_rate']:.2%}; fully equal steps {row['equal_steps_rate']:.2%}")
                print(f"      max-abs median/p90/max {row['max_abs_median']:.3g}/{row['max_abs_p90']:.3g}/{row['max_abs_max']:.3g}; median RMS {row['rms_median']:.3g}")
                for threshold in row["thresholds"]:
                    print(f"      Entire history <= {threshold['atol']:g}: {threshold['count']} ({threshold['rate']:.2%}); target separation median/max metres: {threshold['target_separation_median_m']}/{threshold['target_separation_max_m']}")
        separation = result["target_separation"]
        print(f"\nIdentical GRU-input histories: maximum target separation {separation['max_any_m']} m; cross-cell {separation['max_cross_cell_m']} m")
        for example in separation["examples"]:
            print(f"  n={example['samples']}; {example['separation_m']:.3f} m; episodes {example['episode_ids']}; cells {example['cells']}")
        print("Absolute tolerances; no skipped steps at --every 1; metres measure target separation, not route distance.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--every", type=int, default=1, help="Record every X steps, always including capture.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--checkpoint", help="Defaults to policy.checkpoint in geo_experiments/config.yaml.")
    parser.add_argument("--level", help="Defaults to policy.level in geo_experiments/config.yaml.")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if min(args.samples, args.every, args.batch_size) < 1:
        parser.error("samples, every and batch-size must be positive")
    limit = args.max_episodes if args.max_episodes is not None else max(20000, args.samples * 4)
    if limit < args.samples:
        parser.error("max-episodes must be at least samples")

    from omegaconf import OmegaConf
    from geo_experiments.collection import make_collector, prepare_policy

    config = OmegaConf.to_container(OmegaConf.load(Path(__file__).with_name("config.yaml")), resolve=True)
    for key in ("checkpoint", "level"):
        if getattr(args, key) is not None:
            config["policy"][key] = getattr(args, key)
    if args.seed is not None:
        config["dataset"]["seed"] = args.seed
    seed = config["dataset"]["seed"]
    if type(seed) is not int or not 0 <= seed < 2**32:
        parser.error("seed must be an unsigned 32-bit integer")
    print(f"[audit] {datetime.now(ZoneInfo('Europe/Berlin')).isoformat(timespec='seconds')}", flush=True)
    print(f"[audit] samples={args.samples}; every={args.every}; batch={args.batch_size}; seed={seed}; successful matched episodes only", flush=True)
    model, level, identity = prepare_policy(config)
    print(f"[audit] checkpoint={config['policy']['checkpoint']}; level={config['policy']['level']}", flush=True)
    for key in ("checkpoint_sha256", "map_sha256", "level_sha256", "runtime_sha256", "collector_sha256"):
        print(f"[identity] {key}={identity[key]}", flush=True)
    print(f"[identity] audit_sha256={hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}", flush=True)
    slots = (level.env.max_steps + args.every - 1) // args.every + 1
    memory = slots * level.env.num_agents * (model.obs_dim + model.hidden_dim + model.tarmac_val_dim) * 4 * args.batch_size
    print(f"[audit] Team history buffers about {memory / 2**20:.0f} MiB per batch (plus model/runtime); first batch compiles.", flush=True)
    if args.every > 1:
        print("[audit] Subsampled histories: identical samples do NOT prove equality at skipped steps.", flush=True)

    import jax
    import jax.numpy as jnp
    from swarmecho.training.train import _validate_evaluation_pairs

    collect = make_collector(level, history_stride=args.every)
    audit = Audit()
    attempted = succeeded = missing = batches = 0
    started = time.monotonic()
    while audit.samples < args.samples and attempted < limit:
        count = min(args.batch_size, limit - attempted)
        ids = np.arange(attempted, attempted + count, dtype=np.uint32)
        keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(seed), i))(jnp.asarray(ids))
        batch = jax.device_get(collect(model, keys))
        _validate_evaluation_pairs(level, batch["spawn_pair_attempts"])
        selected = np.flatnonzero(batch["success"] & batch["complete_sample"])[:args.samples - audit.samples]
        for row in selected:
            audit.add(batch, int(row), int(ids[row]), level.building.cell_size_m)
        succeeded += int(batch["success"].sum())
        missing += int((batch["success"] & ~batch["complete_sample"]).sum())
        attempted += count
        batches += 1
        del batch
        if batches == 1 or batches % 16 == 0 or audit.samples == args.samples:
            print(f"[collect] {audit.samples}/{args.samples}; attempted={attempted}; unique GRUs={len(audit.groups)}; elapsed={time.monotonic() - started:.1f}s", flush=True)
    print(f"[collection] successful={succeeded}/{attempted} ({succeeded / attempted:.2%}); successful missing snapshot={missing}")
    print(f"[collection] elapsed={time.monotonic() - started:.1f}s; every={args.every}; policy unchanged; no files written")
    if audit.samples != args.samples:
        raise RuntimeError(f"Only {audit.samples}/{args.samples} collected. Increase --max-episodes or inspect policy success.")
    audit.report()


if __name__ == "__main__":
    main()
