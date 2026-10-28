"""Saved-vector collision groups and episode similarities; no policy replay."""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from geo_experiments.audit_common import digest, cross_cell_pairs

DECODERS = ("map_specific", "map_cell", "shared_cell")


def vector_groups(features, indices=None):
    rows = range(len(features)) if indices is None else indices
    groups = defaultdict(list)
    for i in rows:
        groups[digest(features[i])].append(int(i))
    return [np.asarray(rows, dtype=np.int32) for rows in groups.values()]


def locations(dataset, cell_size):
    xyz = np.floor(dataset["target_position"] / cell_size).astype(np.int32)
    maps = dataset.get("map_id", np.zeros(len(xyz), dtype=np.int32))
    return np.column_stack((maps, xyz))


def conditional_groups(groups, location, decoder):
    for rows in groups:
        if decoder == "map_specific":
            for m in np.unique(location[rows, 0]):
                yield rows[location[rows, 0] == m]
        else:
            yield rows


def ambiguous_rows(groups, location, decoders):
    selected = set()
    for decoder in decoders:
        for rows in conditional_groups(groups, location, decoder):
            labels = location[rows, 1:] if decoder == "shared_cell" else location[rows]
            if len(np.unique(labels, axis=0)) > 1:
                selected.update(map(int, rows))
    return np.asarray(sorted(selected), dtype=np.int32)


def ambiguity_counts(groups, location, decoder):
    mixed = samples = pairs = unavoidable = 0
    for rows in conditional_groups(groups, location, decoder):
        values = location[rows, 1:] if decoder == "shared_cell" else location[rows]
        counts = Counter(map(tuple, values))
        if len(counts) > 1:
            mixed += 1
            samples += len(rows)
            pairs += cross_cell_pairs(counts)
            unavoidable += len(rows) - max(counts.values())
    return {"conflicting_groups": mixed, "conflicting_samples": samples,
            "conflicting_pairs": pairs, "empirical_minimum_errors": unavoidable}


def _keys(values):
    values = np.asarray(values)
    return list(map(tuple, values)) if values.ndim > 1 else values.tolist()


def categorical_similarity(values, groups, *, valid=None):
    values = np.asarray(values)
    valid = np.ones(len(values), dtype=bool) if valid is None else valid
    baseline_counts = Counter(_keys(values[valid]))
    n = int(valid.sum())
    baseline_pairs = n * (n - 1) // 2
    same = pairs = homogeneous = complete = 0
    for rows in groups:
        known = rows[valid[rows]]
        if len(known) != len(rows):
            continue
        counts = Counter(_keys(values[known]))
        complete += 1
        homogeneous += int(len(counts) == 1)
        pairs += len(known) * (len(known) - 1) // 2
        same += sum(c * (c - 1) // 2 for c in counts.values())
    return {"same_groups": homogeneous, "known_groups": complete,
            "same_pairs": same, "pairs": pairs,
            "pair_agreement": same / pairs if pairs else None,
            "baseline_pair_agreement": sum(c * (c - 1) // 2 for c in baseline_counts.values()) / baseline_pairs if baseline_pairs else None}


def _counts(values):
    return [{"value": list(k) if isinstance(k, tuple) else k, "samples": int(n)}
            for k, n in sorted(Counter(_keys(values)).items())]


def _range(values):
    return {"min": float(np.min(values)), "max": float(np.max(values)), "mean": float(np.mean(values))}


def analyse_collisions(dataset, source, cell_size):
    """Analyse every retained episode, with task-specific ambiguity definitions."""
    all_groups = vector_groups(dataset[source])
    groups = sorted((g for g in all_groups if len(g) > 1), key=lambda g: (-len(g), int(g[0])))
    location = locations(dataset, cell_size)
    fields = {"Map": location[:, 0], "Map + target cell": location,
              "Target XYZ cell": location[:, 1:], "Target storey": location[:, 3] + 1}
    for name, key in (("Target-required agents", "target_required_agents"),
                      ("Finder index", "finder_id"), ("Reporter index", "reporter_id"),
                      ("Active agents at finder capture", "finder_active_agents"),
                      ("Active agents at base receipt", "base_active_agents"),
                      ("Chain success", "success")):
        if key in dataset:
            fields[name] = dataset[key]
    if "finder_id" in dataset and "reporter_id" in dataset:
        fields["Reporter is the finder"] = dataset["finder_id"] == dataset["reporter_id"]
    missing = [name for name in ("Active agents at finder capture", "Active agents at base receipt") if name not in fields]
    similarities = {}
    for name, values in fields.items():
        known = values > 0 if name == "Target-required agents" else values >= 0 if name.startswith("Active agents") else None
        similarities[name] = categorical_similarity(values, groups, valid=known)
    details = []
    member_rows, offsets = [], [0]
    for i, rows in enumerate(groups):
        member_rows.extend(map(int, rows))
        offsets.append(len(member_rows))
        item = {"id": i + 1, "samples": len(rows), "properties": {k: _counts(v[rows]) for k, v in fields.items()}}
        positions = dataset["target_position"][rows]
        item["target_position"] = {"min_xyz_m": positions.min(axis=0).tolist(), "max_xyz_m": positions.max(axis=0).tolist(),
            "rms_spread_m": float(np.sqrt(np.mean(np.sum((positions - positions.mean(axis=0)) ** 2, axis=1))))}
        for name, key in (("Discovery step", "discovery_step"), ("Finder capture step", "finder_step"), ("Base receipt step", "base_step")):
            if key in dataset:
                item.setdefault("timing", {})[name] = _range(dataset[key][rows])
        if "base_step" in dataset and "finder_step" in dataset:
            item.setdefault("timing", {})["Finder capture to base receipt (steps)"] = _range(dataset["base_step"][rows] - dataset["finder_step"][rows])
        for key in ("finder_gru", "finder_message"):
            item.setdefault("upstream_distinct", {})[key] = len(vector_groups(dataset[key], rows))
        details.append(item)
    n = len(location)
    summary = {"samples": n, "distinct_vectors": len(all_groups), "repeated_groups": len(groups),
               "repeated_samples": len(member_rows), "repeated_fraction": len(member_rows) / n if n else 0,
               "by_decoder": {d: ambiguity_counts(groups, location, d) for d in DECODERS},
               "similarities": similarities, "missing_properties": missing, "groups": details,
               "upstream": {key: sum(item["upstream_distinct"][key] == item["samples"] for item in details)
                            for key in ("finder_gru", "finder_message")}}
    return summary, {"sample_indices": np.asarray(member_rows, dtype=np.int32), "group_offsets": np.asarray(offsets, dtype=np.int32)}
