"""Freeze five seeded bank maps with inspectable base-neighbourhood noSpawn cells.

CPU geometry only; no JAX, training, pool regeneration or original-map edits.
Run from the repository root with a Python containing numpy and PyYAML.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swarmecho.env.buildings import compile_building, distance_comment, target_spawn_boxes


def visible(start, end, lower, upper):
    direction = end - start
    parallel = np.abs(direction) <= 1e-5
    safe = np.where(parallel, 1., direction)
    first, last = (lower - start) / safe, (upper - start) / safe
    near = np.max(np.where(parallel, -np.inf, np.minimum(first, last)), axis=-1)
    far = np.min(np.where(parallel, np.inf, np.maximum(first, last)), axis=-1)
    outside = np.any(parallel & ((start < lower) | (start > upper)), axis=-1)
    return not np.any((far >= np.maximum(near, 1e-5)) & (near <= 1. - 1e-5) & ~outside)


def exclusions(data):
    building = compile_building({**data, "target_exclusion_cells": []})
    lower, upper, volumes = target_spawn_boxes(
        building, SimpleNamespace(target_wall_buffer_fraction=.1), include_excluded=True)
    cells = np.floor((lower + upper) * .5 / building.cell_size_m).astype(int)
    # Largest free-box centre represents stair/door cells without putting a
    # centre inside a tread. LOS uses actual compiled solids and base position.
    points = {}
    for index in np.argsort(-volumes):
        points.setdefault(tuple(map(int, cells[index])), (lower[index] + upper[index]) * .5)
    base = building.base_position_m
    base_cell = tuple(map(int, np.floor(base / building.cell_size_m)))
    direct = [cell for cell, point in points.items()
              if visible(base, point, building.solid_min_m, building.solid_max_m)]
    direct.sort(key=lambda cell: (cell != base_cell, np.linalg.norm(points[cell] - base), cell))
    if len(direct) < 3 or base_cell not in direct:
        raise ValueError("Fewer than three directly visible free-cell representatives.")
    selected = direct[:3]
    corners = []
    for cell, point in points.items():
        if cell in direct:
            continue
        for entry in selected:
            if (sum(abs(a-b) for a, b in zip(cell, entry)) == 1
                    and visible(points[entry], point, building.solid_min_m, building.solid_max_m)):
                corners.append((np.linalg.norm(point-base), cell, entry))
    if not corners:
        raise ValueError("No adjacent, connected cell around a LOS-breaking turn.")
    _, around_corner, via = min(corners)
    chosen = [*selected, around_corner]
    return chosen, {"direct_los_cells": [list(c) for c in selected],
                    "around_corner_cell": list(around_corner), "around_corner_via": list(via),
                    "representative_points_m": {str(list(c)): points[c].tolist() for c in chosen}}


def starter_witness(building):
    """Conservative CPU example, not a production relay-calculator certificate."""
    cfg = SimpleNamespace(target_wall_buffer_fraction=.1)
    lo, hi, _ = target_spawn_boxes(building, cfg, include_excluded=True)
    grid = np.array(np.meshgrid(*[[.15, .5, .85]] * 3, indexing="ij")).reshape(3, -1).T
    candidates = (lo[:, None] + grid[None] * (hi-lo)[:, None]).reshape(-1, 3)
    candidates = [point for point in candidates
                  if np.linalg.norm(point-building.base_position_m) < 14.999
                  and visible(building.base_position_m, point, building.solid_min_m, building.solid_max_m)]
    lo, hi, _ = target_spawn_boxes(building, cfg)
    for target in (lo+hi) * .5:
        for relay in candidates:
            if (np.linalg.norm(relay-target) < 14.999
                    and visible(relay, target, building.solid_min_m, building.solid_max_m)):
                # The finder occupies the target position, which has the usual
                # spawn-wall clearance and therefore an unobstructed visual link.
                return {"relay_m": relay.tolist(), "finder_and_target_m": target.tolist()}
    raise ValueError("No conservative sampled two-agent starter placement after noSpawn exclusions.")


def main():
    seed = 456789
    map_root = ROOT / "src/swarmecho/curriculum_config/maps"
    pools = sorted((map_root / "random_building_bank/pools").glob("*/metadata.json"))
    available = []
    for metadata_path in pools:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        settings = metadata.get("settings", {}).get("random_buildings", {})
        if (metadata.get("complete") and settings.get("length_m") == 15.
                and settings.get("width_m") == 15. and settings.get("stories") == 3
                and settings.get("cell_size_m") == 5.):
            available.extend(metadata_path.parent / "maps" / (entry["id"] + ".yaml")
                             for entry in metadata["maps"])
    if len(available) < 5:
        raise ValueError("No complete 15x15x15 bank contains five maps.")
    random.Random(seed).shuffle(available)
    records, skipped = [], []
    for source in available:
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
        try:
            cells, details = exclusions(data)
            data["target_exclusion_cells"] = [list(cell) for cell in cells]
            details["cpu_two_agent_starter_witness"] = starter_witness(compile_building(data))
        except ValueError as error:
            skipped.append({"source": source.relative_to(ROOT).as_posix(), "reason": str(error)})
            continue
        name = f"G02_static_map_{len(records)+1:02d}"
        data.update(name=name, target_exclusion_cells=[list(cell) for cell in cells])
        building = compile_building(data)
        destination = map_root / (name + ".yaml")
        header = (f"# G02 selection seed: {seed}; source: {source.parent.parent.name}/{source.name}\n"
                  "# noSpawn: three base-LOS cells (including base), plus one adjacent LOS-breaking cell.\n")
        content = distance_comment(building) + "\n" + header + yaml.safe_dump(data, sort_keys=False)
        if destination.exists() and destination.read_text(encoding="utf-8") != content:
            if ("--replace-generated" not in sys.argv
                    or "# G02 selection seed:" not in destination.read_text(encoding="utf-8")):
                raise ValueError(f"Refusing to overwrite an edited static map: {destination}; use --replace-generated explicitly.")
        destination.write_text(content, encoding="utf-8")
        records.append({"name": name, "source": source.relative_to(ROOT).as_posix(),
                        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "base_position_m": building.base_position_m.tolist(),
                        "target_exclusion_cells": [list(c) for c in cells], **details})
        if len(records) == 5:
            break
    if len(records) != 5:
        raise ValueError("Could not select five maps satisfying the documented noSpawn rule.")
    manifest = {"seed": seed, "maps": records, "skipped_candidates": skipped,
                "rule": "Three nearest representative-cell base LOS cells including base, plus one face-adjacent non-LOS cell visible from one of those three. Cell representatives are largest legal spawn-box centres; this is not an all-points LOS guarantee. Retain only candidates with a sampled two-agent physical starter placement; production JAX certification is still required."}
    (map_root / "G02_static_selection.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"maps": records, "skipped": len(skipped)}, indent=2))


if __name__ == "__main__":
    main()
