"""Inspector geometry from frozen snapshots or verified historical map files."""
import json

import numpy as np
import yaml

from geo_experiments.shared.common import fingerprint, resolve_path


def building_for_report(arrays, map_id, manifest):
    from swarmecho.visualize.inspector import building_geometry

    prefix = f"maps/{map_id}"
    if f"{prefix}/building_snapshot" in arrays:
        snapshot = json.loads(str(arrays[f"{prefix}/building_snapshot"]))
        return building_geometry({"building_snapshot": snapshot})

    # Older collections saved collision boxes only. Resolve their exact authored
    # map without generating maps or restoring/running the policy. Verify against
    # the collection identity before using the inspector's geometry constructor.
    names = list(map(str, arrays["geometry/map_names"]))
    identities = manifest.get("provenance", {}).get("identity", {}).get("maps", [])
    expected = next((row.get("sha256") for row in identities if row["name"] == names[map_id]), None)
    checkpoint = manifest["config"]["policy"].get("checkpoint")
    candidates = []
    if checkpoint and expected:
        run = resolve_path(checkpoint).parent.parent
        for path in sorted((run / "probe_maps").glob("*.json")):
            selected = json.loads(path.read_text(encoding="utf-8"))
            if [row["name"] for row in selected.get("maps", [])] == names and selected.get("pool"):
                candidates.append(resolve_path(selected["pool"]) / "maps" / f"{names[map_id]}.yaml")
        bank_path = run / "random_buildings.json"
        if bank_path.is_file():
            bank = json.loads(bank_path.read_text(encoding="utf-8"))
            if bank.get("pool"):
                candidates.append(resolve_path(bank["pool"]) / "maps" / f"{names[map_id]}.yaml")
        candidates.append(run / "validation" / "generated_maps" / f"{names[map_id]}.yaml")
        from swarmecho.core.config import MAP_DIR
        from swarmecho.core.compatibility import resolve_map_file
        candidates.append(resolve_map_file(names[map_id], MAP_DIR))
    for path in candidates:
        if path.is_file() and fingerprint(path) == expected:
            return building_geometry({"building_snapshot": yaml.safe_load(path.read_text(encoding="utf-8"))})

    # Portable historical artifacts still render their actual saved solids if
    # the original authored map is unavailable. New collections store snapshots.
    cell = float(arrays["geometry/cell_size_m"][map_id])
    cells = arrays[f"{prefix}/cells"]
    layers = int(np.max(cells[:, 2])) + 1
    walls, floors = [], []
    for lo, hi in zip(arrays[f"{prefix}/solid_min"], arrays[f"{prefix}/solid_max"]):
        storey = max(0, min(layers - 1, int(np.floor((lo[2] + hi[2]) / (2 * cell)))))
        piece = dict(lo=lo.tolist(), hi=hi.tolist(), storey=storey)
        if hi[2] - lo[2] < cell / 2 and np.argmin(hi - lo) == 2:
            floors.append(piece)
        else:
            walls.append(dict(**piece, outer=False))
    return dict(layers=layers, cell_size_m=cell, walls=walls, floors=floors, roofs=[], source="saved collision geometry")
