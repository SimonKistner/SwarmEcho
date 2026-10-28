"""Map-aware position labels shared by target probes and future own-cell probes."""
from __future__ import annotations

import numpy as np

from geo_experiments.geometry import build_cell_graph

DECODERS = {
    "map_specific": "Map-specific cell probes",
    "map_cell": "Joint map + cell probe",
    "shared_cell": "Shared-coordinate cell probe",
}


def suite_geometry(levels, clearance=None):
    graphs = [build_cell_graph(level.building, clearance if clearance is not None else
                              level.env.drone_radius + level.env.obstacle_planning_clearance_m)
              for level in levels]
    sizes = np.asarray([float(g["cell_size_m"]) for g in graphs])
    if not np.all(sizes == sizes[0]):
        raise ValueError("Shared cell coordinates require a common cell size.")
    offsets = np.cumsum([0, *(len(g["cells"]) for g in graphs)]).astype(np.int32)
    count = int(offsets[-1])
    # Preserve old small-suite archives; large banks store only within-map blocks.
    if count <= 1024:
        distances = np.full((count, count), np.inf, dtype=np.float32)
        for i, graph in enumerate(graphs):
            distances[offsets[i]:offsets[i + 1], offsets[i]:offsets[i + 1]] = graph["distances"]
    else:
        width = max(len(g["cells"]) for g in graphs)
        distances = np.full((len(graphs), width, width), np.inf, dtype=np.float32)
        for i, graph in enumerate(graphs):
            n = len(graph["cells"])
            distances[i, :n, :n] = graph["distances"]
    cells = np.concatenate([g["cells"] for g in graphs])
    coordinates = np.asarray(sorted(set(map(tuple, cells)), key=lambda p: (p[2], p[1], p[0])), dtype=np.int32)
    lookup = {tuple(cell): i for i, cell in enumerate(coordinates)}
    return {
        "cells": cells, "cell_size_m": sizes[0], "distances": distances,
        "map_offsets": offsets, "map_names": np.asarray([l.building_name for l in levels]),
        "cell_map_id": np.repeat(np.arange(len(levels), dtype=np.int32), np.diff(offsets)),
        "coordinate_values": coordinates,
        "cell_coordinate_id": np.asarray([lookup[tuple(c)] for c in cells], dtype=np.int32),
    }


def position_labels(dataset, graph, *, position_key="target_position", task="target_cell"):
    """No map identity enters probe inputs. It is ground truth and expert routing only."""
    actual = np.floor(dataset[position_key] / float(graph["cell_size_m"])).astype(np.int32)
    map_ids = np.asarray(dataset["map_id"], dtype=np.int32)
    lookup = {(int(m), *map(int, cell)): i
              for i, (m, cell) in enumerate(zip(graph["cell_map_id"], graph["cells"], strict=True))}
    try:
        ids = np.asarray([lookup[(int(m), *map(int, c))] for m, c in zip(map_ids, actual, strict=True)], dtype=np.int32)
    except KeyError as exc:
        raise ValueError(f"Position outside the corresponding map's interior: {exc}") from exc
    floors = np.unique(graph["cells"][:, 2])
    return {task: ids, "coordinate_cell": graph["cell_coordinate_id"][ids],
            "actual_storey": actual[:, 2] + 1, "target_storey": np.searchsorted(floors, actual[:, 2]),
            "storey_offsets": floors, "map_id": map_ids,
            "required_agents": np.asarray(dataset["target_required_agents"], dtype=np.int32),
            "cell_offsets": graph["cells"], "cell_coordinates": actual}


def class_space(labels, graph, task, decoder, map_id=None):
    """Return training labels/class coordinates and an optional local-to-global lookup."""
    if task == "target_storey":
        return labels[task], labels["storey_offsets"], None
    if decoder == "shared_cell":
        return labels["coordinate_cell"], graph["coordinate_values"], None
    if decoder == "map_specific":
        start, stop = graph["map_offsets"][map_id:map_id + 2]
        return labels[task] - start, graph["cells"][start:stop], np.arange(start, stop, dtype=np.int32)
    # Map ID is explicitly encoded into the class definition, not the input.
    values = np.column_stack((graph["cell_map_id"], graph["cells"]))
    return labels[task], values, None


def display_predictions(predicted, labels, graph, test, decoder):
    """Convert shared coordinate IDs to cells on the true map for distance scoring."""
    if decoder != "shared_cell":
        return np.asarray(predicted)
    lookup = {(int(m), int(c)): i for i, (m, c) in
              enumerate(zip(graph["cell_map_id"], graph["cell_coordinate_id"], strict=True))}
    return np.asarray([lookup.get((int(m), int(c)), -1)
                       for m, c in zip(labels["map_id"][test], predicted, strict=True)], dtype=np.int32)


def cell_errors(predicted, true, graph):
    """-1 means a wrong-map prediction; infinity means no route/invalid cell."""
    predicted, true = np.asarray(predicted), np.asarray(true)
    valid = predicted >= 0
    result = np.full(len(true), np.inf, dtype=np.float32)
    wrong_map = valid & (graph["cell_map_id"][np.maximum(predicted, 0)] != graph["cell_map_id"][true])
    same_map = valid & ~wrong_map
    if graph["distances"].ndim == 3:
        maps = graph["cell_map_id"][true[same_map]]
        offsets = graph["map_offsets"][maps]
        result[same_map] = graph["distances"][maps, true[same_map] - offsets, predicted[same_map] - offsets]
    else:
        result[same_map] = graph["distances"][true[same_map], predicted[same_map]]
    result[wrong_map] = -1
    return result
