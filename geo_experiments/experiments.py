"""Encapsulated readouts over one immutable collection; no simulator dependencies."""
from __future__ import annotations

import numpy as np

from geo_experiments.artifacts import add_group, group, save_data, update_run
from geo_experiments.common import experiment_axes
from geo_experiments.labels import class_space
from geo_experiments.probes import train_probe
from geo_experiments.console import terminal_print as print


def fit_experiment(config, shared, run_dir, *, shared_reference=True):
    dataset, labels, splits, graph = (group(shared, k) for k in ("dataset", "labels", "splits", "geometry"))
    tasks, sources = experiment_axes(config)
    decoder = config["protocol"]["decoder"]
    if config.get("experiment", "target") != "target":
        raise ValueError("The suite currently executes target probes only; own-cell collection is a backlog item.")
    output, jobs = ({} if shared_reference else dict(shared)), []
    # Experts receive the same episode partition as joint readouts, restricted to
    # their map. Map IDs never get concatenated to the feature vectors.
    experts = range(len(graph["map_names"])) if decoder == "map_specific" else [None]
    total = len(tasks) * len(sources) * len(config["probe"]["seeds"]) * len(experts)
    for task in tasks:
        for source in sources:
            for seed in config["probe"]["seeds"]:
                prefix = f"probes/{task}/{source}/seed_{seed}"
                class_count = len(labels["storey_offsets"]) if task == "target_storey" else (
                    len(graph["coordinate_values"]) if decoder == "shared_cell" else len(graph["cells"]))
                probabilities = np.zeros((len(splits["test"]), class_count), dtype=np.float32)
                for m in experts:
                    y, values, global_ids = class_space(labels, graph, task, decoder, m)
                    rows = np.arange(len(y)) if m is None else np.flatnonzero(labels["map_id"] == m)
                    local_splits = {name: np.flatnonzero(np.isin(rows, indices)) for name, indices in splits.items()}
                    local_dataset = {source: dataset[source][rows]}
                    local_labels = {task: y[rows]}
                    print(f"[probe] {len(jobs) + 1}/{total}: {decoder}/{task}/{source}/seed={seed}" +
                          (f"/map={m + 1}" if m is not None else ""), flush=True)
                    metadata, fitted = train_probe(task, source, seed, local_dataset, local_labels,
                                local_splits, graph, config, class_values=values)
                    slot = np.flatnonzero(np.isin(splits["test"], rows))
                    if m is None:
                        probabilities[:] = fitted["probabilities"]
                        add_group(output, prefix, fitted)
                    else:
                        target_ids = global_ids if global_ids is not None else np.arange(class_count)
                        probabilities[np.ix_(slot, target_ids)] = fitted["probabilities"]
                        add_group(output, f"{prefix}/maps/{m}", fitted)
                        metadata["map_id"] = m
                    jobs.append(metadata)
                    update_run(run_dir, jobs=jobs, progress={"stage": "probes", "completed": len(jobs), "total": total})
                # A single held-out output contract for all three alternatives.
                output[f"{prefix}/probabilities"] = probabilities
                output[f"{prefix}/predicted"] = probabilities.argmax(axis=1).astype(np.int32)
                save_data(run_dir, output)
    return output, jobs
