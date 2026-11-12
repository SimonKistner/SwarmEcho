"""Fit separate readouts from a definition against immutable collected samples."""
from geo_experiments.shared.artifacts import add_group, group, save_data, update_run
from geo_experiments.shared.axes import experiment_axes
from geo_experiments.shared.targets import encode_output
from geo_experiments.shared.training import train_probe
from geo_experiments.shared.console import terminal_print


def fit_experiment(config, shared, run_dir, *, shared_reference=True):
    if config.get("experiment") == "path":
        from geo_experiments.shared.mask_training import fit_masks
        return fit_masks(config, shared, run_dir, shared_reference=shared_reference)
    dataset, labels, splits, graph = (group(shared, key) for key in ("dataset", "labels", "splits", "geometry"))
    tasks, sources = experiment_axes(config)
    y, values = encode_output(config, labels, graph)
    output, jobs = ({} if shared_reference else dict(shared)), []
    total = len(tasks) * len(sources) * len(config["probe"]["seeds"])
    for task in tasks:
        for source in sources:
            for seed in config["probe"]["seeds"]:
                terminal_print(f"[probe] {len(jobs) + 1}/{total}: {task}/{source}/seed={seed}", flush=True)
                metadata, fitted = train_probe(task, source, seed, dataset, {task: y}, splits,
                                                graph, config, class_values=values)
                add_group(output, f"probes/{task}/{source}/seed_{seed}", fitted)
                jobs.append(metadata)
                save_data(run_dir, output)
                update_run(run_dir, jobs=jobs, progress={"stage": "probes", "completed": len(jobs), "total": total})
    return output, jobs
