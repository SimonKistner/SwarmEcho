"""One numeric archive and one manifest; no script copies or figure files."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np

from geo_experiments.common import ROOT, SOURCES, TASKS, fingerprint, write_json

HISTORY_COLUMNS = ("epoch", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy")
FORMAT = "swarmecho-geo-run/v2"


def save_data(run_dir, arrays):
    temporary = Path(run_dir) / ".data.tmp.npz"
    np.savez_compressed(temporary, **arrays)
    temporary.replace(Path(run_dir) / "data.npz")


def update_run(run_dir, **changes):
    path = Path(run_dir) / "run.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"format": FORMAT}
    manifest.update(changes)
    temporary = Path(run_dir) / ".run.tmp.json"
    write_json(temporary, manifest)
    temporary.replace(path)
    return manifest


def group(arrays, prefix):
    prefix += "/"
    return {name[len(prefix):]: value for name, value in arrays.items() if name.startswith(prefix)}


def add_group(arrays, prefix, values):
    arrays.update({f"{prefix}/{name}": value for name, value in values.items()})


def archive(path):
    with np.load(path, allow_pickle=False) as stream:
        return {name: stream[name] for name in stream.files}


def load_run(run_dir):
    """Also read the original layout so existing results need no retraining."""
    run_dir = Path(run_dir)
    if (run_dir / "run.json").exists() and (run_dir / "data.npz").exists():
        manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        if manifest.get("format") != FORMAT:
            raise ValueError("Unsupported experiment data format.")
        arrays = {}
        if manifest.get("shared_data"):
            shared = (run_dir / manifest["shared_data"]).resolve()
            if not shared.is_relative_to((ROOT / "geo_experiments").resolve()):
                raise ValueError("Shared collections must stay inside geo_experiments/.")
            arrays.update(archive(shared))
        arrays.update(archive(run_dir / "data.npz"))
        return manifest, arrays, False

    import yaml
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    collection = json.loads((run_dir / "dataset.json").read_text(encoding="utf-8"))
    identity = collection.pop("identity")
    provenance["identity"] = identity
    arrays = {}
    for prefix, filename in (("dataset", "dataset.npz"), ("geometry", "geometry.npz"),
                             ("labels", "labels.npz"), ("splits", "splits.npz")):
        add_group(arrays, prefix, archive(run_dir / filename))
    episodes = run_dir / "evaluation_episodes.csv"
    if episodes.exists():
        with episodes.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        add_group(arrays, "episodes", {
            "episode_id": np.asarray([int(r["episode_id"]) for r in rows], dtype=np.uint32),
            "success": np.asarray([r["success"].lower() == "true" for r in rows]),
            "complete_sample": np.asarray([r["complete_sample"].lower() == "true" for r in rows]),
            "episode_length": np.asarray([int(r["episode_length"]) for r in rows], dtype=np.int32),
            "target_position": np.asarray([[float(r[f"target_{a}"]) for a in "xyz"] for r in rows], dtype=np.float32),
        })
    jobs = []
    for task in TASKS:
        for source in SOURCES:
            for seed in config["probe"]["seeds"]:
                directory = run_dir / "probes" / task / source / f"seed_{seed}"
                result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
                result.pop("metrics", None)
                jobs.append(result)
                prefix = f"probes/{task}/{source}/seed_{seed}"
                prediction = archive(directory / "predictions.npz")
                add_group(arrays, prefix, {k: prediction[k] for k in ("predicted", "probabilities")})
                add_group(arrays, prefix + "/model", archive(directory / "model.npz"))
                with (directory / "history.csv").open(encoding="utf-8", newline="") as stream:
                    history = list(csv.DictReader(stream))
                arrays[prefix + "/history"] = np.asarray([[float(row[k]) for k in HISTORY_COLUMNS] for row in history])
    definitions = {name: (run_dir / name).read_text(encoding="utf-8")
                   for name in ("level.yaml", "map.yaml") if (run_dir / name).exists()}
    source_dir = run_dir / "source"
    provenance["experiment_files"] = {p.name: fingerprint(p) for p in source_dir.glob("*.py")}
    manifest = {"format": FORMAT, "config": config, "provenance": provenance,
                "definitions": definitions, "collection": collection, "jobs": jobs,
                "history_columns": HISTORY_COLUMNS, "progress": {"stage": "complete"}}
    return manifest, arrays, True


def prune_legacy(run_dir):
    """Remove only known generated artifacts, after compact files are saved."""
    run_dir = Path(run_dir).resolve()
    geo_root = (ROOT / "geo_experiments").resolve()
    if run_dir == geo_root or not run_dir.is_relative_to(geo_root):
        raise ValueError("Legacy cleanup must stay inside an experiment run in geo_experiments/.")
    for name in ("source", "figures", "probes", "baselines"):
        path = run_dir / name
        if path.is_symlink() or not path.resolve().is_relative_to(run_dir):
            raise ValueError(f"Refusing cleanup outside the run: {path}")
        if path.is_dir():
            shutil.rmtree(path)
    for name in ("config.yaml", "level.yaml", "map.yaml", "provenance.json", "dataset.npz", "dataset.json",
                 "labels.npz", "splits.npz", "geometry.npz", "class_counts.csv", "storey_counts.csv",
                 "evaluation_episodes.csv", "metrics_by_seed.csv", "metrics_summary.csv", "metrics_summary.json",
                 "dataset_summary.json", "report_settings.json", "report.md", "status.json"):
        path = run_dir / name
        if path.is_symlink() or not path.resolve().is_relative_to(run_dir):
            raise ValueError(f"Refusing cleanup outside the run: {path}")
        if path.is_file():
            path.unlink()
