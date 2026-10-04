"""Small, path-free training summaries shared by local and public inspectors."""

from __future__ import annotations

import json
from pathlib import Path


def artifact_configuration(source: Path, payload: dict) -> dict:
    """Resolve the producing run's saved settings, never the current level YAML.

    Map counts describe this run/stage, not all earlier curriculum checkpoints.
    Only this small summary is published; config snapshots and paths stay local.
    """
    import yaml

    def positive_integer(value):
        return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None

    def read_json(path):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    manifest = payload["manifest"]
    agents = positive_integer(manifest.get("agents", manifest.get("num_agents")))
    if agents is None and payload.get("position"):
        agents = len(payload["position"][0]) or None
    summary = {"training_agents": None, "evaluation_agents": agents,
               "training_maps": {"kind": None, "count": None}}
    candidates = []
    checkpoint = manifest.get("checkpoint")
    if isinstance(checkpoint, str):
        checkpoint_path = Path(checkpoint)
        for path in (checkpoint_path, source.parent / checkpoint_path):
            if path.parent.name == "checkpoints" and path.is_dir():
                candidates.append(path.parent.parent)
    # Training replays need no explicit checkpoint path. This also covers moved
    # run folders whose original checkpoint path is no longer accessible.
    artifacts = next((parent for parent in source.parents if parent.name == "artifacts"), None)
    if artifacts is not None:
        candidates.append(artifacts.parent)
    record_names = ("config.yaml", "random_buildings.json", "static_maps.json")
    candidates.extend(parent for parent in source.parents
                      if any((parent / name).is_file() for name in record_names))
    run = next((path for path in candidates
                if any((path / name).is_file() for name in record_names)), None)
    if run is None:
        return summary
    try:
        config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        config = {}
    config = config if isinstance(config, dict) else {}
    env = config.get("env") if isinstance(config.get("env"), dict) else {}
    summary["training_agents"] = positive_integer(env.get("num_agents"))
    random = read_json(run / "random_buildings.json")
    static = read_json(run / "static_maps.json")
    settings = config.get("random_buildings")
    settings = settings if isinstance(settings, dict) else {}
    if isinstance(random.get("maps"), list) and random["maps"]:
        summary["training_maps"] = {"kind": "random", "count": len(random["maps"])}
    elif isinstance(static.get("maps"), list) and static["maps"]:
        # A selected map with zero assigned training lanes was not trained on.
        maps = [item for item in static["maps"] if isinstance(item, dict)
                and ("training_lanes" not in item or positive_integer(item["training_lanes"]))]
        summary["training_maps"] = {"kind": "static", "count": len(maps) or None}
    elif settings.get("enabled") is True:
        training = config.get("training") if isinstance(config.get("training"), dict) else {}
        summary["training_maps"] = {"kind": "random", "count": positive_integer(training.get("num_envs"))}
    elif config.get("building") and config["building"] != "random_buildings":
        names = env.get("map_names")
        count = len(set(names)) if isinstance(names, list) and all(isinstance(name, str) for name in names) and names else 1
        summary["training_maps"] = {"kind": "static", "count": count}
    return summary



