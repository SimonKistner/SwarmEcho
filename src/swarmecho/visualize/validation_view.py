"""Small validation result artifacts for the local inspector; no geometry loads."""
import json
from pathlib import Path


def active_validation_history(directory):
    """Apply session rollback markers without deleting historical result files."""
    rows = {}
    source = Path(directory) / "history.jsonl"
    if source.is_file():
        for line in source.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                if row.get("kind") == "session":
                    rows = {key: value for key, value in rows.items() if key <= row["update"]}
                elif row.get("kind") == "validation":
                    rows[row["update"]] = row
            except (ValueError, KeyError, TypeError):
                continue
    return rows


def discover_validations(root, *, single_run=False):
    root = Path(root)
    patterns = ["validation/results_u*.json"]
    if not single_run:
        patterns += ["*/validation/results_u*.json", "*/*/validation/results_u*.json"]
    candidates = {path for pattern in patterns for path in root.glob(pattern)}
    histories = {directory: active_validation_history(directory) for directory in {path.parent for path in candidates}}
    candidates = {path for path in candidates if not (path.parent / "history.jsonl").is_file()
                  or int(path.stem.removeprefix("results_u")) in histories[path.parent]}
    return sorted(candidates,
                  key=lambda path: path.stat().st_mtime_ns, reverse=True)


def validation_label(path):
    path = Path(path)
    result = json.loads(path.read_text(encoding="utf-8"))
    steps = int(result["global_step"])
    display = f"{steps/1e6:g}M" if steps >= 1000000 else f"{steps/1000:g}k" if steps >= 1000 else str(steps)
    return f"{path.parent.parent.name}_[{display}]_[Validation]"


def validation_payload(path):
    path = Path(path)
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("format") != "swarmecho-validation-result/v1":
        raise ValueError("Unknown validation result format.")
    active = active_validation_history(path.parent)
    if (path.parent / "history.jsonl").is_file() and result["update"] not in active:
        raise ValueError("This validation pass is outside the current resume timeline.")
    history = [row for row in active.values() if row.get("identity") == result["identity"]
               and row["update"] <= result["update"]]
    if not history:
        history = [result]
    return {**result, "kind": "validation", "history": history,
            "manifest": {"map_name": "Validation", "agents": result.get("agents"),
                         "training_update": result["update"], "environment_steps": result["global_step"]}}
