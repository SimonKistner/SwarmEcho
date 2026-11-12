"""Read-only PFSS export service, executed over a non-interactive SSH tunnel.

Requests are JSON lines. Responses are length-prefixed JSON, with raw file bytes
only after a successful get header. No training imports or remote writes on PFSS.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys
import tempfile


def send(value):
    data = json.dumps(value).encode("utf-8")
    sys.stdout.buffer.write(struct.pack("!Q", len(data)) + data)
    sys.stdout.buffer.flush()


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def signature(path):
    st = path.stat()
    return st.st_size, st.st_mtime_ns, st.st_ino


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def progress(folder):
    last_update = 0
    global_step = None
    history = folder / "training_history.jsonl"
    try:
        with history.open("rb") as stream:
            stream.seek(max(0, history.stat().st_size - 65536))
            for line in stream:
                try:
                    row = json.loads(line)
                    if "update" in row:
                        last_update = int(row["update"])
                        global_step = row.get("global_step")
                except (ValueError, TypeError):
                    pass
    except OSError:
        pass
    checkpoints = []
    for ready in (folder / "checkpoints").glob("*/recovery_ready.json"):
        if ready.parent.name.startswith("."):
            continue
        value = read_json(ready)
        if value:
            checkpoints.append({"name": ready.parent.name, "update": value.get("update"),
                                "final": value.get("final", False)})
    validations = {}
    history = folder / "validation/history.jsonl"
    if history.is_file():
        for line in history.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                if row.get("kind") == "session":
                    validations = {update: True for update in validations if update <= row["update"]}
                elif row.get("kind") == "validation":
                    validations[int(row["update"])] = True
            except (ValueError, KeyError, TypeError):
                pass
    else:
        validations = {int(p.stem[9:]): True for p in (folder / "validation").glob("results_u*.json")
                       if p.stem[9:].isdigit()}
    metrics = read_json(folder / "metrics.json")
    return {"update": last_update, "checkpoints": sorted(checkpoints, key=lambda x: x["name"]),
            "validation_update": max(validations, default=None),
            "global_step": global_step, "mastered": metrics.get("agent_curriculum_mastered") if isinstance(metrics, dict) else None,
            "complete": isinstance(metrics, dict)}


class Export:
    def __init__(self):
        self.paths = {}
        self.files = {}
        self.unstable = []

    def best_checkpoints(self, request):
        """Read published best markers for exact run names without exporting models."""
        root = Path(request["persist_root"]).resolve(strict=True)
        outputs = root / "outputs"
        records = []
        for name in request["run_names"]:
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                raise ValueError("Invalid run name")
            folder = outputs / name
            checkpoint = folder / "checkpoints/best_val_checkpoint"
            if folder.is_symlink() or checkpoint.is_symlink():
                raise ValueError("Refusing symlinked run/checkpoint")
            if checkpoint.exists() and outputs not in checkpoint.resolve().parents:
                raise ValueError("Checkpoint escapes output storage")
            ready = read_json(checkpoint / "recovery_ready.json")
            records.append({"run_name": name, "best": ready})
        return {"ok": True, "runs": records}

    def tree(self, folder, prefix, allowed_root, include=None):
        def selected(relative, directory=False):
            return include is None or any(relative == item or relative.startswith(item + "/")
                or (directory and item.startswith(relative + "/")) for item in include)

        for parent, dirs, names in os.walk(folder):
            parent = Path(parent)
            dirs[:] = sorted(d for d in dirs if d not in {"wandb", "__pycache__", ".git", ".cluster"}
                             and not d.startswith(".") and not d.endswith("_pending")
                             and not (parent / d).is_symlink()
                             and selected((parent / d).relative_to(folder).as_posix(), directory=True))
            if parent.parent.name == "checkpoints" and not (parent / "recovery_ready.json").is_file():
                dirs[:] = []
                continue
            for name in sorted(names):
                if name.endswith((".tmp", ".partial", ".lock")) or name.startswith(".nfs"):
                    continue
                path = parent / name
                if not selected(path.relative_to(folder).as_posix()):
                    continue
                try:
                    if path.is_symlink() or not path.is_file() or allowed_root not in path.resolve().parents:
                        continue
                    key = prefix + "/" + path.relative_to(folder).as_posix()
                    before = signature(path)
                    sha = digest(path)
                    if signature(path) != before:
                        self.unstable.append(key)
                        continue
                    self.paths[key] = path
                    self.files[key] = {"size": before[0], "mtime_ns": before[1], "sha256": sha}
                except OSError:
                    self.unstable.append(str(path))

    def evaluation_files(self, folder, checkpoint, count, root, pools):
        """Select evaluation dependencies before walking or hashing run/pool trees."""
        checkpoint_dir = folder / "checkpoints" / checkpoint
        if checkpoint_dir.is_symlink() or not (checkpoint_dir / "recovery_ready.json").is_file():
            raise ValueError("Checkpoint is missing or not published: " + str(checkpoint_dir))
        if not (folder / "config.yaml").is_file():
            raise ValueError("Evaluation requires the saved config.yaml: " + str(folder))
        include = {"config.yaml", "checkpoints/" + checkpoint}
        selection_path = folder / "adaptive_maps/selection.json"
        selection = read_json(selection_path, {})
        if selection:
            include.add("adaptive_maps/selection.json")
        bank_path = folder / "random_buildings.json"
        bank = read_json(bank_path, {})
        if bank_path.is_file():
            include.add("random_buildings.json")
        pool = None
        bank_pool = bank.get("pool")
        selection_pool = selection.get("pool") if selection.get("kind") == "bank" else None
        if bank_pool and selection_pool and Path(bank_pool).resolve() != Path(selection_pool).resolve():
            raise ValueError("Run bank and frozen selection reference different pools")
        if bank_pool or selection_pool:
            pool = Path(bank_pool or selection_pool).resolve()
            if root / "map_pools" not in pool.parents:
                raise ValueError("Refusing pool outside this project's map_pools: " + str(pool))
            pools[pool.as_posix()] = "CLU_" + hashlib.sha256(pool.as_posix().encode()).hexdigest()[:12] + "_" + pool.name
        static = not selection and (folder / "static_eval_maps").is_dir()
        directory, prefix = ("static_eval_maps", "static_eval") if static else ("random_eval_maps", "random_eval")
        if static and (folder / "static_maps.json").is_file():
            include.add("static_maps.json")
        pool_files = set()
        for index in range(count):
            relative = "{}/{}_{:04d}.yaml".format(directory, prefix, index)
            if (folder / relative).is_file():
                include.add(relative)
                continue
            maps = selection.get("evaluation_maps", [])
            if pool is None or index >= len(maps):
                raise ValueError("Missing frozen inspection map {} and no bank source; fetch fewer maps or use a full import.".format(relative))
            map_id = maps[index]["id"]
            if not isinstance(map_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", map_id):
                raise ValueError("Unsafe evaluation map ID")
            source = "maps/" + map_id + ".yaml"
            if not (pool / source).is_file():
                raise ValueError("Missing inspection map bank source: " + str(pool / source))
            pool_files.add(source)
        return include, pool, pool_files

    def inventory(self, request):
        self.paths, self.files, self.unstable = {}, {}, []
        root = Path(request["persist_root"]).resolve(strict=True)
        outputs = root / "outputs"
        prefix = request["remote_name"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", prefix):
            raise ValueError("Invalid run prefix")
        checkpoint = request.get("checkpoint")
        count = request.get("eval_maps", 3)
        if checkpoint is not None and (not isinstance(checkpoint, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", checkpoint)):
            raise ValueError("Checkpoint must be one directory name")
        if checkpoint is not None and (type(count) is not int or count < 1):
            raise ValueError("Evaluation map count must be positive")
        candidates = sorted(p for p in outputs.iterdir() if p.is_dir() and not p.is_symlink()
                            and (p.name == prefix or p.name.startswith(prefix + "_")
                                 or p.name.startswith("curr_" + prefix + "_"))) if outputs.is_dir() else []
        stages, pools, missing_pools = {}, {}, []
        pool_includes = {}
        journals, curriculum = {}, []
        for folder in candidates:
            for name in ("agent_curriculum.json", "target_curriculum.json"):
                journal = read_json(folder / name, {})
                if journal:
                    curriculum.append(journal)
                for item in journal.get("stages", []):
                    journals[item["run_name"]] = item
        for folder in candidates:
            # A provenance-only parent folder is not a training stage.
            if not any((folder / item).exists() for item in ("config.yaml", "checkpoints", "training_history.jsonl", "metrics.json", "tuning_run.json")):
                continue
            if checkpoint and not (folder / "checkpoints" / checkpoint).is_dir():
                continue
            stats = progress(folder)
            stats.update({"mastered": journals.get(folder.name, {}).get("mastered"),
                          "budget": journals.get(folder.name, {}).get("agents"), "pools": []})
            stages[folder.name] = stats
            if checkpoint:
                include, pool, pool_files = self.evaluation_files(folder, checkpoint, count, root, pools)
                self.tree(folder, "outputs/" + folder.name, root, include=include)
                required = ["outputs/" + folder.name + "/" + item for item in include
                            if not item.startswith("checkpoints/")]
                required.append("outputs/{}/checkpoints/{}/recovery_ready.json".format(folder.name, checkpoint))
                missing = [key for key in required if key not in self.files and key not in self.unstable]
                if missing:
                    raise ValueError("Could not safely export evaluation dependencies: " + ", ".join(missing))
                if pool is not None:
                    stats["pools"].append(pool.as_posix())
                    pool_includes.setdefault(pool.as_posix(), set()).update(pool_files)
                continue
            self.tree(folder, "outputs/" + folder.name, root)
            for record in (folder / "random_buildings.json", folder / "adaptive_maps/selection.json"):
                pool = read_json(record, {}).get("pool")
                if not pool:
                    continue
                path = Path(pool).resolve()
                if path.as_posix() not in stats["pools"]:
                    stats["pools"].append(path.as_posix())
                if root / "map_pools" not in path.parents:
                    raise ValueError("Refusing pool outside this project's map_pools: " + str(path))
                if not path.is_dir():
                    missing_pools.append(str(path))
                else:
                    pools[path.as_posix()] = "CLU_" + hashlib.sha256(path.as_posix().encode()).hexdigest()[:12] + "_" + path.name
        if checkpoint and not stages:
            raise ValueError("No published stage has checkpoint " + checkpoint)
        for pool, local_name in sorted(pools.items()):
            include = pool_includes.get(pool, set()) if checkpoint else None
            if include is None or include:
                self.tree(Path(pool), "pools/" + local_name, root, include=include)
                if checkpoint:
                    missing = ["pools/" + local_name + "/" + item for item in include
                               if "pools/" + local_name + "/" + item not in self.files
                               and "pools/" + local_name + "/" + item not in self.unstable]
                    if missing:
                        raise ValueError("Could not safely export evaluation map sources: " + ", ".join(missing))
        return {"ok": True, "stages": stages, "pools": pools, "missing_pools": missing_pools,
                "files": self.files, "unstable": self.unstable, "curriculum": curriculum}

    def get(self, key):
        path = self.paths[key]
        expected = self.files[key]
        before = signature(path)
        # Snapshot one file in temporary container storage, never on PFSS.
        with tempfile.SpooledTemporaryFile(max_size=16 * 1024 * 1024) as snapshot:
            sha = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    snapshot.write(block)
                    sha.update(block)
            if before != signature(path) or sha.hexdigest() != expected["sha256"]:
                send({"ok": False, "changed": True, "error": "File changed during export: " + key})
                return
            send({"ok": True, "size": snapshot.tell(), "sha256": sha.hexdigest()})
            snapshot.seek(0)
            for block in iter(lambda: snapshot.read(1024 * 1024), b""):
                sys.stdout.buffer.write(block)
            sys.stdout.buffer.flush()


def main():
    exporter = Export()
    for line in sys.stdin.buffer:
        try:
            request = json.loads(line)
            if request["action"] == "inventory":
                send(exporter.inventory(request))
            elif request["action"] == "best_checkpoints":
                send(exporter.best_checkpoints(request))
            elif request["action"] == "get":
                exporter.get(request["key"])
            else:
                raise ValueError("Unknown export action")
        except Exception as error:
            send({"ok": False, "error": str(error)})


if __name__ == "__main__":
    main()
