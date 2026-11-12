"""Import tracked PFSS runs, or one checkpoint and frozen evaluation dependencies.

Standard library only; no training imports.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import ExitStack, contextmanager
import ctypes
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import struct
import subprocess
import sys
import uuid

from submit import check_det_version
from tracking import ROOT, OUTPUTS, TRACK_FILE, discover, now, read_json, save_json

CLUSTER = ROOT / "cluster"
POOL_ROOT = ROOT / "src/swarmecho/curriculum_config/maps/random_building_bank/pools"
FETCH_ROOT = CLUSTER / ".fetch"
TEXT_SUFFIXES = {".json", ".jsonl", ".yaml", ".yml"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def remove_staging(path):
    path = Path(path).resolve()
    if FETCH_ROOT.resolve() not in path.parents:
        raise ValueError("Refusing cleanup outside cluster/.fetch")
    if path.exists():
        shutil.rmtree(path)


def command_argv(line):
    if sys.platform != "win32":
        return shlex.split(line)
    # det prints subprocess.list2cmdline, not POSIX shell syntax, on Windows.
    from ctypes import wintypes
    count = ctypes.c_int()
    parser = ctypes.windll.shell32.CommandLineToArgvW
    parser.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
    parser.restype = ctypes.POINTER(wintypes.LPWSTR)
    pointer = parser(line, ctypes.byref(count))
    if not pointer:
        raise OSError("Cannot parse Determined's SSH command")
    try:
        return [pointer[i] for i in range(count.value)]
    finally:
        ctypes.windll.kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        ctypes.windll.kernel32.LocalFree(pointer)


class Transfer:
    def __init__(self, settings, *, read_only=True, service_file=None):
        self.settings = settings
        self.read_only = read_only
        self.service_file = Path(service_file) if service_file is not None else CLUSTER / "tools/remote_fetch.py"
        self.det = shutil.which("det")
        if not self.det:
            raise RuntimeError("det is not on PATH")
        check_det_version(self.det, settings["determined_version"])
        self.env = dict(os.environ, DET_MASTER=settings["master"])
        self.shell_id = None
        self.process = None
        self.directory = FETCH_ROOT / ("connection-" + uuid.uuid4().hex)
        self.directory.mkdir(parents=True)
        self.error_log = None

    def cli(self, *args):
        result = subprocess.run([self.det, *args], env=self.env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True, timeout=600)
        if result.returncode:
            raise RuntimeError("det " + " ".join(args[:2]) + " failed: " + result.stderr + result.stdout)
        return re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)

    def __enter__(self):
        try:
            config = self.directory / "shell.json"
            save_json(config, {"description": "SwarmEcho output retrieval (zero GPU)" if self.read_only else "SwarmEcho W&B key setup (zero GPU)",
                       "resources": {"resource_pool": self.settings["resource_pool"], "slots": 0},
                       "bind_mounts": [{"host_path": self.settings["pfss_mount"],
                                        "container_path": self.settings["pfss_mount"], "read_only": self.read_only}]})
            print("Starting temporary zero-GPU transfer shell...", flush=True)
            result = self.cli("shell", "start", "--detach", "-w", self.settings["workspace"], "--config-file", str(config))
            match = re.search(r"\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", result)
            if not match:
                raise RuntimeError("Could not identify the created shell. Inspect det shell list: " + result)
            self.shell_id = match.group()
            save_json(self.directory / "connection.json", {"shell_id": self.shell_id, "master": self.settings["master"]})
            print("Waiting for transfer shell " + self.shell_id + "...", flush=True)
            result = self.cli("shell", "show-ssh-command", self.shell_id)
            line = next((line for line in result.splitlines() if line.lstrip().startswith("ssh ")), None)
            if line is None:
                raise RuntimeError("No SSH command returned: " + result)
            args = command_argv(line.strip())
            args = [arg for arg in args if arg != "-tt"]
            args[0] = shutil.which("ssh") or "ssh"
            args[1:1] = ["-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4"]
            encoded = base64.b64encode(self.service_file.read_bytes()).decode("ascii")
            remote = "import base64;exec(base64.b64decode(" + repr(encoded) + "))"
            args.append("python3 -u -c " + shlex.quote(remote))
            self.error_log = (self.directory / "ssh.log").open("wb")
            self.process = subprocess.Popen(args, env=self.env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.error_log)
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def __exit__(self, *args):
        try:
            if self.process is not None:
                try:
                    self.process.stdin.close()
                except OSError:
                    pass
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    self.process.wait(timeout=10)
            if self.error_log:
                self.error_log.close()
        finally:
            if self.shell_id:
                try:
                    self.cli("shell", "kill", self.shell_id)
                    print("Released transfer shell " + self.shell_id, flush=True)
                except Exception as error:
                    print("SHELL CLEANUP NEEDED: det shell kill " + self.shell_id + "; " + str(error), file=sys.stderr)

    def exact(self, count):
        data = bytearray()
        while len(data) < count:
            block = self.process.stdout.read(min(count - len(data), 1024 * 1024))
            if not block:
                raise RuntimeError("Transfer connection closed. See " + str(self.directory / "ssh.log"))
            data.extend(block)
        return bytes(data)

    def request(self, value):
        self.process.stdin.write((json.dumps(value) + "\n").encode("utf-8"))
        self.process.stdin.flush()
        length = struct.unpack("!Q", self.exact(8))[0]
        if length > 256 * 1024 * 1024:
            raise RuntimeError("Unexpected SSH output; see " + str(self.directory / "ssh.log"))
        result = json.loads(self.exact(length))
        if not result.get("ok") and not result.get("changed"):
            raise RuntimeError(result.get("error", "Remote export failed"))
        return result

    def inventory(self, state, *, checkpoint=None, eval_maps=3):
        request = {"action": "inventory", "persist_root": state["settings"]["persist_root"],
                   "remote_name": state["remote_name"]}
        if checkpoint:
            request.update(checkpoint=checkpoint, eval_maps=eval_maps)
        result = self.request(request)
        response = json.loads(self.cli("experiment", "describe", str(state["experiment_id"]), "--json"))
        exp = response[0] if isinstance(response, list) else response
        result["experiment_state"] = exp.get("experiment", exp).get("state", "UNKNOWN")
        result["observed_at"] = now()
        return result

    def download(self, key, path, metadata):
        result = self.request({"action": "get", "key": key})
        if not result["ok"]:
            print("Deferred changing file: " + key)
            return False
        if result["size"] != metadata["size"] or result["sha256"] != metadata["sha256"]:
            raise RuntimeError("Export metadata mismatch: " + key)
        path.parent.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256()
        remaining = result["size"]
        with path.open("wb") as stream:
            while remaining:
                block = self.exact(min(remaining, 1024 * 1024))
                stream.write(block)
                h.update(block)
                remaining -= len(block)
        if h.hexdigest() != metadata["sha256"]:
            raise RuntimeError("Download checksum mismatch: " + key)
        return True


def stage_name(state, remote):
    prefix = state["remote_name"]
    if remote == prefix:
        return state["name"]
    if remote.startswith(prefix + "_"):
        return state["name"] + remote[len(prefix):]
    if remote.startswith("curr_" + prefix + "_"):
        return state["name"] + remote[len("curr_" + prefix):]
    raise ValueError("Unrelated stage: " + remote)


def target_path(state, key):
    parts = PurePosixPath(key).parts
    if len(parts) < 3 or any(p in {"..", "."} or ":" in p or "\\" in p for p in parts):
        raise ValueError("Unsafe remote file key: " + key)
    if parts[0] == "outputs":
        target = OUTPUTS / stage_name(state, parts[1]) / Path(*parts[2:])
        boundary = OUTPUTS
    elif parts[0] == "pools" and parts[1].startswith("CLU_"):
        target = POOL_ROOT / Path(*parts[1:])
        boundary = POOL_ROOT
    else:
        raise ValueError("Unexpected remote file: " + key)
    if boundary.resolve() not in target.resolve().parents or any(p.is_symlink() for p in [target, *target.parents] if p != ROOT.parent):
        raise ValueError("Destination escapes local output storage: " + str(target))
    return target


def replacements(state, inventory):
    persist = state["settings"]["persist_root"].rstrip("/")
    pairs = [(persist + "/outputs/" + name, "outputs/" + stage_name(state, name)) for name in inventory["stages"]]
    pairs.extend((pool, POOL_ROOT.relative_to(ROOT).as_posix() + "/" + name) for pool, name in inventory["pools"].items())
    pairs.append((persist + "/outputs", "outputs"))
    return sorted(pairs, key=lambda pair: len(pair[0]), reverse=True)


def relocate(path, pairs, suffix):
    if suffix not in TEXT_SUFFIXES:
        return
    text = path.read_text(encoding="utf-8")
    for remote, local in pairs:
        text = text.replace(remote, local)
    path.write_bytes(text.encode("utf-8"))


def load_state(path):
    state = read_json(path)
    journal = path.parent / ".cluster/fetch_journal.jsonl"
    if journal.exists():
        receipts = {}
        for line in journal.read_text(encoding="utf-8").splitlines():
            try:
                receipts.update(json.loads(line)["files"])
            except ValueError:
                pass
        for key, value in receipts.items():
            target = target_path(state, key)
            # A write-ahead receipt is committed only if promotion actually happened.
            if target.is_file() and sha256(target) == value["local_sha256"]:
                state["files"][key] = value
    return state


def record_files(path, state, files, apply=True):
    journal = path.parent / ".cluster/fetch_journal.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    needs_newline = False
    if journal.exists() and journal.stat().st_size:
        with journal.open("rb") as stream:
            stream.seek(-1, 2)
            needs_newline = stream.read(1) != b"\n"
    with journal.open("a", encoding="utf-8") as stream:
        if needs_newline:
            stream.write("\n")
        stream.write(json.dumps({"utc": now(), "files": files}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    if apply:
        state["files"].update(files)


@contextmanager
def run_lock(path):
    lock = path.parent / ".cluster/fetch.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+b") as stream:
        if lock.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError("Another fetch is already importing " + path.parent.name)
        try:
            yield
        finally:
            stream.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def entry(key, metadata, source):
    return {"remote_sha256": metadata["sha256"], "local_sha256": sha256(source),
            "size": metadata["size"], "imported_at": now()}


def choose(options, labels, title, allow_all=True):
    print("\n" + title)
    for index, label in enumerate(labels, 1):
        print("  {}  {}".format(index, label))
    if allow_all:
        print("  {}  all".format(len(options) + 1))
    print("  0  cancel")
    while True:
        raw = input("Select: ").strip()
        if raw == "0":
            return []
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return [options[int(raw) - 1]]
        if allow_all and (raw.lower() == "all" or raw == str(len(options) + 1)):
            return options
        print("Enter a listed number.")


def checkpoint_group(key):
    parts = key.split("/")
    return "/".join(parts[:4]) if len(parts) >= 5 and parts[0] == "outputs" and parts[2] == "checkpoints" else None


def local_progress(folder):
    # This helper has only standard-library imports; it does not execute its service.
    from remote_fetch import progress
    return progress(folder)


def sync_run(connection, path, select_stages, *, checkpoint=None, eval_maps=3):
    state = load_state(path)
    print("\nInspecting ex{} {}...".format(state["experiment_id"], state["name"]), flush=True)
    inventory = connection.inventory(state, checkpoint=checkpoint, eval_maps=eval_maps)
    state["experiment_state"] = inventory["experiment_state"]
    state["remote_observed_at"] = inventory["observed_at"]
    state["missing_pools"] = inventory["missing_pools"]
    state["remote_curriculum"] = inventory.get("curriculum", [])
    names = list(inventory["stages"])
    if not names:
        state["last_attempt"] = now()
        save_json(path, state)
        print("No stage outputs yet; experiment state: " + state["experiment_state"])
        return
    for name in names:
        local = local_progress(OUTPUTS / stage_name(state, name))
        remote = inventory["stages"][name]
        remote["status"] = ("FINISHED" if remote["complete"] else
                            "INTERRUPTED" if state["experiment_state"] in {"STATE_ERROR", "STATE_CANCELED", "STATE_COMPLETED", "STATE_DELETED"}
                            else "IN_PROGRESS")
        previous = state["stages"].get(name, {})
        state["stages"][name] = dict(previous, remote=remote, local=local,
                                    local_name=stage_name(state, name),
                                    observed_at=inventory["observed_at"], synced_to_observation=False)
        print("  {} | remote update {}{} | local update {}{}".format(name, remote["update"],
              " (finished)" if remote["complete"] else "", local["update"], " (finished)" if local["complete"] else ""))
    save_json(path, state)
    selected = choose(names, names, "Select curriculum stage to fetch") if select_stages and len(names) > 1 else names
    if not selected:
        return
    needed_pools = {pool for name in selected for pool in inventory["stages"][name].get("pools", [])}
    pool_names = {inventory["pools"][pool] for pool in needed_pools if pool in inventory["pools"]}
    wanted = {key: meta for key, meta in inventory["files"].items()
              if (key.startswith("pools/") and key.split("/")[1] in pool_names)
              or (key.startswith("outputs/") and key.split("/")[1] in selected)}
    if checkpoint:
        total_bytes = sum(meta["size"] for meta in wanted.values())
        print("Evaluation-only fetch: {}, {} inspection map(s), {} files ({:.1f} MiB before skipping existing files).".format(
            checkpoint, eval_maps, len(wanted), total_bytes / (1024 * 1024)), flush=True)
    pairs = replacements(state, inventory)
    work = FETCH_ROOT / ("import-" + uuid.uuid4().hex)
    work.mkdir(parents=True)
    groups = {}
    for key in wanted:
        groups.setdefault(checkpoint_group(key) or key, []).append(key)
    changed, skipped, conflicts, deferred = 0, 0, [], []
    print("Comparing {} remote files; unchanged imports need no download...".format(len(wanted)), flush=True)
    try:
        for group, keys in groups.items():
            is_checkpoint = checkpoint_group(keys[0]) is not None
            need, collision, before_local = [], False, {}
            for key in keys:
                target = target_path(state, key)
                old = state["files"].get(key)
                current = sha256(target) if target.is_file() else None
                before_local[key] = current
                if current is not None and old and current != old["local_sha256"]:
                    conflicts.append(key)
                    collision = True
                elif current is not None and not old:
                    if current == wanted[key]["sha256"]:
                        record_files(path, state, {key: entry(key, wanted[key], target)})
                        old = state["files"][key]
                    else:
                        # Never replace an untracked or locally created diagnostic file.
                        conflicts.append(key)
                        collision = True
                if current is None or not old or old["remote_sha256"] != wanted[key]["sha256"]:
                    need.append(key)
            if collision:
                continue
            if not need:
                skipped += len(keys)
                continue
            if is_checkpoint:
                staged = work / "checkpoint"
                remove_staging(staged)
                destination = target_path(state, keys[0]).parents[len(keys[0].split("/")) - 5]
                if destination.exists():
                    shutil.copytree(destination, staged)
                else:
                    staged.mkdir()
                # Remove only previous cluster-owned checkpoint members from the new staging copy.
                for old_key in list(state["files"]):
                    if checkpoint_group(old_key) == group and old_key not in wanted:
                        candidate = staged / Path(*old_key.split("/")[4:])
                        if candidate.is_file() and sha256(candidate) == state["files"][old_key]["local_sha256"]:
                            candidate.unlink()
                sources = {key: staged / Path(*key.split("/")[4:]) for key in keys}
            else:
                sources = {key: work / "file" for key in keys}
            ready = True
            for key in need:
                if not connection.download(key, sources[key], wanted[key]):
                    deferred.append(key)
                    ready = False
                    break
                relocate(sources[key], pairs, PurePosixPath(key).suffix)
            if not ready:
                continue
            for key in keys:
                target = target_path(state, key)
                current = sha256(target) if target.is_file() else None
                if current != before_local[key]:
                    conflicts.append(key)
                    ready = False
            if not ready:
                continue
            records = {key: entry(key, wanted[key], sources[key]) for key in (keys if is_checkpoint else need)}
            # Write-ahead receipt: interrupted promotion can be retried from hashes.
            record_files(path, state, records, apply=False)
            if is_checkpoint:
                # Verify no local checkpoint members changed since comparison.
                destination.parent.mkdir(parents=True, exist_ok=True)
                backup = work / "previous-checkpoint"
                remove_staging(backup)
                if destination.exists():
                    destination.rename(backup)
                try:
                    staged.rename(destination)
                except BaseException:
                    if backup.exists() and not destination.exists():
                        backup.rename(destination)
                    raise
                remove_staging(backup)
            else:
                for key in need:
                    target = target_path(state, key)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    sources[key].replace(target)
            state["files"].update(records)
            changed += len(need)
            if changed and changed % 50 < len(need):
                print("  Imported {} new/updated files...".format(changed), flush=True)
        for name in selected:
            local_name = stage_name(state, name)
            folder = OUTPUTS / local_name
            folder.mkdir(parents=True, exist_ok=True)
            save_json(folder / "cluster_stage.json", {"format": "swarmecho-cluster-stage/v1",
                "experiment_id": state["experiment_id"], "parent": state["name"],
                "remote_name": name, "local_name": local_name})
            keys = [key for key in wanted if key.startswith("outputs/" + name + "/") or key.startswith("pools/")]
            missing = [key for key in keys if not target_path(state, key).is_file()
                       or state["files"].get(key, {}).get("remote_sha256") != wanted[key]["sha256"]]
            if checkpoint:
                state["stages"][name].setdefault("eval_fetches", {})[checkpoint] = {
                    "eval_maps": eval_maps, "observed_at": inventory["observed_at"],
                    "last_fetch": now(), "missing_files": len(missing),
                    "complete": not missing and not any(k in conflicts for k in keys)
                                and not inventory["unstable"],
                }
                state["stages"][name]["local"] = local_progress(folder)
                complete = state["stages"][name]["eval_fetches"][checkpoint]["complete"]
                print("Evaluation fetch {}: {} / {} with {} inspection map(s).".format(
                    "complete" if complete else "incomplete", local_name, checkpoint, eval_maps), flush=True)
                continue
            state["stages"][name].update({"local": local_progress(folder), "last_fetch": now(),
                "missing_files": len(missing), "conflicts": [k for k in conflicts if k in keys],
                "synced_to_observation": not missing and not any(k in conflicts for k in keys)
                                          and not any(pool in inventory["missing_pools"] for pool in inventory["stages"][name].get("pools", []))
                                          and not inventory["unstable"]})
        state["last_attempt"] = now()
        if changed or skipped:
            state["last_fetch"] = now()
        state["last_conflicts"] = conflicts
        state["deferred_files"] = deferred + inventory["unstable"]
        state["path_map"] = dict(pairs)
        save_json(path, state)
        print("Imported {} new/updated files; {} unchanged; {} conflicts; {} changing files deferred.".format(
              changed, skipped, len(conflicts), len(state["deferred_files"])))
        if conflicts:
            print("Local edits preserved. See last_conflicts in " + str(path))
        if inventory["missing_pools"]:
            print("Referenced remote map pools are missing; see missing_pools in " + str(path))
    finally:
        # Keep the record even when an import is interrupted. Never delete local run outputs.
        save_json(path, state)
        remove_staging(work)


def print_status(paths):
    for path in paths:
        state = load_state(path)
        print("ex{} {} | {} | last fetch {}".format(state["experiment_id"], state["name"],
              state["experiment_state"], state["last_fetch"] or "never"))
        for name, stage in state["stages"].items():
            print("  {}: remote update {}, local update {}; synced={} (as of {})".format(name,
                  stage["remote"]["update"], stage["local"]["update"], stage.get("synced_to_observation", False),
                  stage["observed_at"]))
            print("    checkpoints remote/local: {}/{}; validation update remote/local: {}/{}".format(
                  len(stage["remote"]["checkpoints"]), len(stage["local"]["checkpoints"]),
                  stage["remote"]["validation_update"], stage["local"]["validation_update"]))
            for checkpoint, fetched in stage.get("eval_fetches", {}).items():
                print("    evaluation {}: {} map(s), complete={} (as of {})".format(
                    checkpoint, fetched["eval_maps"], fetched["complete"], fetched["observed_at"]))
        if state.get("remote_curriculum"):
            for journal in state["remote_curriculum"]:
                budgets = [str(item["agents"]) for item in journal.get("stages", [])]
                print("  {} curriculum {}: observed budgets {}; maximum {}".format(
                    "Target" if journal.get("target_only") else "Agent",
                      journal.get("level", ""), ", ".join(budgets) or "none", journal.get("maximum_agents", "unknown")))


def menu_label(path):
    state = read_json(path)
    if state["last_fetch"] is None:
        progress = "not fetched"
    elif any(not stage.get("synced_to_observation", False) for stage in state["stages"].values()):
        progress = "partially fetched / behind last observation"
    else:
        progress = "fetched through last observation"
    return "ex{} {} [{}; {}]".format(state["experiment_id"], state["name"], progress, state["experiment_state"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--choose", action="store_true")
    parser.add_argument("--run", help="Tracked local folder name or recorded experiment ID")
    parser.add_argument("--checkpoint", help="Evaluation-only import: one checkpoint directory and inspection maps")
    parser.add_argument("--eval-maps", type=int, default=None,
                        help="Inspection map count for --checkpoint (default: 3; match the evaluation config)")
    parser.add_argument("--select-stages", action="store_true")
    parser.add_argument("--status", action="store_true", help="Last observed status; no network")
    args = parser.parse_args()
    if sum(bool(value) for value in (args.all, args.choose, args.run)) > 1:
        parser.error("Use only one of --all, --choose or --run")
    if args.checkpoint and not re.fullmatch(r"[A-Za-z0-9_-]+", args.checkpoint):
        parser.error("--checkpoint must be one directory name, such as ckpt_001050")
    if args.eval_maps is not None and (not args.checkpoint or args.eval_maps < 1):
        parser.error("--eval-maps requires --checkpoint and a positive count")
    if args.checkpoint and args.status:
        parser.error("--checkpoint cannot be combined with --status")
    paths = discover()
    if not paths:
        print("No successfully submitted cluster runs are tracked yet.")
        return
    if args.status:
        print_status(paths)
        return
    if args.run:
        paths = [p for p in paths if p.parent.name == args.run or str(read_json(p)["experiment_id"]) == args.run]
        if not paths:
            raise ValueError("No tracked run matches " + args.run)
    elif not args.all:
        pending = [p for p in paths if read_json(p).get("last_fetch") is None]
        if len(pending) == 1 and not args.choose:
            paths = pending
            print("Selecting the only never-fetched run: " + paths[0].parent.name)
        elif len(paths) > 1 or args.choose:
            labels = [menu_label(p) for p in paths]
            paths = choose(paths, labels, "Select cluster experiment to fetch")
    # Share one owned transfer shell across all runs on the same master/workspace/mount.
    with ExitStack() as stack:
        connections = {}
        for path in paths:
            settings = read_json(path)["settings"]
            key = settings["master"], settings["workspace"], settings["pfss_mount"]
            if key not in connections:
                connections[key] = stack.enter_context(Transfer(settings))
            with run_lock(path):
                sync_run(connections[key], path, args.select_stages,
                         checkpoint=args.checkpoint, eval_maps=args.eval_maps or 3)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.SubprocessError, EOFError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nRetrieval interrupted; completed imports are retained. Run fetch again to continue.", file=sys.stderr)
        sys.exit(130)
