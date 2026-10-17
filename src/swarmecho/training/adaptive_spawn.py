"""CPU-side cell curriculum; no JAX callbacks and no W&B metrics.

One controller serves every lane on a fixed map. Journal rows describe the
probabilities applied during a rollout and the success estimate at its end.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def atomic_json(path: Path, payload: dict) -> None:
    """Readers see the complete old or new file, never a partial replacement."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class AdaptiveTargetSpawnController:
    def __init__(self, building, env, config, *, feasibility_budget=None):
        from swarmecho.env.buildings import target_spawn_boxes

        self.config = config
        self.feasibility_budget = feasibility_budget
        self.eligibility = None
        self.cell_size = float(building.cell_size_m)
        lower, upper, volumes = target_spawn_boxes(building, env)
        box_cells = np.floor(((lower + upper) * .5) / self.cell_size).astype(int)
        self.base_cell = np.floor(building.base_position_m / self.cell_size).astype(int)
        cells = set(map(tuple, box_cells.tolist()))
        self.cells = np.asarray(sorted(cells, key=lambda cell: (
            sum(abs(int(x) - int(y)) for x, y in zip(cell, self.base_cell)), *cell)), dtype=int)
        self.lookup = {tuple(cell): i for i, cell in enumerate(self.cells.tolist())}
        self.box_cell_indices = np.asarray([self.lookup[tuple(cell)] for cell in box_cells.tolist()])
        self.box_volumes = volumes.astype(np.float64)
        identity = hashlib.sha256()
        for array in (lower, upper, volumes, building.base_position_m):
            identity.update(np.asarray(array, dtype="<f8").tobytes())
        identity.update(str(env.success_condition).encode("utf-8"))
        if feasibility_budget is not None:
            identity.update(b"startup-cell-support-1pct-v1:")
            identity.update((f"relay-v2:{feasibility_budget}:{env.drone_radius}:"
                            f"{env.obstacle_planning_clearance_m}:{env.comm_radius_base}:"
                            f"{env.comm_radius}:{env.visual_radius}:{env.roadmap_approach}:"
                            f"{env.roadmap_node_density}:{env.roadmap_merge_wall_end_nodes}").encode("utf-8"))
        self.sampler_identity = identity.hexdigest()
        self.cell_volumes = np.bincount(self.box_cell_indices, weights=self.box_volumes,
                                       minlength=len(self.cells))
        self.baseline = self.cell_volumes / self.cell_volumes.sum()
        self.eligible = np.ones(len(self.cells), dtype=bool)
        self.probabilities = self.baseline.copy()
        self.windows = [deque(maxlen=config.success_window_episodes_per_cell) for _ in self.cells]
        self.total_episodes = np.zeros(len(self.cells), dtype=np.int64)
        self.rates = np.zeros(len(self.cells), dtype=np.float64)
        self.improvement_reference = self.rates.copy()
        self.previously_improved = False
        self.warning_history = deque(maxlen=config.plateau_warning_horizon_updates + 1)
        self.warned_at = np.full(len(self.cells), -1., dtype=np.float64)
        self.update = 0
        self.steps = 0
        self.warning_history.append({"update": 0, "probabilities": self.probabilities.tolist(),
                                     "success_rates": self.rates.tolist(),
                                     "completed_counts": self.total_episodes.tolist()})

    @property
    def eligibility_ready(self):
        return self.feasibility_budget is None or self.eligibility is not None

    def initialize_eligibility(self, target_positions, map_ids=None):
        """Mask cells from the already accepted, uniform startup reset batch."""
        if self.feasibility_budget is None or self.eligibility_ready:
            raise ValueError("Startup eligibility can only be initialized once for a filtered controller.")
        positions = np.asarray(target_positions).reshape(-1, 3)
        counts = np.zeros(len(self.cells), dtype=np.int64)
        for cell in np.floor(positions / self.cell_size).astype(int):
            index = self.lookup.get(tuple(cell))
            if index is None:
                raise ValueError(f"Startup target lies outside adaptive spawn cells: {cell.tolist()}")
            counts[index] += 1
        payload = {"format": "swarmecho-adaptive-spawn-eligibility/v1",
                   "sampler_identity": self.sampler_identity,
                   "agent_budget": self.feasibility_budget,
                   "cells": self.cells.tolist(),
                   "sample_count": len(positions), "minimum_fraction": .01,
                   "minimum_count": (len(positions) + 99) // 100,
                   "accepted_counts": counts.tolist()}
        self._apply_eligibility(payload)

    def _apply_eligibility(self, payload):
        if (payload.get("format") != "swarmecho-adaptive-spawn-eligibility/v1"
                or payload.get("sampler_identity") != self.sampler_identity
                or payload.get("agent_budget") != self.feasibility_budget
                or payload.get("cells") != self.cells.tolist()
                or payload.get("minimum_fraction") != .01):
            raise ValueError("Adaptive spawn eligibility does not match this stage's sampler and agent budget.")
        size = payload.get("sample_count")
        counts = np.asarray(payload.get("accepted_counts"))
        if (type(size) is not int or size < 1 or counts.shape != (len(self.cells),)
                or counts.dtype.kind not in "iu" or np.any(counts < 0)
                or counts.sum() != size or payload.get("minimum_count") != (size + 99) // 100):
            raise ValueError("Invalid adaptive spawn startup eligibility counts.")
        eligible = counts >= payload["minimum_count"]
        if not np.any(eligible):
            raise ValueError("No target cell reached 1% of the accepted startup batch; no adaptive spawn support remains.")
        self.eligible = eligible
        self.eligibility = payload
        self.baseline = self.cell_volumes * eligible
        self.baseline /= self.baseline.sum()
        self.probabilities = self.baseline.copy()
        self.warning_history[0]["probabilities"] = self.baseline.tolist()

    def restore_eligibility(self, path):
        """Reuse a completed startup discovery after OOM before a checkpoint."""
        path = Path(path)
        if not path.is_file():
            return False
        self._apply_eligibility(json.loads(path.read_text(encoding="utf-8")))
        return True

    def save_eligibility(self, path):
        if self.feasibility_budget is None:
            return
        if not self.eligibility_ready:
            raise ValueError("Adaptive spawn eligibility has not been initialized.")
        atomic_json(Path(path), self.eligibility)

    @property
    def box_weights(self):
        if not self.eligibility_ready:
            raise ValueError("Use uniform startup reset before requesting masked adaptive spawn weights.")
        # Normalizing within each cell avoids dependence on the number of slabs.
        mass = self.probabilities[self.box_cell_indices]
        weights = mass * self.box_volumes / self.cell_volumes[self.box_cell_indices]
        return weights.astype(np.float32)

    def metadata(self, run_name):
        if not self.eligibility_ready:
            raise ValueError("Do not publish adaptive probabilities before startup eligibility is ready.")
        return {"format": "swarmecho-adaptive-spawn/v1", "run_name": run_name,
                "cells": self.cells.tolist(), "base_cell": self.base_cell.tolist(),
                "cell_size_m": self.cell_size, "baseline": self.baseline.tolist(),
                "cell_volumes_m3": self.cell_volumes.tolist(), "config": asdict(self.config),
                "sampler_identity": self.sampler_identity,
                **({"eligibility": self.eligibility} if self.feasibility_budget is not None else {}),
                "probability_timing": "applied during rollout; success measured at rollout end"}

    def observe(self, target_positions, successes, map_ids=None):
        if not self.eligibility_ready:
            raise ValueError("Initialize adaptive spawn eligibility before observing policy outcomes.")
        positions = np.asarray(target_positions).reshape(-1, 3)
        successes = np.asarray(successes).reshape(-1)
        if len(positions) != len(successes):
            raise ValueError("Adaptive spawn positions and outcomes have different lengths.")
        cells = np.floor(positions / self.cell_size).astype(int)
        for cell, success in zip(cells, successes):
            index = self.lookup.get(tuple(cell))
            if index is None:
                raise ValueError(f"Completed target lies outside adaptive spawn cells: {cell.tolist()}")
            # Discovery targets are retained for their first episodes. Outcomes
            # in cells excluded by that discovery do not drive adaptation.
            if not self.eligible[index]:
                continue
            self.windows[index].append(int(bool(success)))
            self.total_episodes[index] += 1
        self.rates = np.asarray([window.success_rate if hasattr(window, "success_rate") else
                                 sum(window) / len(window) if window else 0.
                                 for window in self.windows])

    def advance(self, update, steps):
        """Compute next-rollout probabilities once, returning a truthful history row."""
        if not self.eligibility_ready:
            raise ValueError("Initialize adaptive spawn eligibility before advancing the controller.")
        applied = self.probabilities.copy()
        failure = np.where(self.eligible, 1. - self.rates, 0.)
        worst_failure = float(failure.max())
        if worst_failure == 0.:
            desired = self.baseline.copy()
        else:
            weighted_failure = self.baseline * failure
            f = self.config.min_probability_fraction
            desired = f * self.baseline + (1. - f) * (
                (1. - worst_failure) * self.baseline
                + worst_failure * weighted_failure / weighted_failure.sum())
        blend = -np.expm1(-np.log(2.) / self.config.probability_smoothing_half_life_updates)
        self.probabilities += blend * (desired - self.probabilities)
        self.probabilities[~self.eligible] = 0.
        # Convex interpolation preserves the floor, including after resume.
        self.probabilities /= self.probabilities.sum()
        self.update, self.steps = int(update), int(steps)
        row = {"kind": "update", "update": self.update, "steps": self.steps,
               "probabilities": applied.tolist(), "success_rates": self.rates.tolist(),
               "next_probabilities": self.probabilities.tolist(),
               "window_counts": [len(window) for window in self.windows],
               "completed_counts": self.total_episodes.tolist()}
        warnings = self._plateau_warnings(row) if self.config.plateau_warning_enabled else []
        return row, warnings

    def _plateau_warnings(self, row):
        threshold = self.config.plateau_warning_improvement_threshold_pp / 100.
        improved = self.eligible & (self.rates > self.improvement_reference + threshold)
        if np.any(improved):
            self.previously_improved = True
            self.improvement_reference[improved] = self.rates[improved]
        # A real gain, rather than elapsed time, rearms a continuing plateau.
        self.warned_at[self.rates > self.warned_at + threshold] = -1.
        self.warning_history.append({key: row[key] for key in (
            "update", "probabilities", "success_rates", "completed_counts")})
        horizon = self.config.plateau_warning_horizon_updates
        if not self.previously_improved or len(self.warning_history) < horizon + 1:
            return []
        start = self.warning_history[0]
        if self.update - start["update"] != horizon:
            return []
        initial = np.asarray(start["success_rates"])
        samples = list(self.warning_history)[1:]
        peaks = np.max([sample["success_rates"] for sample in samples], axis=0)
        mean_probability = np.mean([sample["probabilities"] for sample in samples], axis=0)
        fresh = self.total_episodes > np.asarray(start["completed_counts"])
        stalled = self.eligible & fresh & (initial + threshold < 1.) & (peaks <= initial + threshold) & (self.warned_at < 0.)
        indices = np.flatnonzero(stalled)
        if not len(indices):
            return []
        self.warned_at[indices] = initial[indices]
        messages = [f"[WARNING] Adaptive target spawning: {len(indices)}/{len(self.cells)} cells "
                    f"showed no improvement above {threshold * 100:g} percentage points "
                    f"over {horizon} PPO updates."]
        for index in indices:
            messages.append(f"  Cell {self.cells[index].tolist()}: success {initial[index]:.1%} "
                            f"→ {self.rates[index]:.1%}, peak {peaks[index]:.1%}; "
                            f"mean spawn probability {mean_probability[index]:.2%}; "
                            f"completed episodes {self.total_episodes[index] - start['completed_counts'][index]}.")
        return messages

    def state_dict(self, *, include_windows=True):
        if not self.eligibility_ready:
            raise ValueError("Cannot checkpoint adaptive spawn before startup eligibility is ready.")
        payload = {"format": 2 if self.feasibility_budget is not None else 1,
                   "config": asdict(self.config), "cells": self.cells.tolist(),
                   "sampler_identity": self.sampler_identity,
                   **({"eligibility": self.eligibility} if self.feasibility_budget is not None else {}),
                   "cell_volumes": self.cell_volumes.tolist(),
                   "probabilities": self.probabilities.tolist(),
                   **({"windows": [list(w) for w in self.windows]} if include_windows else {}),
                   "total_episodes": self.total_episodes.tolist(), "rates": self.rates.tolist(),
                   "improvement_reference": self.improvement_reference.tolist(),
                   "previously_improved": self.previously_improved,
                   "warning_history": list(self.warning_history), "warned_at": self.warned_at.tolist(),
                   "update": self.update, "steps": self.steps}
        return payload

    def save(self, checkpoint_path):
        atomic_json(Path(checkpoint_path) / "adaptive_spawn_state.json", self.state_dict())

    def restore(self, checkpoint_path):
        path = Path(checkpoint_path) / "adaptive_spawn_state.json"
        if not path.is_file():
            return False
        state = json.loads(path.read_text(encoding="utf-8"))
        return self.load_state_dict(state)

    def load_state_dict(self, state):
        expected_format = 2 if self.feasibility_budget is not None else 1
        if (state.get("format") != expected_format or state["config"] != asdict(self.config)
                or state.get("sampler_identity") != self.sampler_identity
                or state["cells"] != self.cells.tolist()
                or not np.allclose(state["cell_volumes"], self.cell_volumes, rtol=1e-7, atol=0.)):
            raise ValueError("Adaptive spawn resume requires matching controller configuration and spawn geometry; "
                             "use ckpt_loading_mode=branch to start a new controller.")
        if self.feasibility_budget is not None:
            self._apply_eligibility(state["eligibility"])
        count = len(self.cells)
        probabilities = np.asarray(state["probabilities"], dtype=np.float64)
        if (probabilities.shape != (count,) or not np.all(np.isfinite(probabilities))
                or not np.isclose(probabilities.sum(), 1.)
                or np.any(probabilities[~self.eligible] != 0.)
                or np.any(probabilities < self.config.min_probability_fraction * self.baseline - 1e-12)):
            raise ValueError("Invalid adaptive spawn checkpoint probability distribution.")
        if len(state["windows"]) != count or any(
            len(w) > self.config.success_window_episodes_per_cell or any(v not in (0, 1) for v in w)
            for w in state["windows"]
        ):
            raise ValueError("Invalid adaptive spawn checkpoint episode windows.")
        self.probabilities = probabilities
        self.windows = [deque(w, maxlen=self.config.success_window_episodes_per_cell) for w in state["windows"]]
        for name, dtype in (("total_episodes", np.int64), ("rates", np.float64),
                            ("improvement_reference", np.float64), ("warned_at", np.float64)):
            value = np.asarray(state[name], dtype=dtype)
            if value.shape != (count,) or not np.all(np.isfinite(value)):
                raise ValueError(f"Invalid adaptive spawn checkpoint {name}.")
            setattr(self, name, value)
        self.previously_improved = bool(state["previously_improved"])
        self.warning_history.clear()
        self.warning_history.extend(state["warning_history"])
        self.update, self.steps = int(state["update"]), int(state["steps"])
        return True


class MapAdaptiveTargetSpawnController:
    """Independent existing controllers, with one padded device weight table.

    Intended for small persistent static suites. The map dimension is indexed
    inside JAX reset; success windows and probability updates stay on the host.
    """
    def __init__(self, buildings, map_names, env, config, *, feasibility_budget=None, compact=False):
        if len(buildings) != len(map_names) or len(buildings) < (1 if compact else 2):
            raise ValueError("A map spawn controller requires matching names for at least two maps.")
        if len(set(map_names)) != len(map_names):
            raise ValueError("Adaptive spawn map names must be unique.")
        self.map_names = tuple(map_names)
        self.controllers = tuple(AdaptiveTargetSpawnController(
            building, env, config, feasibility_budget=feasibility_budget) for building in buildings)
        self.feasibility_budget = feasibility_budget
        self.update = self.steps = 0
        self.compact = compact
        if compact:
            from swarmecho.training.adaptive_maps import OutcomeWindows, PackedWindowView
            self.packed_windows = OutcomeWindows(sum(len(c.cells) for c in self.controllers),
                                                 config.success_window_episodes_per_cell)
            offset = 0
            for child in self.controllers:
                child.windows = [PackedWindowView(self.packed_windows, offset+i) for i in range(len(child.cells))]
                offset += len(child.cells)

    @property
    def eligibility_ready(self):
        return all(controller.eligibility_ready for controller in self.controllers)

    @property
    def box_weights(self):
        rows = [controller.box_weights for controller in self.controllers]
        result = np.zeros((len(rows), max(map(len, rows))), dtype=np.float32)
        for index, row in enumerate(rows):
            result[index, :len(row)] = row
        return result

    def _groups(self, target_positions, map_ids):
        positions = np.asarray(target_positions).reshape(-1, 3)
        if map_ids is None:
            raise ValueError("Multi-map adaptive spawning requires terminal map IDs.")
        ids = np.asarray(map_ids).reshape(-1)
        if (ids.dtype.kind not in "iu" or len(ids) != len(positions)
                or np.any(ids < 0) or np.any(ids >= len(self.controllers))):
            raise ValueError("Invalid map IDs for adaptive spawn outcomes.")
        return positions, ids

    def initialize_eligibility(self, target_positions, map_ids):
        positions, ids = self._groups(target_positions, map_ids)
        for index, controller in enumerate(self.controllers):
            selected = ids == index
            if not np.any(selected):
                raise ValueError(f"No startup discovery lanes for map {self.map_names[index]!r}.")
            controller.initialize_eligibility(positions[selected])

    def save_eligibility(self, path):
        if self.feasibility_budget is None:
            return
        if not self.eligibility_ready:
            raise ValueError("Initialize every map's spawn eligibility before saving.")
        atomic_json(Path(path), {"format": "swarmecho-adaptive-map-eligibility/v1",
            "map_names": list(self.map_names),
            "eligibility": [controller.eligibility for controller in self.controllers]})

    def restore_eligibility(self, path):
        path = Path(path)
        if not path.is_file():
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (payload.get("format") != "swarmecho-adaptive-map-eligibility/v1"
                or payload.get("map_names") != list(self.map_names)
                or len(payload.get("eligibility", [])) != len(self.controllers)):
            raise ValueError("Adaptive spawn eligibility map order changed; branch into a new run.")
        for controller, eligibility in zip(self.controllers, payload["eligibility"]):
            controller._apply_eligibility(eligibility)
        return True

    def observe(self, target_positions, successes, map_ids):
        positions, ids = self._groups(target_positions, map_ids)
        successes = np.asarray(successes).reshape(-1)
        if len(successes) != len(positions):
            raise ValueError("Adaptive spawn positions and outcomes have different lengths.")
        if self.compact:
            order = np.argsort(ids, kind="stable")
            keys, starts, counts = np.unique(ids[order], return_index=True, return_counts=True)
            for key, start, count in zip(keys, starts, counts):
                indices = order[start:start+count]
                self.controllers[key].observe(positions[indices], successes[indices])
            return
        for index, controller in enumerate(self.controllers):
            selected = ids == index
            controller.observe(positions[selected], successes[selected])

    def advance(self, update, steps):
        rows, warnings = [], []
        for name, controller in zip(self.map_names, self.controllers):
            row, messages = controller.advance(update, steps)
            rows.append(row)
            warnings.extend(f"[{name}] {message}" for message in messages)
        if self.compact and len(self.controllers) > 32 and len(warnings) > 20:
            warnings = [f"[WARNING] {len(warnings)} adaptive cell warning lines; showing the first 20.", *warnings[:20]]
        self.update, self.steps = int(update), int(steps)
        return {"kind": "map_update", "update": self.update, "steps": self.steps, "maps": rows}, warnings

    def save(self, checkpoint_path):
        if self.compact:
            from swarmecho.training.adaptive_maps import atomic_npz
            atomic_npz(Path(checkpoint_path) / "adaptive_cell_windows.npz", **self.packed_windows.arrays())
        atomic_json(Path(checkpoint_path) / "adaptive_spawn_state.json", {
            "format": "swarmecho-adaptive-map-spawn/v1", "map_names": list(self.map_names),
            "controllers": [controller.state_dict(include_windows=not self.compact) for controller in self.controllers],
            **({"packed_windows": True} if self.compact else {}),
            "update": self.update, "steps": self.steps})

    def restore(self, checkpoint_path):
        path = Path(checkpoint_path) / "adaptive_spawn_state.json"
        if not path.is_file():
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (payload.get("format") != "swarmecho-adaptive-map-spawn/v1"
                or payload.get("map_names") != list(self.map_names)
                or len(payload.get("controllers", [])) != len(self.controllers)):
            raise ValueError("Adaptive spawn checkpoint map order changed; branch into a new run.")
        packed = None
        if payload.get("packed_windows"):
            from swarmecho.training.adaptive_maps import OutcomeWindows
            packed = OutcomeWindows(sum(len(c.cells) for c in self.controllers),
                                    self.controllers[0].config.success_window_episodes_per_cell)
            with np.load(Path(checkpoint_path) / "adaptive_cell_windows.npz", allow_pickle=False) as data:
                packed.load(data)
        offset = 0
        for controller, state in zip(self.controllers, payload["controllers"]):
            if packed is not None:
                state["windows"] = [packed.values(offset+i) for i in range(len(controller.cells))]
            controller.load_state_dict(state)
            if self.compact:
                from swarmecho.training.adaptive_maps import PackedWindowView
                if packed is None:
                    for i, values in enumerate(state["windows"]):
                        for value in values:
                            PackedWindowView(self.packed_windows, offset+i).append(value)
                else:
                    self.packed_windows = packed
                controller.windows = [PackedWindowView(self.packed_windows, offset+i) for i in range(len(controller.cells))]
            offset += len(controller.cells)
        self.update, self.steps = int(payload["update"]), int(payload["steps"])
        if any(controller.update != self.update or controller.steps != self.steps
               for controller in self.controllers):
            raise ValueError("Adaptive spawn checkpoint maps have inconsistent update counters.")
        return True


class AdaptiveMapSpawnJournal:
    """One ordinary, dashboard-compatible journal per map; no pooled statistics."""
    def __init__(self, destination, controller, run_name, *, compact=False, **options):
        self.directory = Path(destination) / "adaptive_spawn_maps"
        self.compact = compact and len(controller.controllers) > 32
        if self.compact:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.path = self.directory / "history.jsonl"
            metadata = {"format": "swarmecho-adaptive-cell-bank/v1", "run_name": run_name,
                        "maps": [{"name": name, **child.metadata(run_name)}
                                 for name, child in zip(controller.map_names, controller.controllers)]}
            previous = self.directory / "metadata.json"
            if self.path.is_file() and self.path.stat().st_size:
                if not options.get("resume") or not previous.is_file() or json.loads(previous.read_text()) != metadata:
                    raise ValueError("Existing cell-bank journal changed; branch into a new run.")
                with self.path.open("rb+") as stream:
                    stream.seek(-1, os.SEEK_END)
                    if stream.read(1) != b"\n":
                        stream.seek(0, os.SEEK_END)
                        stream.write(b"\n")
            atomic_json(previous, metadata)
            for child in controller.controllers:
                if child.update == 0 and not np.any(child.total_episodes):
                    child.update, child.steps = int(options.get("update", 0)), int(options.get("steps", 0))
                    child.warning_history[0]["update"] = child.update
            controller.update, controller.steps = controller.controllers[0].update, controller.controllers[0].steps
            self._append({"kind": "session", "update": controller.update, "steps": controller.steps})
            return
        self.journals = tuple(AdaptiveSpawnJournal(
            self.directory / f"map_{index:04d}", child, f"{run_name} / {name}", **options)
            for index, (name, child) in enumerate(zip(controller.map_names, controller.controllers)))
        controller.update, controller.steps = controller.controllers[0].update, controller.controllers[0].steps

    def append(self, row):
        if self.compact:
            indices = np.linspace(0, len(row["maps"])-1, min(8, len(row["maps"])), dtype=int)
            self._append({"kind": "update", "update": row["update"], "steps": row["steps"],
                          "sampled_maps": [{"map_index": int(i), **row["maps"][i]} for i in indices]})
            return
        if len(row["maps"]) != len(self.journals):
            raise ValueError("Adaptive spawn journal map count changed.")
        for journal, map_row in zip(self.journals, row["maps"]):
            journal.append(map_row)

    def _append(self, row):
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


class AdaptiveSpawnJournal:
    """Append one small row per PPO update; no history rewrites or GPU transfers.

Closing each append makes it visible immediately. No per-update fsync: readers
ignore an incomplete last row after a crash. Sessions mark resume rollback
without erasing the original journal's provenance.
"""
    def __init__(self, destination, controller, run_name, *, resume=False, steps=0, update=0):
        self.directory = Path(destination) / "adaptive_spawn"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "history.jsonl"
        metadata_path = self.directory / "metadata.json"
        metadata = controller.metadata(run_name)
        if self.path.exists() and self.path.stat().st_size:
            if not resume or not metadata_path.exists():
                raise ValueError("Adaptive spawn history already exists; resume explicitly or choose a new run name.")
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            if previous != metadata:
                raise ValueError("Existing adaptive spawn journal has incompatible metadata; choose a new run name.")
            # Finish a crash-interrupted row so future complete rows remain readable.
            with self.path.open("rb+") as stream:
                stream.seek(-1, os.SEEK_END)
                if stream.read(1) != b"\n":
                    stream.seek(0, os.SEEK_END)
                    stream.write(b"\n")
        atomic_json(metadata_path, metadata)
        if controller.update == 0 and not np.any(controller.total_episodes):
            controller.update, controller.steps = int(update), int(steps)
            controller.warning_history[0]["update"] = int(update)
        self.append({"kind": "session", "steps": int(steps), "update": int(update)})
        self.append({"kind": "update", "steps": int(steps), "update": int(update),
                     "probabilities": controller.probabilities.tolist(),
                     "success_rates": controller.rates.tolist(),
                     "next_probabilities": controller.probabilities.tolist(),
                     "window_counts": [len(w) for w in controller.windows],
                     "completed_counts": controller.total_episodes.tolist()})

    def append(self, row):
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
