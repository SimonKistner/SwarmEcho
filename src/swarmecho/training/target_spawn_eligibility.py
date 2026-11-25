"""Frozen cell support and requirement histograms, independent of policy success."""
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from swarmecho.training.adaptive_spawn import atomic_json


@contextmanager
def cache_lock(directory):
    """One builder across sweep workers; OS releases the lock on worker exit."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "prepare.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            if handle.tell() == 0:
                handle.write(b"\0"); handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class TargetSpawnEligibility:
    format = "swarmecho-target-spawn-eligibility/v1"

    def __init__(self, level, bank=None, *, map_identities=None):
        from swarmecho.env.buildings import target_spawn_boxes
        if map_identities is None:
            from swarmecho.training.map_selection import building_identity
            map_identities = [building_identity(level.building, level.building_name)]
        self.level = level
        self.map_identities = map_identities
        self.config = level.training.target_spawn_eligibility
        self.maximum = level.env.num_agents
        self.cell_size = (bank.envelope if bank is not None else level.building).cell_size_m
        if bank is None:
            lower, upper, volumes = target_spawn_boxes(level.building, level.env)
            self.lower, self.upper, self.volumes = lower[None], upper[None], volumes[None]
        else:
            self.lower, self.upper, self.volumes = (
                np.asarray(bank.arrays["target_" + suffix]) for suffix in ("lower", "upper", "volumes"))
        self.cells, self.box_cells, self.histograms = [], [], []
        for lower, upper, volumes in zip(self.lower, self.upper, self.volumes):
            valid = volumes > 0
            cells, inverse = np.unique(np.floor(((lower[valid] + upper[valid]) * .5) / self.cell_size).astype(int),
                                       axis=0, return_inverse=True)
            box_cells = np.zeros(len(volumes), dtype=int)
            box_cells[valid] = inverse
            self.cells.append(cells)
            self.box_cells.append(box_cells)
            self.histograms.append(np.zeros((len(cells), self.maximum + 1), dtype=np.int64))
        identity = {"format": self.format, "config": asdict(self.config), "maps": map_identities,
                    "env": asdict(level.env), "relay": asdict(level.training.dynamic_max_geo_distance),
                    "seed": level.training.bank_selection_seed, "attempts": level.training.spawn_pair_max_attempts}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
        for value in (self.lower, self.upper, self.volumes):
            digest.update(np.asarray(value, dtype="<f8").tobytes())
        # Invalidate cached measurements when geometry, certification or reset code changes.
        root = Path(__file__).resolve().parents[1]
        for name in ("env/environment.py", "env/buildings.py", "env/building_bank.py", "env/relay.py",
                     "env/relay_candidates.py", "training/target_spawn_eligibility.py"):
            digest.update((root / name).read_bytes().replace(b"\r\n", b"\n"))
        self.identity = digest.hexdigest()
        self.seed = level.training.bank_selection_seed
        self._tables = {}

    def _record(self, index, targets, required):
        coordinates = np.floor(np.asarray(targets) / self.cell_size).astype(int)
        lookup = {tuple(cell): i for i, cell in enumerate(self.cells[index])}
        try:
            indices = np.array([lookup[tuple(cell)] for cell in coordinates])
        except KeyError as error:
            raise ValueError("Eligibility probe landed outside the prepared target cells.") from error
        required = np.asarray(required)
        if required.shape != (len(indices),) or np.any((required < 1) | (required > self.maximum)):
            raise ValueError("Eligibility probes require certified target agent counts.")
        histogram = np.zeros_like(self.histograms[index])
        np.add.at(histogram, (indices, required.astype(int)), 1)
        return histogram

    def _load_map(self, index, row):
        histogram = np.asarray(row.get("histogram"), dtype=np.int64)
        if (row.get("identity") != self.identity or row.get("cells") != self.cells[index].tolist()
                or histogram.shape != self.histograms[index].shape or np.any(histogram < 0)
                or np.any(histogram[:, 0]) or histogram.sum() != self.config.samples_per_map):
            raise ValueError("Target eligibility cache does not match the prepared maps/settings.")
        self.histograms[index] = histogram

    def prepare(self, reset, directory, *, checkpoint=None, log=lambda message: None):
        import jax
        import jax.numpy as jnp
        frozen = Path(checkpoint) / "target_spawn_eligibility.json" if checkpoint else None
        if frozen is not None and frozen.is_file():
            self.restore(frozen)
            log("Restored frozen target-cell eligibility and required-agent histograms.")
            return
        directory = Path(directory) / self.identity
        block = 32
        count = self.config.samples_per_map
        padded = ((count + block - 1) // block) * block

        @jax.jit
        def probe(index, keys):
            def point(key):
                state = reset(key, map_id=index)
                return state.target_pos, state.target_required_agents, state.spawn_pair_attempts
            targets, required, attempts = jax.lax.map(jax.vmap(point), keys.reshape((-1, block, 2)))
            return targets.reshape(padded, 3)[:count], required.reshape(padded)[:count], attempts.reshape(padded)[:count]

        log(f"Preparing target-cell eligibility ({count} full-budget probes/map); shared cache {directory}.")
        # Completed per-map entries survive an interrupted build. Other sweep
        # workers wait and reuse them rather than repeating millions of probes.
        with cache_lock(directory):
            atomic_json(directory / "metadata.json", {
                "format": self.format, "identity": self.identity, "config": asdict(self.config),
                "maps": self.map_identities, "maximum_agents": self.maximum, "seed": self.seed})
            for index in range(len(self.cells)):
                path = directory / f"map_{index:05d}.json"
                if path.is_file():
                    row = json.loads(path.read_text(encoding="utf-8"))
                else:
                    root = jax.random.fold_in(jax.random.PRNGKey(self.seed + self.maximum * 1009 + 90001), index)
                    keys = jax.random.split(root, padded)
                    targets, required, attempts = jax.device_get(probe(jnp.int32(index), keys))
                    if np.any(attempts == 0):
                        raise ValueError(f"Map {index} has an uncertified eligibility probe; no unfiltered fallback.")
                    row = {"identity": self.identity, "cells": self.cells[index].tolist(),
                           "histogram": self._record(index, targets, required).tolist()}
                    atomic_json(path, row)
                self._load_map(index, row)
                if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(self.cells):
                    log(f"Target-cell eligibility: {index+1}/{len(self.cells)} maps prepared or loaded.")

    def tables(self, minimum=1, maximum=None, *, adaptive_weights=None, map_cdf=None):
        """Volume proposals within supported cells; never weight by probe hit count."""
        maximum = self.maximum if maximum is None else maximum
        key = int(minimum), int(maximum)
        if key not in self._tables:
            weights = np.zeros_like(self.volumes, dtype=np.float64)
            cutoff = math.ceil(self.config.samples_per_map * self.config.minimum_cell_fraction)
            for index, histogram in enumerate(self.histograms):
                eligible = histogram.sum(axis=1) >= cutoff
                if minimum > 1 or maximum < self.maximum:
                    eligible &= histogram[:, minimum:maximum+1].sum(axis=1) > 0
                weights[index] = self.volumes[index] * eligible[self.box_cells[index]]
            self._tables[key] = weights
        weights = self._tables[key].copy()
        if adaptive_weights is not None:
            # Adaptive probabilities are already volume-aware. Apply only the
            # frozen mask, preserving their success weights within that support.
            weights = np.where(weights > 0, np.asarray(adaptive_weights), 0.)
        supported = weights.sum(axis=1) > 0
        if not np.any(supported):
            raise ValueError(f"No probed target cells support requirements {minimum}..{maximum}. "
                             "Increase training.target_spawn_eligibility.samples_per_map; no easier fallback.")
        probabilities = np.ones(len(supported)) if map_cdf is None else np.diff(np.r_[0., np.asarray(map_cdf)])
        probabilities = np.where(supported, probabilities, 0.)
        if probabilities.sum() <= 0:
            raise ValueError("Map probabilities give no mass to the current target-cell support.")
        # An unsupported map should never be selected. A positive placeholder
        # row also keeps padded/masked JAX lanes away from all-minus-infinity logits.
        weights[~supported] = self.volumes[~supported]
        cdf = np.cumsum(probabilities / probabilities.sum())
        cdf[-1] = 1.
        return weights.astype(np.float32), cdf.astype(np.float32), supported

    def initialize_adaptive(self, controller):
        """Reuse the same measurements instead of running the old prepass twice."""
        children = getattr(controller, "controllers", (controller,))
        for index, child in enumerate(children):
            if child.eligibility_ready:
                continue
            counts = {tuple(cell): int(row.sum()) for cell, row in zip(self.cells[index], self.histograms[index])}
            child._apply_eligibility({"format": "swarmecho-adaptive-spawn-eligibility/v1",
                "sampler_identity": child.sampler_identity, "agent_budget": child.feasibility_budget,
                "cells": child.cells.tolist(), "sample_count": self.config.samples_per_map,
                "minimum_fraction": .01, "minimum_count": (self.config.samples_per_map + 99) // 100,
                "accepted_counts": [counts.get(tuple(cell), 0) for cell in child.cells]})

    def save(self, directory):
        rows = []
        for cells, histogram in zip(self.cells, self.histograms):
            present = histogram > 0
            minimum = np.where(present, np.arange(self.maximum+1), self.maximum+1).min(axis=1)
            maximum = np.where(present, np.arange(self.maximum+1), 0).max(axis=1)
            rows.append({"identity": self.identity, "cells": cells.tolist(), "histogram": histogram.tolist(),
                         "observed_min_agents": np.where(maximum > 0, minimum, 0).tolist(),
                         "observed_max_agents": maximum.tolist()})
        atomic_json(Path(directory) / "target_spawn_eligibility.json", {
            "format": self.format, "identity": self.identity, "config": asdict(self.config),
            "map_identities": self.map_identities, "maps": rows})

    def _matching_legacy_checkpoint(self, path, saved):
        """Migrate frozen measurements after a name-only source identity change."""
        if (saved.get("config") != asdict(self.config)
                or saved.get("map_identities") != self.map_identities):
            return False
        source = path.parent.parent.parent / "config.yaml"
        if not source.is_file():
            return False
        import yaml
        from swarmecho.core.config import load_level
        from swarmecho.core.compatibility import historical_environment_identity

        data = yaml.safe_load(source.read_text(encoding="utf-8"))
        data.pop("name", None)
        data.pop("building", None)
        previous = load_level(source, config_data=data)
        current = self.level
        return (historical_environment_identity(previous.env) == historical_environment_identity(current.env)
                and previous.training.bank_selection_seed == current.training.bank_selection_seed
                and previous.training.spawn_pair_max_attempts == current.training.spawn_pair_max_attempts
                and previous.training.dynamic_max_geo_distance == current.training.dynamic_max_geo_distance)

    def restore(self, path):
        path = Path(path)
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("format") != self.format or len(saved.get("maps", [])) != len(self.cells):
            raise ValueError("Frozen target-cell eligibility settings changed; start a new run or branch.")
        renamed = saved.get("identity") != self.identity
        if renamed and not self._matching_legacy_checkpoint(path, saved):
            raise ValueError("Frozen target-cell eligibility settings changed; start a new run or branch.")
        for index, row in enumerate(saved["maps"]):
            self._load_map(index, {**row, "identity": self.identity} if renamed else row)
        self._tables.clear()
