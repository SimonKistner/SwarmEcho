"""Host-side adaptive map probabilities; fixed-shape arrays enter JAX resets."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from swarmecho.training.adaptive_spawn import atomic_json


def load_selection(destination, checkpoint):
    candidates = [Path(destination) / 'adaptive_maps' / 'selection.json']
    if checkpoint:
        candidates.append(Path(checkpoint) / 'adaptive_map_selection.json')
        marker = Path(checkpoint) / 'recovery_ready.json'
        if marker.is_file():
            source_run = json.loads(marker.read_text()).get('run_dir')
            if source_run:
                candidates.append(Path(source_run) / 'adaptive_maps' / 'selection.json')
        candidates.append(Path(checkpoint).parents[1] / 'adaptive_maps' / 'selection.json')
    for path in candidates:
        if path.is_file():
            return json.loads(path.read_text(encoding='utf-8'))
    return None


def save_selection(destination, payload):
    path = Path(destination) / 'adaptive_maps' / 'selection.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, payload)


def verify_selection_geometry(destination, identities, *, version=None, evaluation_identities=None):
    selection = load_selection(destination, None)
    if (selection.get('identities') is not None and selection['identities'] != identities
            or version is not None and selection.get('version', version) != version):
        raise ValueError('Frozen random map geometry or generator version changed.')
    selection['identities'] = identities
    if evaluation_identities is not None:
        if selection.get('evaluation_identities', evaluation_identities) != evaluation_identities:
            raise ValueError('Frozen held-out map geometry changed.')
        selection['evaluation_identities'] = evaluation_identities
    if version is not None:
        selection['version'] = version
    save_selection(destination, selection)


def building_identity(building, name):
    digest = hashlib.sha256()
    for field in ('tiles', 'x_walls', 'y_walls', 'target_exclusion', 'interior_cells',
                  'solid_min_m', 'solid_max_m', 'base_position_m', 'world_size_m'):
        value = np.asarray(getattr(building, field))
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.tobytes())
    return {'name': name, 'sha256': digest.hexdigest()}


def discover_cell_support(reset, controller, states, sample_count, seed, log):
    """Fixed per-map accepted sample budget, independent of training lane count."""
    import jax
    import jax.numpy as jnp
    positions, ids = jax.device_get((states.target_pos, states.map_id))
    order = np.argsort(ids, kind='stable')
    counts = np.bincount(ids, minlength=len(controller.controllers))
    offsets = np.r_[0, np.cumsum(counts)]
    block = 32
    padded = ((sample_count + block - 1) // block) * block

    @jax.jit
    def probe(index, keys):
        def point(key):
            state = reset(key, map_id=index)
            return state.target_pos, state.spawn_pair_attempts
        targets, attempts = jax.lax.map(jax.vmap(point), keys.reshape((-1, block) + keys.shape[1:]))
        return targets.reshape(padded, 3)[:sample_count], attempts.reshape(padded)[:sample_count]

    for index, child in enumerate(controller.controllers):
        available = positions[order[offsets[index]:offsets[index+1]]]
        if len(available) >= sample_count:
            targets = available[:sample_count]
        else:
            root = jax.random.fold_in(jax.random.PRNGKey(seed + 90001), index)
            keys = jax.random.split(root, padded)
            targets, attempts = jax.device_get(probe(jnp.int32(index), keys))
            if np.any(attempts == 0):
                raise ValueError(f'No certified startup target for {controller.map_names[index]} within the '
                                 'configured proposal attempts; inspect this map and stage budget.')
        child.initialize_eligibility(targets)
        if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(controller.controllers):
            log(f'Cell support discovery: {index+1}/{len(controller.controllers)} maps; {sample_count} targets per map.')


class PackedWindowView:
    """Compatibility view for the existing single-map cell controller."""
    def __init__(self, bank, index):
        self.bank, self.index = bank, index

    def __len__(self):
        return int(self.bank.counts[self.index])

    def __iter__(self):
        return iter(self.bank.values(self.index))

    @property
    def success_rate(self):
        return float(self.bank.sums[self.index] / max(1, len(self)))

    def append(self, value):
        b, i = self.bank, self.index
        slot = b.cursor[i]
        b.sums[i] += int(value) - int(b.buffer[i, slot])
        b.buffer[i, slot] = value
        b.cursor[i] = (slot + 1) % b.buffer.shape[1]
        b.counts[i] = min(b.buffer.shape[1], b.counts[i] + 1)
        b.total[i] += 1


class OutcomeWindows:
    """Packed latest-N outcome rings; repeated IDs retain chronological order."""
    def __init__(self, count, size):
        self.buffer = np.zeros((count, size), dtype=np.uint8)
        self.cursor = np.zeros(count, dtype=np.int64)
        self.counts = np.zeros(count, dtype=np.int64)
        self.sums = np.zeros(count, dtype=np.int64)
        self.total = np.zeros(count, dtype=np.int64)

    @property
    def rates(self):
        return np.divide(self.sums, self.counts, out=np.zeros(len(self.counts)), where=self.counts > 0)

    def observe(self, ids, successes):
        ids = np.asarray(ids).reshape(-1)
        outcomes = np.asarray(successes).reshape(-1)
        if (ids.dtype.kind not in 'iu' or len(ids) != len(outcomes)
                or np.any(ids < 0) or np.any(ids >= len(self.counts))
                or not np.all(np.isin(outcomes, (0, 1)))):
            raise ValueError('Invalid map/cell IDs or success outcomes.')
        order = np.argsort(ids, kind='stable')
        ordered = ids[order]
        keys, starts, lengths = np.unique(ordered, return_index=True, return_counts=True)
        width = self.buffer.shape[1]
        for key, start, length in zip(keys, starts, lengths):
            values = outcomes[order[start:start+length]].astype(np.uint8)
            if length >= width:
                self.buffer[key] = values[-width:]
                self.cursor[key] = 0
                self.sums[key] = self.buffer[key].sum()
            else:
                slots = (self.cursor[key] + np.arange(length)) % width
                self.sums[key] += int(values.sum()) - int(self.buffer[key, slots].sum())
                self.buffer[key, slots] = values
                self.cursor[key] = (self.cursor[key] + length) % width
            self.counts[key] = min(width, self.counts[key] + length)
            self.total[key] += length

    def values(self, index):
        if self.counts[index] < self.buffer.shape[1]:
            return self.buffer[index, :self.counts[index]].tolist()
        return np.roll(self.buffer[index], -self.cursor[index]).tolist()

    def arrays(self):
        return {name: getattr(self, name) for name in ('buffer', 'cursor', 'counts', 'sums', 'total')}

    def load(self, data):
        for name, original in self.arrays().items():
            value = np.asarray(data[name])
            if value.shape != original.shape or value.dtype.kind not in 'iu':
                raise ValueError(f'Incompatible adaptive map window {name}.')
            setattr(self, name, value.astype(original.dtype))
        width = self.buffer.shape[1]
        if (np.any(self.buffer > 1) or np.any(self.cursor < 0) or np.any(self.cursor >= width)
                or np.any(self.counts < 0) or np.any(self.counts > width)
                or np.any(self.total < self.counts)
                or not np.array_equal(self.sums, self.buffer.sum(axis=1))):
            raise ValueError('Invalid adaptive map outcome rings.')


def atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


class AdaptiveMapSampler:
    def __init__(self, identities, config, *, stage_identity):
        from swarmecho.core.config import validate_adaptive_map_sampling
        validate_adaptive_map_sampling(config)
        if not config.enabled or len(identities) < 1:
            raise ValueError('Create a map sampler only for an enabled, nonempty map bank.')
        self.config = config
        self.identities = identities
        self.identity = hashlib.sha256(json.dumps(
            [identities, stage_identity], sort_keys=True).encode()).hexdigest()
        self.windows = OutcomeWindows(len(identities), config.success_window_episodes_per_map)
        self.probabilities = np.full(len(identities), 1. / len(identities))
        self.selection_probabilities = self.probabilities.copy()
        self.step_counts = np.zeros(len(identities), dtype=np.int64)
        self.episode_starts = np.zeros(len(identities), dtype=np.int64)
        self.update = self.steps = 0

    @property
    def ready(self):
        return bool(np.all(self.windows.counts == self.config.success_window_episodes_per_map))

    @property
    def cdf(self):
        result = np.cumsum(self.selection_probabilities).astype(np.float32)
        result[-1] = 1.
        return result

    def observe(self, map_ids, successes, *, step_map_ids, started_map_ids):
        self.windows.observe(map_ids, successes)
        for ids, total in ((step_map_ids, self.step_counts), (started_map_ids, self.episode_starts)):
            ids = np.asarray(ids).reshape(-1)
            if ids.dtype.kind not in 'iu' or np.any(ids < 0) or np.any(ids >= len(total)):
                raise ValueError('Invalid map exposure IDs.')
            total += np.bincount(ids, minlength=len(total))

    def advance(self, update, steps, active_map_ids):
        applied = self.selection_probabilities.copy()
        active = np.bincount(np.asarray(active_map_ids), minlength=len(self.identities))
        if self.ready:
            failure = 1. - self.windows.rates
            worst = failure.max()
            baseline = np.full(len(failure), 1. / len(failure))
            desired = baseline if worst == 0 else (
                self.config.min_probability_fraction * baseline
                + (1. - self.config.min_probability_fraction) * (
                    (1. - worst) * baseline + worst * failure / failure.sum()))
            blend = -np.expm1(-np.log(2.) / self.config.probability_smoothing_half_life_updates)
            self.probabilities += blend * (desired - self.probabilities)
            self.probabilities /= self.probabilities.sum()
            self.selection_probabilities = self.probabilities.copy()
        else:
            # Prefer unfinished warmup buckets, allowing for episodes in flight.
            remaining = np.maximum(self.config.success_window_episodes_per_map - self.windows.counts, 0)
            demand = np.maximum(remaining - active, 0)
            if not np.any(demand):
                demand = remaining
            self.selection_probabilities = demand / demand.sum()
        self.update, self.steps = int(update), int(steps)
        metrics = self.metrics()
        row = {'kind': 'update', 'update': self.update, 'steps': self.steps, **metrics}
        if len(self.identities) <= 32:
            row.update(probabilities=applied.tolist(), next_probabilities=self.selection_probabilities.tolist(),
                       success_rates=self.windows.rates.tolist(), completed_counts=self.windows.total.tolist(),
                       step_counts=self.step_counts.tolist(), episode_starts=self.episode_starts.tolist())
        else:
            indices = np.argsort(-self.selection_probabilities)[:10]
            row['most_sampled_maps'] = [{'index': int(i), 'probability': float(self.selection_probabilities[i]),
                                        'success': float(self.windows.rates[i])} for i in indices]
        return row, metrics

    def metrics(self):
        observed = self.windows.counts > 0
        rates = self.windows.rates[observed]
        result = {'adaptive_maps/ready': float(self.ready),
                  'adaptive_maps/warmup_fraction': float(np.mean(
                      self.windows.counts == self.config.success_window_episodes_per_map)),
                  'adaptive_maps/observed_fraction': float(np.mean(observed)),
                  'adaptive_maps/min_probability': float(self.selection_probabilities.min()),
                  'adaptive_maps/max_probability': float(self.selection_probabilities.max())}
        if len(rates):
            result.update({'train/map_balanced_success_rate': float(rates.mean()),
                           'train/worst_map_success_rate': float(rates.min())})
        if len(self.identities) <= 32:
            for i, identity in enumerate(self.identities):
                prefix = f"adaptive_maps/{identity['name']}"
                result.update({f'{prefix}/probability': float(self.selection_probabilities[i]),
                               f'{prefix}/success_rate': float(self.windows.rates[i]),
                               f'{prefix}/step_share': float(self.step_counts[i] / max(1, self.step_counts.sum())),
                               f'{prefix}/episode_start_share': float(
                                   self.episode_starts[i] / max(1, self.episode_starts.sum()))})
        return result

    def save(self, checkpoint_path):
        header = {'format': 'swarmecho-adaptive-maps/v1', 'identity': self.identity,
                  'config': asdict(self.config), 'update': self.update, 'steps': self.steps}
        atomic_npz(Path(checkpoint_path) / 'adaptive_maps_state.npz',
                   header=np.frombuffer(json.dumps(header).encode(), dtype=np.uint8),
                   probabilities=self.probabilities, selection_probabilities=self.selection_probabilities,
                   step_counts=self.step_counts, episode_starts=self.episode_starts, **self.windows.arrays())

    def restore(self, checkpoint_path):
        path = Path(checkpoint_path) / 'adaptive_maps_state.npz'
        if not path.is_file():
            return False
        with np.load(path, allow_pickle=False) as data:
            header = json.loads(data['header'].tobytes())
            saved_config = dict(header.get('config', {}))
            legacy_warmup = 'warmup_episodes_per_map' in saved_config
            saved_config.pop('warmup_episodes_per_map', None)
            if (header.get('format') != 'swarmecho-adaptive-maps/v1' or header.get('identity') != self.identity
                    or saved_config != asdict(self.config)):
                raise ValueError('Adaptive map geometry/order, stage or configuration changed; branch into a new run.')
            self.windows.load(data)
            for name in ('probabilities', 'selection_probabilities', 'step_counts', 'episode_starts'):
                value = data[name]
                if value.shape != self.probabilities.shape or not np.all(np.isfinite(value)) or np.any(value < 0):
                    raise ValueError(f'Invalid adaptive map checkpoint {name}.')
                setattr(self, name, value.copy())
        if (not np.isclose(self.probabilities.sum(), 1.) or not np.isclose(self.selection_probabilities.sum(), 1.)
                or np.any(self.probabilities < self.config.min_probability_fraction / len(self.identities) - 1e-12)):
            raise ValueError('Invalid adaptive map checkpoint probability distribution.')
        self.update, self.steps = int(header['update']), int(header['steps'])
        if legacy_warmup and not self.ready:
            # A legacy checkpoint may have adapted after a partial window.
            # Continue filling its saved outcomes before using adaptive weights.
            remaining = self.config.success_window_episodes_per_map - self.windows.counts
            self.selection_probabilities = remaining / remaining.sum()
        return True


class AdaptiveMapJournal:
    def __init__(self, destination, sampler, *, resume=False):
        self.directory = Path(destination) / 'adaptive_maps'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'history.jsonl'
        metadata = {'format': 'swarmecho-adaptive-maps/v1', 'identity': sampler.identity,
                    'maps': sampler.identities, 'config': asdict(sampler.config)}
        previous = self.directory / 'metadata.json'
        if self.path.is_file() and self.path.stat().st_size:
            if not resume or not previous.is_file() or json.loads(previous.read_text()) != metadata:
                raise ValueError('Adaptive map history exists or changed; resume explicitly or choose a new run name.')
            with self.path.open('rb+') as stream:
                stream.seek(-1, os.SEEK_END)
                if stream.read(1) != b'\n':
                    stream.seek(0, os.SEEK_END)
                    stream.write(b'\n')
        atomic_json(previous, metadata)
        self.append({'kind': 'session', 'update': sampler.update, 'steps': sampler.steps})

    def append(self, row):
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
