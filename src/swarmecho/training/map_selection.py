"""Frozen random-map datasets, independent of training sampling probabilities.

Keep the historical file locations readable so existing runs/checkpoints retain
their split when adaptive map weighting is disabled.
"""
from __future__ import annotations

import hashlib
import json
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
    if selection is None:
        raise ValueError('Frozen random map selection is missing.')
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
