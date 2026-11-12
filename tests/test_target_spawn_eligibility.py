"""Frozen support must preserve volume proposals and exact group selection."""
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from swarmecho.core.config import load_level
from swarmecho.training.target_spawn_eligibility import TargetSpawnEligibility


def test_atomic_json_concurrent_writers_use_independent_temporary_files(tmp_path, monkeypatch):
    from swarmecho.training import adaptive_spawn
    original_replace = adaptive_spawn.os.replace
    ready = threading.Barrier(2)
    sources = []

    def simultaneous_replace(source, destination):
        sources.append(source)
        # Force both writers to finish their temporary files before either
        # rename. The former shared '.tmp' name deterministically fails here.
        ready.wait(timeout=10)
        original_replace(source, destination)

    monkeypatch.setattr(adaptive_spawn.os, "replace", simultaneous_replace)
    path = tmp_path / "shared_map.json"
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda i: adaptive_spawn.atomic_json(path, {"writer": i, "data": [i]*100}), (0, 1)))
    assert len(set(sources)) == 2
    saved = json.loads(path.read_text())
    assert saved["data"] == [saved["writer"]] * 100
    assert not list(tmp_path.glob("*.tmp"))


def support():
    level = load_level("G03_CLU_random_buildings")
    level = replace(level, training=replace(level.training, target_spawn_eligibility=replace(
        level.training.target_spawn_eligibility, samples_per_map=100)))
    lower = np.array([[[0., 0., 0.], [5., 0., 0.], [10., 0., 0.]]] * 2)
    bank = SimpleNamespace(envelope=SimpleNamespace(cell_size_m=5.), arrays={
        "target_lower": lower, "target_upper": lower + 1.,
        "target_volumes": np.array([[1., 2., 3.], [1., 2., 3.]])})
    sampler = TargetSpawnEligibility(level, bank, map_identities=["first", "second"])
    # The first map's mixed cell has counts {2, 7}; its min/max interval
    # includes 6 but no six-agent target was actually observed there.
    sampler.histograms[0][0, 2] = 10
    sampler.histograms[0][1, 2] = 40
    sampler.histograms[0][1, 7] = 50
    sampler.histograms[1][0, 3] = 100
    return sampler


def test_requirement_histogram_filters_cells_and_maps_without_hit_count_weighting():
    sampler = support()
    weights, cdf, supported = sampler.tables(7, 7)
    np.testing.assert_array_equal(supported, [True, False])
    np.testing.assert_array_equal(cdf, [1., 1.])
    np.testing.assert_array_equal(weights[0], [0., 2., 0.])
    weights, _, _ = sampler.tables(1, 7)
    # Different probe frequencies do not replace the cell-volume ratio 1:2.
    np.testing.assert_array_equal(weights[0], [1., 2., 0.])
    with pytest.raises(ValueError, match="No probed target cells"):
        sampler.tables(6, 6)


def test_adaptive_weights_are_only_masked_and_map_cdf_retains_relative_mass():
    sampler = support()
    weights, cdf, supported = sampler.tables(1, 7,
        adaptive_weights=np.array([[.7, .2, .1], [.3, .3, .4]]), map_cdf=np.array([.25, 1.]))
    np.testing.assert_allclose(weights[0], [.7, .2, 0.])
    np.testing.assert_allclose(weights[1], [.3, 0., 0.])
    np.testing.assert_allclose(cdf, [.25, 1.])
    assert supported.all()


def test_frozen_histograms_round_trip_and_reject_changed_identity(tmp_path):
    sampler = support()
    sampler.save(tmp_path)
    restored = support()
    restored.histograms = [np.zeros_like(h) for h in restored.histograms]
    restored.restore(tmp_path / "target_spawn_eligibility.json")
    for original, actual in zip(sampler.histograms, restored.histograms):
        np.testing.assert_array_equal(original, actual)
    restored.identity = "different-map-settings"
    with pytest.raises(ValueError, match="settings changed"):
        restored.restore(tmp_path / "target_spawn_eligibility.json")
