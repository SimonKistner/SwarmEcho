"""Deterministic advanced exclusions and generation/cache integration."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from swarmecho.core.config import EnvConfig, RandomBuildingConfig, load_level_cli
from swarmecho.curriculum_config.maps.random_building_bank.generate_bank import (
    _compile_one, _map_digest, required_settings,
)
from swarmecho.env.buildings import load_building, target_spawn_boxes
from swarmecho.env.random_buildings import (
    advanced_target_exclusions, generate_building, generate_maps, random_building_grid,
)


G03_ENV = EnvConfig(comm_radius_base=15., visual_radius=5.)
SETTINGS = RandomBuildingConfig(enabled=True, length_m=15., width_m=15., stories=3,
                               rooms_per_story_min=2, rooms_per_story_max=4, staircase_max=2)


def representative_fixture(monkeypatch, *, wall=False, radius=20.):
    # All 12 cells have a legal representative. The wall blocks the diagonal
    # (0,0)->(1,1), while (0,1)->(1,1) is clear around its far end.
    cells = np.array([(x, y, 0) for x in range(4) for y in range(3)])
    centres = (cells + .5) * 5.
    lo, hi = centres - 1., centres + 1.
    monkeypatch.setattr("swarmecho.env.random_buildings.target_spawn_boxes",
                        lambda *args, **kwargs: (lo, hi, np.full(len(cells), 8.)))
    building = SimpleNamespace(base_position_m=np.array([2.5, 2.5, 2.5]), cell_size_m=5.,
        solid_min_m=np.array([[4.9, 0., 0.]]) if wall else np.empty((0, 3)),
        solid_max_m=np.array([[5.1, 5.1, 5.]]) if wall else np.empty((0, 3)))
    cfg = replace(G03_ENV, comm_radius_base=radius-1., visual_radius=1.)
    return advanced_target_exclusions(building, cfg)


def test_all_visible_cells_are_excluded_instead_of_only_three(monkeypatch):
    cells, details = representative_fixture(monkeypatch, radius=10.1)
    assert {tuple(cell) for cell in cells} == {
        (0, 0, 0), (0, 1, 0), (0, 2, 0), (1, 0, 0), (1, 1, 0), (2, 0, 0),
    }
    assert len(details["direct_los_cells"]) == 6
    assert details["around_corner_cell"] is None


def test_diagonal_wall_blocks_los_and_only_one_hidden_corner_is_added(monkeypatch):
    cells, details = representative_fixture(monkeypatch, wall=True, radius=15.1)
    direct = {tuple(cell) for cell in details["direct_los_cells"]}
    assert (1, 1, 0) not in direct  # diagonal clips the wall
    assert (0, 1, 0) in direct
    assert details["around_corner_cell"] == [1, 1, 0]
    assert details["around_corner_via"] == [0, 1, 0]
    assert len(cells) == len(direct) + 1
    assert [1, 0, 0] not in cells  # second hidden cell is not also excluded


def test_fully_excluded_layout_fails_instead_of_silent_fallback(monkeypatch):
    with pytest.raises(ValueError, match="excludes all legal target volume"):
        representative_fixture(monkeypatch)


@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_toggle_requires_a_boolean(value):
    with pytest.raises(ValueError, match="advanced_exclustion_base_LOS must be boolean"):
        random_building_grid(replace(SETTINGS, advanced_exclustion_base_LOS=value))


@pytest.mark.parametrize("seed", [0, 1, 2, 456789])
def test_generation_preserves_geometry_and_default_exclusions(seed):
    ordinary, base_building = generate_building(SETTINGS, seed, env_cfg=G03_ENV)
    enabled = replace(SETTINGS, advanced_exclustion_base_LOS=True)
    advanced, building = generate_building(enabled, seed, env_cfg=G03_ENV)
    repeated, _ = generate_building(enabled, seed, env_cfg=G03_ENV)
    assert base_building.target_exclusion.sum() == 1
    assert advanced == repeated
    assert advanced["geometry"] == ordinary["geometry"]
    assert advanced["base_position_m"] == ordinary["base_position_m"]
    np.testing.assert_array_equal(building.solid_min_m, base_building.solid_min_m)
    np.testing.assert_array_equal(building.solid_max_m, base_building.solid_max_m)
    assert building.target_exclusion.sum() > 1
    assert target_spawn_boxes(building, G03_ENV)[2].sum() > 0
    assert advanced["generation"]["target_exclusion_parameters"]["comm_radius_base"] == 15.


def test_serialized_generated_maps_preserve_exclusions_and_environment(tmp_path):
    enabled = replace(SETTINGS, advanced_exclustion_base_LOS=True)
    buildings, manifest = generate_maps(enabled, 456789, 2, directory=tmp_path, env_cfg=G03_ENV)
    for record, building in zip(manifest["maps"], buildings):
        path = tmp_path / f"{record['id']}.yaml"
        np.testing.assert_array_equal(load_building(path).target_exclusion, building.target_exclusion)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data["generation"]["target_exclusion_parameters"]["visual_radius"] == 5.
    assert manifest["target_exclusion_parameters"]["comm_radius_base"] == 15.


@pytest.mark.parametrize("level_name", ["G03_random_buildings", "G03_CLU_random_buildings"])
def test_cli_and_bank_identity_include_toggle_and_active_ranges(level_name):
    level = load_level_cli([f"level={level_name}", "random_buildings.advanced_exclustion_base_LOS=false"])
    assert not level.random_buildings.advanced_exclustion_base_LOS
    advanced = load_level_cli([f"level={level_name}"])
    assert advanced.random_buildings.advanced_exclustion_base_LOS
    assert advanced.training.map_pool_path is None
    assert advanced.building.target_exclusion.sum() > 1
    plain_settings, active_settings = required_settings(level), required_settings(advanced)
    assert plain_settings != active_settings
    assert active_settings["random_buildings"]["advanced_exclustion_base_LOS"] is True
    for field in ("comm_radius_base", "visual_radius"):
        changed_plain = replace(level, env=replace(level.env, **{field: 12.}))
        changed_active = replace(advanced, env=replace(advanced.env, **{field: 12.}))
        assert required_settings(changed_plain) == plain_settings
        assert required_settings(changed_active) != active_settings


def test_bank_worker_saves_only_nonexcluded_target_boxes(tmp_path):
    enabled = replace(SETTINGS, advanced_exclustion_base_LOS=True)
    cfg = replace(G03_ENV, comm_radius_base=9., visual_radius=3., target_wall_buffer_fraction=.15)
    data, _ = generate_building(enabled, 456789, name="map_00000", env_cfg=cfg)
    (tmp_path / "maps").mkdir()
    (tmp_path / "records").mkdir()
    _compile_one(0, 456789, _map_digest(data), enabled, cfg, False, 0., tmp_path)
    building = load_building(tmp_path / "maps/map_00000.yaml")
    with np.load(tmp_path / "records/map_00000.npz") as record:
        centres = (record["target_lower"] + record["target_upper"]) * .5
        cells = np.floor(centres / building.cell_size_m).astype(int)
        assert not building.target_exclusion[tuple(cells.T)].any()
        assert record["target_volumes"].sum() > 0


def test_generated_validation_reloads_matching_policy_and_rejects_changed_range(tmp_path):
    from swarmecho.training.validation import prepare_generated_validation_maps

    level = load_level_cli(["level=G03_random_buildings", "validation.num_maps=2",
                           "inspection_evaluation.inspection_maps=2",
                           "random_buildings.advanced_exclustion_base_LOS=true",
                           "env.comm_radius_base=9", "env.visual_radius=3"])
    _, training = generate_building(level.random_buildings, 456789, env_cfg=level.env)
    paths = prepare_generated_validation_maps(level, tmp_path, lambda message: None, [training])
    assert len(paths) == 2
    for path in paths:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data["generation"]["target_exclusion_parameters"]["comm_radius_base"] == 9.
        assert load_building(path).target_exclusion.sum() > 1
    assert prepare_generated_validation_maps(level, tmp_path, lambda message: None) == paths
    changed = replace(level, env=replace(level.env, comm_radius_base=10.))
    with pytest.raises(ValueError, match="validation pool settings changed"):
        prepare_generated_validation_maps(changed, tmp_path, lambda message: None)
