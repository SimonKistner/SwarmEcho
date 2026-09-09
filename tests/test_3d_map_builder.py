from copy import deepcopy

import numpy as np
import pytest
import yaml

from swarmecho.curriculum_config.maps.scripts.maze_builder.building_builder_core import (
    add_outer_walls,
    add_roof,
    document_from_map_data,
    map_data_from_document,
    new_document,
    save_document,
    validate_document,
)
from swarmecho.curriculum_config.maps.scripts.validate_building import (
    validate_building_file,
)
from swarmecho.env.buildings import BuildingValidationError, compile_building


def test_new_builder_document_passes_training_map_contract():
    document = new_document(6, 4, 2)
    report = validate_document(document)

    assert report["valid"]
    assert report["world_size_m"] == [30.0, 20.0, 10.0]
    assert report["interior_cells"] == 48
    assert report["target_candidate_cells"] == 47


def test_existing_v1_map_opens_without_migration():
    source = yaml.safe_load(
        open(
            "src/swarmecho/curriculum_config/maps/M00_no_maze_open_cuboid.yaml",
            encoding="utf-8",
        )
    )
    document = document_from_map_data(source)
    rebuilt = map_data_from_document(document)

    assert rebuilt["format"] == "swarmecho-map/v1"
    for collection in ("tiles", "x_walls", "y_walls"):
        assert {tuple(value) for value in rebuilt["geometry"][collection]} == {
            tuple(value) for value in source["geometry"][collection]
        }
    assert rebuilt["base_position_m"] == source["base_position_m"]
    assert validate_document(document)["valid"]


def test_outer_walls_make_exact_eight_segments_for_two_by_two_footprint():
    document = new_document(4, 4, 1)
    document["interior_cells"] = [
        [x, y, 0] for x in (1, 2) for y in (1, 2)
    ]
    document["tiles"] = [[x, y, 0] for x in (1, 2) for y in (1, 2)]
    document["x_walls"] = []
    document["y_walls"] = []
    document["base_cell"] = [1, 1, 0]
    document["target_exclusion_cells"] = [[1, 1, 0]]

    document = add_outer_walls(document, 0)
    document = add_roof(document)

    assert len(document["x_walls"]) + len(document["y_walls"]) == 8
    assert len([tile for tile in document["tiles"] if tile[2] == 1]) == 4
    assert validate_document(document)["valid"]


def test_add_roof_only_closes_exposed_top_faces():
    document = new_document(2, 1, 2)
    document["tiles"] = [tile for tile in document["tiles"] if tile[2] == 0]
    document["interior_cells"].remove([1, 0, 1])

    roofed = add_roof(document)
    roof_tiles = {tuple(tile) for tile in roofed["tiles"] if tile[2] > 0}

    assert roof_tiles == {(0, 0, 2), (1, 0, 1)}


def test_validator_reports_an_exterior_leak_like_level_loading():
    document = new_document(2, 2, 1)
    document["x_walls"].remove([0, 0, 0])

    with pytest.raises(BuildingValidationError, match=r"leaks through -X"):
        validate_document(document)


def test_internal_walls_compile_to_runtime_aabbs():
    document = new_document(2, 1, 2)
    document["x_walls"].append([1, 0, 0])
    building = compile_building(map_data_from_document(document))

    assert building.solid_min_m.shape == (1, 3)
    np.testing.assert_allclose(building.solid_min_m[0], [4.875, 0.0, 0.0])
    np.testing.assert_allclose(building.solid_max_m[0], [5.125, 5.0, 5.0])


def test_runtime_saved_builder_map_can_run_cpu_training_contract_smoke(tmp_path):
    path = save_document(new_document(3, 2, 1), tmp_path / "created_map.yaml")
    report = validate_building_file(path, runtime_smoke=True)

    assert report["runtime_smoke"] is True
    assert report["observation_shape"] == [2, 38]
    assert report["target_candidate_cells"] == 5


def test_runtime_authored_wall_blocks_motion_and_visibility():
    import jax
    import jax.numpy as jnp

    from swarmecho.env.baseline3d import Baseline3DConfig, make_baseline_3d_fns

    document = new_document(2, 1, 2)
    document["x_walls"].append([1, 0, 0])
    building = compile_building(map_data_from_document(document))
    cfg = Baseline3DConfig(
        num_agents=1,
        dt=1.0,
        drag=0.0,
        max_force=2.0,
        max_speed=2.0,
        visual_radius=10.0,
        comm_radius_base=0.1,
        comm_radius=0.1,
        target_spawn_buffer=0.0,
        coverage_voxel_size=5.0,
    )
    reset, step, _, _ = make_baseline_3d_fns(building, cfg)
    state = reset(jax.random.PRNGKey(0), target_pos=jnp.asarray([7.5, 2.5, 2.5]))
    state = state._replace(
        pos=jnp.asarray([[4.0, 2.5, 2.5]]),
        vel=jnp.zeros((1, 3)),
        active=jnp.ones(1, dtype=jnp.bool_),
    )

    next_state = step(state, jnp.asarray([[1.0, 0.0, 0.0]]))

    np.testing.assert_allclose(next_state.pos[0], state.pos[0])
    assert bool(next_state.collided[0])
    assert not bool(next_state.directly_sees_target[0])


def test_non_interior_target_exclusion_is_rejected():
    document = new_document(2, 2, 1)
    document["interior_cells"].remove([0, 0, 0])
    document["target_exclusion_cells"] = [[0, 0, 0]]
    document["base_cell"] = [1, 1, 0]

    with pytest.raises(BuildingValidationError, match="non-interior"):
        validate_document(deepcopy(document))
