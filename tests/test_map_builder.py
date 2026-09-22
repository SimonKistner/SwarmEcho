from copy import deepcopy

import numpy as np
import pytest
import yaml

from swarmecho.curriculum_config.maps.scripts.maze_builder.building_builder_core import (
    add_outer_walls,
    add_roof,
    add_layer,
    document_from_map_data,
    document_yaml,
    delete_layer,
    map_data_from_document,
    new_document,
    expand_document,
    save_document,
    validate_document,
)
from swarmecho.curriculum_config.maps.scripts.validate_building import (
    validate_building_file,
)
from swarmecho.env.buildings import BuildingValidationError, compile_building


def _placed_document(cols=6, rows=4, layers=1):
    document = new_document(cols, rows, layers)
    document["base_cell"] = [0, 0, 0]
    return document


def test_new_builder_document_passes_training_map_contract():
    document = new_document(6, 4, 2)
    assert document["base_cell"] is None
    assert document["target_exclusion_cells"] == []
    document["base_cell"] = [0, 0, 0]
    report = validate_document(document)

    assert report["valid"]
    assert report["world_size_m"] == [30.0, 20.0, 10.0]
    assert report["interior_cells"] == 48
    assert report["target_candidate_cells"] == 48


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
    reopened = document_from_map_data(rebuilt)
    for collection in ("tiles", "x_walls", "y_walls"):
        assert reopened[collection] == document[collection]
    assert reopened["base_position_m"] == document["base_position_m"]
    assert validate_document(document)["valid"]


def test_export_maps_studio_north_to_positive_y_and_reopens_unchanged():
    doc = _placed_document(6, 4, 2)
    doc["target_exclusion_cells"] = [[2, 0, 1]]
    doc["tiles"] = [[2, 0, 0], [3, 3, 2]]
    doc["x_walls"] = [[2, 0, 1]]
    doc["y_walls"] = [[2, 0, 1], [3, 4, 0]]
    saved = map_data_from_document(doc)
    assert saved["geometry"]["x_walls"] == [[2, 3, 1]]
    assert {tuple(c) for c in saved["geometry"]["y_walls"]} == {(2, 4, 1), (3, 0, 0)}
    assert saved["target_exclusion_cells"] == [[2, 3, 1]]
    assert saved["base_position_m"][:2] == [2.5, 17.5]
    reopened = document_from_map_data(saved)
    assert reopened["base_cell"] == doc["base_cell"]
    for key in ("tiles", "x_walls", "y_walls", "target_exclusion_cells", "interior_cells"):
        assert {tuple(c) for c in reopened[key]} == {tuple(c) for c in doc[key]}
    assert map_data_from_document(reopened) == saved


def test_outer_walls_make_exact_eight_segments_for_two_by_two_footprint():
    document = _placed_document(4, 4, 1)
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
    document = _placed_document(2, 1, 2)
    document["tiles"] = [tile for tile in document["tiles"] if tile[2] == 0]
    document["interior_cells"].remove([1, 0, 1])

    roofed = add_roof(document)
    roof_tiles = {tuple(tile) for tile in roofed["tiles"] if tile[2] > 0}

    assert roof_tiles == {(0, 0, 2), (1, 0, 1)}


def test_roof_trims_open_facade_cutout_but_keeps_floorless_upper_volume():
    document = _placed_document(3, 2, 2)
    # Recess the facade around (1, 0), on both storeys. Its former roof exists.
    for z in range(2):
        document["y_walls"].remove([1, 0, z])
        document["y_walls"].append([1, 1, z])
        document["x_walls"].extend([[1, 0, z], [2, 0, z]])
    document["tiles"].remove([1, 0, 0])
    roofed = add_roof(document)

    assert [1, 0, 2] not in roofed["tiles"]
    assert [1, 0, 1] not in roofed["interior_cells"]
    assert [0, 0, 1] in roofed["interior_cells"]
    assert [0, 0, 1] not in roofed["tiles"]
    assert [0, 0, 2] in roofed["tiles"]
    assert len([c for c in roofed["tiles"] if c[2] == 2]) == 5
    assert validate_document(roofed)["valid"]
    assert [1, 0, 2] in document["tiles"]  # Input stays unchanged.


def test_validator_reports_an_exterior_leak_like_level_loading():
    document = _placed_document(2, 2, 1)
    document["x_walls"].remove([0, 0, 0])

    with pytest.raises(BuildingValidationError, match=r"leaks through -X"):
        validate_document(document)


def test_internal_walls_compile_to_runtime_aabbs():
    document = _placed_document(2, 1, 2)
    document["x_walls"].append([1, 0, 0])
    building = compile_building(map_data_from_document(document))

    assert building.solid_min_m.shape == (1, 3)
    np.testing.assert_allclose(building.solid_min_m[0], [4.875, 0.0, 0.0])
    np.testing.assert_allclose(building.solid_max_m[0], [5.125, 5.0, 5.0])


def test_runtime_saved_builder_map_can_run_cpu_training_contract_smoke(tmp_path):
    path = save_document(_placed_document(3, 2, 1), tmp_path / "created_map.yaml")
    report = validate_building_file(path, runtime_smoke=True)

    assert report["runtime_smoke"] is True
    assert report["observation_shape"] == [2, 38]
    assert report["target_candidate_cells"] == 6


def test_runtime_authored_wall_blocks_motion_and_visibility():
    import jax
    import jax.numpy as jnp

    from swarmecho.env.environment import EnvConfig, make_env_fns

    document = _placed_document(2, 1, 2)
    document["x_walls"].append([1, 0, 0])
    building = compile_building(map_data_from_document(document))
    cfg = EnvConfig(
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
    reset, step, _, _ = make_env_fns(building, cfg)
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
    document = _placed_document(2, 2, 1)
    document["interior_cells"].remove([0, 0, 0])
    document["target_exclusion_cells"] = [[0, 0, 0]]
    document["base_cell"] = [1, 1, 0]

    with pytest.raises(BuildingValidationError, match="non-interior"):
        validate_document(deepcopy(document))


def test_add_layer_copies_walls_and_target_exclusions_but_not_tiles():
    document = _placed_document(2, 2, 1)
    document["target_exclusion_cells"] = [[1, 1, 0]]
    expanded = add_layer(document)

    assert expanded["layers"] == 2
    assert [0, 0, 1] in expanded["x_walls"]
    assert [1, 1, 1] in expanded["target_exclusion_cells"]
    assert [1, 1, 1] in expanded["interior_cells"]
    assert not any(tile[2] == 1 for tile in expanded["tiles"])
    assert not any(tile[2] == 2 for tile in expanded["tiles"])


@pytest.mark.parametrize("direction", ["west", "east", "north", "south"])
def test_directional_expansion_preserves_and_duplicates_neighbor_state(direction):
    document = _placed_document(2, 2, 1)
    document["target_exclusion_cells"] = [[0, 0, 0]]
    expanded = expand_document(document, direction)

    assert expanded["cols"] == 2 + int(direction in {"west", "east"})
    assert expanded["rows"] == 2 + int(direction in {"north", "south"})
    assert len(expanded["interior_cells"]) == 6
    assert len(expanded["tiles"]) == 12
    assert validate_document(expanded)["valid"]


@pytest.mark.parametrize("direction", ["west", "east", "north", "south"])
def test_directional_shrink_removes_edge_and_preserves_valid_geometry(direction):
    document = _placed_document(3, 3, 1)
    shrunk = expand_document(document, direction, -1)

    assert shrunk["cols"] == 3 - int(direction in {"west", "east"})
    assert shrunk["rows"] == 3 - int(direction in {"north", "south"})
    if direction in {"west", "north"}:
        assert shrunk["base_cell"] is None
    assert len(shrunk["interior_cells"]) == 6


def test_shrink_rejects_dimension_below_one_cell():
    with pytest.raises(ValueError, match="below one cell"):
        expand_document(_placed_document(1, 2, 1), "west", -1)


def test_delete_storey_shifts_geometry_and_base_and_delete_roof_only_removes_roof():
    document = _placed_document(2, 2, 3)
    document["base_cell"] = [0, 0, 2]
    document["target_exclusion_cells"] = [[1, 1, 2]]
    deleted = delete_layer(document, 1)
    assert deleted["layers"] == 2
    assert deleted["base_cell"] == [0, 0, 1]
    assert [1, 1, 1] in deleted["target_exclusion_cells"]
    assert "base_position_m" not in deleted

    without_roof = delete_layer(deleted, deleted["layers"])
    assert without_roof["layers"] == deleted["layers"]
    assert not any(tile[2] == deleted["layers"] for tile in without_roof["tiles"])


def test_delete_only_storey_is_rejected_and_yaml_matches_map_payload():
    document = _placed_document(2, 2, 1)
    with pytest.raises(ValueError, match="at least one"):
        delete_layer(document, 0)
    assert yaml.safe_load(document_yaml(document)) == map_data_from_document(document)
