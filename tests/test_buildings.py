from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from swarmecho.env.buildings import (
    BuildingValidationError,
    compile_building,
    distance_comment,
    load_building,
)


BUILDING_PATH = Path("src/swarmecho/curriculum_config/buildings/B00_baseline_cuboid.yaml")


def _data():
    return yaml.safe_load(BUILDING_PATH.read_text(encoding="utf-8"))


def test_baseline_building_compiles_to_static_arrays():
    building = load_building(BUILDING_PATH)

    assert building.tiles.shape == (4, 4, 5)
    assert building.x_walls.shape == (5, 4, 4)
    assert building.y_walls.shape == (4, 5, 4)
    assert building.target_exclusion.shape == (4, 4, 4)
    np.testing.assert_allclose(building.base_position_m, [12.5, 12.5, 2.5])
    assert building.tile_thickness_m == 0.25
    assert building.wall_thickness_m == 0.25
    assert building.max_base_to_top_corner_m == pytest.approx(24.875, abs=0.001)
    assert distance_comment(building) == "# Maximum base-to-top-corner distance: 24.875 m"
    assert BUILDING_PATH.read_text(encoding="utf-8").splitlines()[0] == distance_comment(building)


@pytest.mark.parametrize(
    ("collection", "coordinate", "message"),
    [
        ("tiles", [0, 0, 0], "bottom floor and top roof"),
        ("x_walls", [0, 0, 0], "outer X walls"),
        ("y_walls", [0, 0, 0], "outer Y walls"),
    ],
)
def test_missing_shell_geometry_is_rejected(collection, coordinate, message):
    data = deepcopy(_data())
    data["geometry"][collection].remove(coordinate)

    with pytest.raises(BuildingValidationError, match=message):
        compile_building(data)


def test_intermediate_tile_holes_are_allowed():
    building = compile_building(_data())

    assert not building.tiles[:, :, 1:-1].any()


def test_out_of_bounds_target_exclusion_is_rejected():
    data = deepcopy(_data())
    data["target_exclusion_cells"].append([4, 0, 0])

    with pytest.raises(BuildingValidationError, match="outside valid bounds"):
        compile_building(data)
