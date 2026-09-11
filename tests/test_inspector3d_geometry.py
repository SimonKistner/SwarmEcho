import yaml

from swarmecho.core import config
from swarmecho.visualize.inspector3d import building_geometry


def test_authored_geometry_classifies_recessed_outer_walls_and_roofs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MAP_DIR", tmp_path)
    data = {
        "format": "swarmecho-map/v1",
        "building_cell_grid": {"cols": 3, "rows": 1, "layers": 2},
        "cell_size_m": 5, "wall_thickness_m": .25, "tile_thickness_m": .5,
        "interior_cells": [[x, 0, z] for x in (0, 1) for z in (0, 1)],
        "geometry": {"x_walls": [[0, 0, 0], [1, 0, 0], [2, 0, 1]],
                     "y_walls": [[0, 0, 1]],
                     "tiles": [[0, 0, 0], [0, 0, 1], [0, 0, 2], [2, 0, 2]]},
    }
    (tmp_path / "shape.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    geometry = building_geometry({"map_name": "shape"})
    assert geometry["layers"] == 2
    assert [wall["outer"] for wall in geometry["walls"]] == [True, False, True, True]
    assert geometry["walls"][1]["lo"] == [4.875, 0, 0]
    assert geometry["walls"][1]["hi"] == [5.125, 5, 5]
    assert geometry["roofs"] == [{"lo": [0, 0, 9.75], "hi": [5, 5, 10.25]}]
    assert geometry["floors"] == [
        {"lo": [0, 0, -.25], "hi": [5, 5, .25], "storey": 0},
        {"lo": [0, 0, 4.75], "hi": [5, 5, 5.25], "storey": 1},
    ]
    assert building_geometry({"map_name": "missing"}) is None
