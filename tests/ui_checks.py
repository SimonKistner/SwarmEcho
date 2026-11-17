"""Opt-in checks for the inspector and building editor."""

from copy import deepcopy
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
import json
import os
import threading
import urllib.request

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from swarmecho.core import config
from swarmecho.core.config import EnvConfig, MAP_DIR
from swarmecho.curriculum_config.maps.scripts.maze_builder.building_builder_core import (
    add_layer,
    add_outer_walls,
    add_roof,
    delete_layer,
    document_from_map_data,
    document_yaml,
    expand_document,
    map_data_from_document,
    new_document,
    save_document,
    validate_document,
)
from swarmecho.curriculum_config.maps.scripts.validate_building import validate_building_file
from swarmecho.env.buildings import BuildingValidationError, compile_building, load_building
from swarmecho.env.environment import make_env_fns
from swarmecho.training.artifacts import save_eval_info_csv
from swarmecho.visualize import inspector
from swarmecho.visualize.inspector import (
    building_geometry,
    discover_heatmaps,
    discover_replays,
    discover_roadmap_tests,
    heatmap_label,
    HTML,
    inspector_html,
    replay_label,
    replay_payload,
)
from swarmecho.visualize.replay import building_snapshot, write_replay


# Inspector artifacts and controls

def test_snapshot_geometry_survives_missing_original_map():
    building = load_building(MAP_DIR / "mirrored_office.yaml")
    snapshot = building_snapshot(building)
    geometry = building_geometry({"map_name": "does_not_exist", "building_snapshot": snapshot})
    assert geometry["source"] == "replay snapshot"
    assert len(geometry["walls"]) > 0
    assert geometry["solid_min"] == building.solid_min_m.tolist()
    assert geometry["solid_max"] == building.solid_max_m.tolist()


def test_legacy_office_visibility_bounds_match_runtime():
    building = load_building(MAP_DIR / "mirrored_office.yaml")
    geometry = building_geometry({"map_name": "mirrored_office"})
    actual = sorted(tuple(lo + hi) for lo, hi in zip(geometry["solid_min"], geometry["solid_max"]))
    expected = sorted(tuple(lo.tolist() + hi.tolist()) for lo, hi in zip(building.solid_min_m, building.solid_max_m))
    assert actual == expected


def test_inspector_payload_and_controls(tmp_path):
    level = SimpleNamespace(
        building=load_building(MAP_DIR / "open_cuboid.yaml"),
        building_name="open_cuboid",
        env=EnvConfig(num_agents=4, coverage_voxel_size=2.5),
    )
    reset, step, _, _ = make_env_fns(level.building, level.env)
    state = reset(jax.random.PRNGKey(0))
    states = [state, step(state, jnp.zeros((level.env.num_agents, 3)))]
    _, manifest = write_replay(
        tmp_path / "test_run" / "replays" / "inspect",
        states,
        map_name=level.building_name,
        dt=level.env.dt,
        metadata={
            "world_size_m": level.building.world_size_m.tolist(),
            "cell_size_m": level.building.cell_size_m,
            "coverage_voxel_size_m": level.env.coverage_voxel_size,
            "comm_radius_m": level.env.comm_radius,
        },
    )
    payload = replay_payload(manifest)
    assert payload["position"][0][0][2] == (
        level.building.base_position_m[2] + level.env.drone_radius
    )
    assert payload["manifest"]["world_size_m"] == [20.0, 20.0, 20.0]
    assert payload["manifest"]["coverage_voxel_size_m"] == 2.5
    assert 'id="timeline"' in HTML
    assert 'id="replaySelect"' in HTML
    assert 'id="artifactPickerButton"' in HTML
    assert 'id="artifactMenu"' in HTML
    assert 'id="artifactSubmenu"' in HTML
    assert "function artifactParts(label)" in HTML
    assert "function openArtifactSubmenu(group,anchor)" in HTML
    assert "group.length>1?' group':''" in HTML
    assert "count.textContent=group.length" in HTML
    assert 'id="showCoverage"' in HTML
    assert 'id="showVisualRange"' in HTML
    assert 'id="showCommRange"' in HTML
    assert 'id="showVisualRange" type="checkbox" checked' not in HTML
    assert 'id="showCommRange" type="checkbox" checked' not in HTML
    assert 'id="refreshReplays"' in HTML
    assert 'id="refreshCurrent"' in HTML
    assert "function refreshCurrent()" in HTML
    assert "cache:'no-store'" in HTML
    assert 'id="autoRotate"' in HTML
    assert 'id="rotateSpeed"' in HTML
    assert 'id="cameraElevation"' in HTML
    assert 'id="cameraElevationValue"' in HTML
    assert 'id="cameraElevation" type="range" min="-85" max="85" step="1" value="25"' in HTML
    assert 'id="autoRotate" type="checkbox" checked' in HTML
    assert "Math.tan(25*Math.PI/180)" in HTML
    assert 'id="cameraControls" class="card"' in HTML
    assert 'id="wallMaterial"' not in HTML
    assert 'id="wallTransparency" type="range" min="0" max="100" step="1" value="50"' in HTML
    assert 'id="loopReplay"' in HTML
    assert 'id="fixedBounds"' not in HTML
    assert "function rotateCamera(timestamp)" in HTML
    assert "rangeRadius=mode==='replay'" not in HTML
    assert "autorange:false" not in HTML
    assert "aspectmode:'data'" in HTML
    assert "function rememberCamera()" in HTML
    assert "scene.__heatmapClickListener=true}updateAutoRotate()" in HTML
    assert "function renderPlot(" in HTML
    assert "if(rendering)" in HTML
    assert "draw(frame>=last?0:frame+1).finally" in HTML
    assert "frame>=last&&!$('loopReplay').checked" in HTML
    assert 'id="heatmapConfidence"' in HTML
    assert 'id="heatmapConfidenceValue">100%' in HTML
    assert "function heatmapStage(index)" in HTML
    assert 'id="coverageOpacity"' in HTML
    assert 'id="visualOpacity"' in HTML
    assert 'id="commOpacity"' in HTML
    assert 'value="0.02"' in HTML
    assert 'value="0.045"' in HTML
    assert 'value="0.025"' in HTML
    assert "type:'mesh3d'" in HTML
    assert "coverage_voxel_size_m||D.manifest.cell_size_m" in HTML
    assert "uirevision:'replay-camera'" in HTML
    assert "Plotly.react" in HTML
    assert '<script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>' not in inspector_html()
    assert discover_replays(tmp_path) == [manifest.resolve()]
    assert replay_label(manifest) == "test_run_[unknown]_[Replay]"


def test_replay_label_uses_run_name_before_artifacts(tmp_path):
    manifest = tmp_path / "M00_baseline" / "artifacts" / "train" / "replays" / "eval.json"
    assert replay_label(manifest) == "M00_baseline_[unknown]_[Replay]"


def test_replay_label_adds_unpadded_training_steps(tmp_path):
    manifest = tmp_path / "M00_baseline_v2" / "artifacts" / "train" / "replays" / "eval_u000061_s00024M.json"
    assert replay_label(manifest) == "M00_baseline_v2_[24M]_[Replay]"


def test_heatmap_labels_distinguish_training_and_standalone_evaluation(tmp_path):
    run = tmp_path / "M00_baseline"
    train_heatmap = (
        run / "artifacts/train/data/eval_info_u000061_s00024M.csv"
    )
    eval_heatmap = (
        run
        / "artifacts/eval/ckpt_u000351_s00140M/data/"
        "eval_info_u000351_s00140M.csv"
    )

    assert heatmap_label(train_heatmap) == (
        "M00_baseline_[24M]_[TRAIN Heatmap]"
    )
    assert heatmap_label(eval_heatmap) == (
        "M00_baseline_[140M]_[EVAL Heatmap]"
    )


def test_inspector_discovers_new_eval_runs_and_hides_timestamps_in_labels(tmp_path):
    run = tmp_path / "M00_baseline"
    eval_run = (
        run
        / "artifacts/eval/ckpt_u000351_s00140M/"
        "eval_agents7_20260815T120000_123456789Z"
    )
    data = eval_run / "data"
    info_path = save_eval_info_csv(
        data / "eval_info_u000351_s00140M.csv",
        target_positions=[[1.0, 2.0, 3.0]],
        base_positions=[[0.0, 0.0, 0.0]],
        successes=[True],
        delivered=[True],
        visually_found=[True],
    )
    info_path.with_suffix(".heatmap.json").write_text(
        json.dumps({"artifact_scope": "eval", "eval_name": "agents7"})
    )
    replay_data, replay_manifest = write_replay(
        eval_run / "replays/eval_u000351_s00140M_SUCCESS_0",
        [
            type("State", (), {
                "pos": [[[0.0, 0.0, 0.0]]],
                "vel": [[[0.0, 0.0, 0.0]]],
                "active": [[True]],
                "target_pos": [[1.0, 2.0, 3.0]],
                "base_pos": [[0.0, 0.0, 0.0]],
                "directly_sees_target": [[False]],
                "is_conn_base": [[False]],
                "is_conn_target": [[False]],
                "target_known": [[False]],
                "success": [False],
                "fully_connected": [False],
                "chain_held_steps": [0],
                "done": [False],
                "collided": [[False]],
                "coverage_credit": [[0.0]],
                "coverage": [[[False]]],
                "step": 0,
                "obstacle_min": [],
                "obstacle_max": [],
            })()
        ],
        map_name="test",
        dt=0.1,
        metadata={"eval_name": "agents7"},
        progress=False,
    )

    assert discover_heatmaps(tmp_path) == [info_path.resolve()]
    assert discover_replays(tmp_path) == [replay_manifest.resolve()]
    assert heatmap_label(info_path) == (
        "M00_baseline_[140M]_[EVAL Heatmap]_[agents7]"
    )
    assert replay_label(replay_manifest) == (
        "M00_baseline_[140M]_[Replay]_[agents7]"
    )
    assert "20260815" not in heatmap_label(info_path)
    assert "20260815" not in replay_label(replay_manifest)


def test_inspector_discovers_dedicated_roadmap_testresult(tmp_path):
    result = tmp_path / "testresults" / "obstacles.roadmap.json"
    result.parent.mkdir()
    result.write_text(json.dumps({"format": "swarmecho-roadmap-test/v1"}))
    assert discover_roadmap_tests(tmp_path) == [result.resolve()]
    assert "function drawRoadmap()" in HTML
    assert "roadmapPaths.forEach" in HTML
    assert 'id="roadmapRoutes"' in HTML
    assert "function cuboidMesh" in HTML
    assert "opacity:.5" in HTML


# Inspector geometry

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
    assert geometry["roofs"] == [{"lo": [0, 0, 9.75], "hi": [5, 5, 10.25], "storey": 1}]
    assert geometry["floors"] == [
        {"lo": [0, 0, -.25], "hi": [5, 5, .25], "storey": 0},
        {"lo": [0, 0, 4.75], "hi": [5, 5, 5.25], "storey": 1},
    ]
    assert building_geometry({"map_name": "missing"}) is None


# Recent inspector results

def test_recent_listing_returns_three_runs_then_full_inventory(tmp_path, monkeypatch):
    replays = []
    for index in range(4):
        path = tmp_path / f"run_{index}" / "artifacts" / "train" / "replays" / "eval.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}", encoding="utf-8")
        os.utime(path, ns=((index + 1) * 1_000_000_000,) * 2)
        replays.append(path.resolve())
    roadmap = tmp_path / "testresults" / "route.roadmap.json"
    roadmap.parent.mkdir()
    roadmap.write_text("{}", encoding="utf-8")

    def discover_replays(root, *, single_run=False):
        if single_run:
            return [path for path in replays if path.parents[3] == root.resolve()]
        return list(reversed(replays))

    monkeypatch.setattr(inspector, "discover_replays", discover_replays)
    monkeypatch.setattr(inspector, "discover_heatmaps", lambda *_: [])
    monkeypatch.setattr(inspector, "discover_roadmap_tests", lambda *_: [roadmap.resolve()])
    monkeypatch.setattr(inspector, "replay_label", lambda path: f"{path.parents[3].name}_[1M]_[Replay]")
    monkeypatch.setattr(inspector, "roadmap_label", lambda _: "route_[Roadmap]")

    server = ThreadingHTTPServer(("127.0.0.1", 0), inspector.make_handler([], root=tmp_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}/api/replays"
        with urllib.request.urlopen(base + "?recent=3") as response:
            recent = json.load(response)
            assert response.headers["X-Inspector-Has-More"] == "true"
        assert len({item["run_id"] for item in recent}) == 3
        assert all(item["kind"] == "replay" for item in recent)
        assert recent[0]["selected"] is True

        with urllib.request.urlopen(base) as response:
            complete = json.load(response)
            assert response.headers["X-Inspector-Has-More"] == "false"
        assert len(complete) == 5
        assert {item["kind"] for item in complete} == {"replay", "roadmap"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


# Building editor

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
            "src/swarmecho/curriculum_config/maps/custom_maps/open_cuboid.yaml",
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
