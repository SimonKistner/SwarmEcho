import json

import jax
import jax.numpy as jnp

from swarmecho.core.config import load_level
from swarmecho.env.environment import make_env_fns
from swarmecho.visualize.inspector import (
    HTML,
    discover_heatmaps,
    discover_replays,
    discover_roadmap_tests,
    heatmap_label,
    inspector_html,
    replay_payload,
    replay_label,
    building_geometry,
)
from swarmecho.training.artifacts import save_eval_info_csv
from swarmecho.visualize.replay import write_replay
from swarmecho.visualize.replay import building_snapshot


def test_snapshot_geometry_survives_missing_original_map():
    level = load_level("B01b_office")
    snapshot = building_snapshot(level.building)
    geometry = building_geometry({"map_name": "does_not_exist", "building_snapshot": snapshot})
    assert geometry["source"] == "replay snapshot"
    assert len(geometry["walls"]) > 0
    assert geometry["solid_min"] == level.building.solid_min_m.tolist()
    assert geometry["solid_max"] == level.building.solid_max_m.tolist()


def test_legacy_office_visibility_bounds_match_runtime():
    level = load_level("B01b_office")
    geometry = building_geometry({"map_name": level.building_name})
    actual = sorted(tuple(lo + hi) for lo, hi in zip(geometry["solid_min"], geometry["solid_max"]))
    expected = sorted(tuple(lo.tolist() + hi.tolist()) for lo, hi in zip(level.building.solid_min_m, level.building.solid_max_m))
    assert actual == expected


def test_inspector_payload_and_controls(tmp_path):
    level = load_level()
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
    manifest = tmp_path / "M00_3d_baseline" / "artifacts" / "train" / "replays" / "eval.json"
    assert replay_label(manifest) == "M00_3d_baseline_[unknown]_[Replay]"


def test_replay_label_adds_unpadded_training_steps(tmp_path):
    manifest = tmp_path / "M00_3d_baseline_v2" / "artifacts" / "train" / "replays" / "eval_u000061_s00024M.json"
    assert replay_label(manifest) == "M00_3d_baseline_v2_[24M]_[Replay]"


def test_heatmap_labels_distinguish_training_and_standalone_evaluation(tmp_path):
    run = tmp_path / "M00_3d_baseline"
    train_heatmap = (
        run / "artifacts/train/data/eval_info_u000061_s00024M.csv"
    )
    eval_heatmap = (
        run
        / "artifacts/eval/ckpt_u000351_s00140M/data/"
        "eval_info_u000351_s00140M.csv"
    )

    assert heatmap_label(train_heatmap) == (
        "M00_3d_baseline_[24M]_[TRAIN Heatmap]"
    )
    assert heatmap_label(eval_heatmap) == (
        "M00_3d_baseline_[140M]_[EVAL Heatmap]"
    )


def test_inspector_discovers_new_eval_runs_and_hides_timestamps_in_labels(tmp_path):
    run = tmp_path / "M00_3d_baseline"
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
        "M00_3d_baseline_[140M]_[EVAL Heatmap]_[agents7]"
    )
    assert replay_label(replay_manifest) == (
        "M00_3d_baseline_[140M]_[Replay]_[agents7]"
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
