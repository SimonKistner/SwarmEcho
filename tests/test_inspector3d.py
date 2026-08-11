import jax
import jax.numpy as jnp

from swarmecho.core.config import load_level_3d
from swarmecho.env.baseline3d import make_baseline_3d_fns
from swarmecho.visualize.inspector3d import (
    HTML,
    discover_replays,
    heatmap_label,
    inspector_html,
    replay_payload,
    replay_label,
)
from swarmecho.visualize.replay3d import write_replay


def test_inspector_payload_and_controls(tmp_path):
    level = load_level_3d()
    reset, step, _, _ = make_baseline_3d_fns(level.building, level.env)
    state = reset(jax.random.PRNGKey(0))
    states = [state, step(state, jnp.zeros((level.env.num_agents, 3)))]
    _, manifest = write_replay(
        tmp_path / "inspect",
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
    assert 'id="showCoverage"' in HTML
    assert 'id="showVisualRange"' in HTML
    assert 'id="showCommRange"' in HTML
    assert 'id="refreshReplays"' in HTML
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
    assert replay_label(manifest) == "inspect"


def test_replay_label_uses_run_name_before_artifacts(tmp_path):
    manifest = tmp_path / "M00_3d_baseline" / "artifacts" / "train" / "replays" / "eval.json"
    assert replay_label(manifest) == "M00_3d_baseline"


def test_replay_label_adds_unpadded_training_steps(tmp_path):
    manifest = tmp_path / "M00_3d_baseline_v2" / "artifacts" / "train" / "replays" / "eval_u000061_s00024M.json"
    assert replay_label(manifest) == "M00_3d_baseline_v2-[24M]"


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
