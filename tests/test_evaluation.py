"""Evaluation metrics, artifact formats, and checkpoint replay checks."""

from pathlib import Path
from types import SimpleNamespace
import csv
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from swarmecho.env.buildings import load_building, make_cuboid_building
from swarmecho.env.environment import EnvConfig, make_env_fns
from swarmecho.training import evaluate, random_evaluation, random_evaluation as evaluation
from swarmecho.training.artifacts import (
    evaluation_stage_from_rates,
    load_eval_info_csv,
    load_eval_layout,
    save_eval_info_csv,
    save_eval_layout,
)
from swarmecho.training.evaluate import (
    diverse_success_lanes,
    run_parallel_evaluation,
    select_eval_replay_lanes,
    select_eval_target_with_lane,
)
from swarmecho.training.validation import summarize_outcomes
from swarmecho.visualize.replay import load_replay, write_replay


# Evaluation and replay selection

def test_replay_only_cli_has_progress_logging_and_no_subprocess_duplication():
    source = Path("src/swarmecho/training/evaluate.py").read_text()
    assert "def render_csv_replays(" in source
    assert "[REPLAY {replay_number + 1}/{replay_count}]" in source
    assert "subprocess.run" not in source


def test_diverse_successes_balance_chain_length_and_spatial_separation():
    records = {
        "positions": np.asarray([[0, 0, 0], [0.1, 0, 0], [10, 0, 0], [0, 10, 0]]),
        "final_chain_length": np.asarray([100.0, 99.0, 95.0, 94.0]),
    }
    selected = diverse_success_lanes(records, np.arange(4), limit=3)
    assert selected.shape == (3,)
    assert selected[0] == 0
    assert set(selected[:3]) == {0, 2, 3}


def test_diverse_success_selection_does_not_rank_all_4000_lanes():
    records = {
        "positions": np.random.default_rng(1).uniform(size=(4000, 3)),
        "final_chain_length": np.arange(4000, dtype=np.float32),
    }
    selected = diverse_success_lanes(records, np.arange(4000), limit=3)
    assert selected.shape == (3,)
    assert selected[0] == 3999


def test_replay_configs_are_selected_only_from_csv_metrics():
    records = {
        "positions": np.asarray([[0, 0, 0], [0.1, 0, 0], [10, 0, 0], [0, 10, 0]]),
        "final_chain_length": np.asarray([100.0, 99.0, 95.0, 94.0]),
        "stages": np.asarray(["chain_success"] * 4),
    }
    lanes, label = select_eval_replay_lanes(records, result="success", count=3)
    assert label == "SUCCESS"
    assert lanes.tolist() == [0, 2, 3]


def test_parallel_evaluation_reuses_targets_and_writes_five_run_rates(
    tmp_path, monkeypatch
):
    level = SimpleNamespace(
        evaluation=SimpleNamespace(
            eval_parallel_envs=2,
            eval_robustness_runs=5,
            eval_action_noise_max=0.011,
            success_condition=None,
            eval_differes_from_training_map=False,
        ),
        env=SimpleNamespace(success_condition="chain_held", max_steps=100),
        reward=SimpleNamespace(target_found_requires_delivery=True),
        training=SimpleNamespace(seed=7, num_envs=4000, num_steps=100),
        building_name="test_map",
        building=make_cuboid_building((4, 4, 6)),
    )
    targets = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    bases = np.zeros((2, 3))
    success_by_run = [
        [True, True],
        [True, True],
        [True, True],
        [True, True],
        [False, True],
    ]
    calls = []

    def fake_evaluate_suite(model, passed_level, **kwargs):
        calls.append(kwargs)
        run_index = kwargs["action_noise_seed"] - (passed_level.training.seed + 20_000)
        successes = np.asarray(success_by_run[run_index])
        return {"eval_success": float(np.mean(successes)),
                "eval_episode_length": 10.0, "eval_coverage": 0.5,
                "eval_target_found_rate": float(np.mean(successes)),
                "eval_visually_found_rate": 1.0,
                "eval_chain_progress_pct": 50.0}, {
            "target_positions": targets.copy(),
            "base_positions": bases.copy(),
            "successes": successes,
            "lengths": np.where(successes, 10, 100),
            "delivered": successes.copy(),
            "visually_found": np.ones(2, dtype=bool),
            "final_chain_lengths": np.asarray([7.0, 8.0]),
            "obstacle_min": np.empty((2, 0, 3), dtype=np.float32),
            "obstacle_max": np.empty((2, 0, 3), dtype=np.float32),
        }

    monkeypatch.setattr(
        "swarmecho.training.evaluate.evaluate_suite",
        fake_evaluate_suite,
    )
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoints" / "ckpt_000001"

    path = run_parallel_evaluation(object(), level, checkpoint, run_dir)
    records = load_eval_info_csv(path)

    assert len(calls) == 5
    assert [call["action_noise_seed"] for call in calls] == [
        20_007,
        20_008,
        20_009,
        20_010,
        20_011,
    ]
    assert all(call["action_noise_max"] == 0.011 for call in calls)
    np.testing.assert_allclose(records["chain_success_rate"], [0.8, 1.0])
    np.testing.assert_allclose(
        records["found_and_delivered_rate"], [0.8, 1.0]
    )
    np.testing.assert_allclose(records["visually_found_rate"], [1.0, 1.0])
    assert records["stages"].tolist() == ["visually_found", "chain_success"]
    assert path.parent.name == "data"
    assert path.parent.parent.name.startswith("eval_")
    assert path.parent.parent.parent.name == "ckpt_u000001_s00400k"


def test_successful_replay_priority_uses_final_chain_length(tmp_path):
    path = save_eval_info_csv(
        tmp_path / "eval.csv",
        target_positions=np.asarray([[20, 0, 0], [10, 0, 0]], dtype=np.float32),
        base_positions=np.zeros((2, 3), dtype=np.float32),
        successes=[True, True], delivered=[True, True], visually_found=[True, True],
        final_chain_lengths=[21.0, 30.0],
    )
    target, label, lane = select_eval_target_with_lane(
        path, result="success", offset=0
    )
    assert label == "SUCCESS_0"
    assert lane == 1
    np.testing.assert_allclose(target, [10, 0, 0])


def test_final_checkpoint_evaluation_falls_back_to_failure(tmp_path, monkeypatch, capsys):
    from swarmecho.training.train import _create_final_checkpoint_evaluation

    info_path = tmp_path / "eval.csv"
    results = []
    monkeypatch.setattr(
        "swarmecho.training.artifacts.create_eval_run_root",
        lambda *args, **kwargs: tmp_path / "eval_run",
    )
    monkeypatch.setattr(
        "swarmecho.training.evaluate.run_parallel_evaluation",
        lambda *args, **kwargs: info_path,
    )

    def fake_render(*args, result, **kwargs):
        results.append(result)
        if result == "success":
            raise ValueError("no successful lanes")

    monkeypatch.setattr(
        "swarmecho.training.evaluate.render_csv_replays", fake_render
    )
    _create_final_checkpoint_evaluation(
        object(), object(), tmp_path / "ckpt", tmp_path
    )

    assert results == ["success", "fail"]
    assert "[WARNING] Final success replay selection failed" in capsys.readouterr().out


def test_final_checkpoint_evaluation_skips_replay_after_both_selections_fail(
    tmp_path, monkeypatch, capsys
):
    from swarmecho.training.train import _create_final_checkpoint_evaluation

    monkeypatch.setattr(
        "swarmecho.training.artifacts.create_eval_run_root",
        lambda *args, **kwargs: tmp_path / "eval_run",
    )
    monkeypatch.setattr(
        "swarmecho.training.evaluate.run_parallel_evaluation",
        lambda *args, **kwargs: tmp_path / "eval.csv",
    )
    monkeypatch.setattr(
        "swarmecho.training.evaluate.render_csv_replays",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("no lanes")),
    )

    _create_final_checkpoint_evaluation(
        object(), object(), tmp_path / "ckpt", tmp_path
    )
    output = capsys.readouterr().out
    assert "[WARNING] Final success and failure replay creation both failed" in output


# Evaluation CSVs

def test_static_eval_layout_is_stored_once(tmp_path):
    lower = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    upper = lower + 1
    path = save_eval_layout(
        tmp_path / "eval.layout.json",
        np.broadcast_to(lower, (4, *lower.shape)),
        np.broadcast_to(upper, (4, *upper.shape)),
    )
    loaded_lower, loaded_upper = load_eval_layout(path)
    np.testing.assert_array_equal(loaded_lower, lower)
    np.testing.assert_array_equal(loaded_upper, upper)


def test_robust_eval_csv_stores_cumulative_rates_and_uses_unanimous_stage(tmp_path):
    path = save_eval_info_csv(
        tmp_path / "eval.csv",
        target_positions=np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        base_positions=np.zeros((2, 3)),
        stage_rates={
            "chain_success": np.asarray([0.8, 1.0]),
            "found_and_delivered": np.asarray([0.8, 1.0]),
            "visually_found": np.asarray([1.0, 1.0]),
        },
    )

    with path.open(newline="") as stream:
        assert next(csv.reader(stream)) == [
            "x",
            "y",
            "z",
            "visually_found_rate",
            "found_and_delivered_rate",
            "chain_success_rate",
            "distance_to_base",
            "final_chain_length",
        ]

    records = load_eval_info_csv(path)
    assert records["obstacle_min"].shape == (2, 0, 3)
    assert records["obstacle_max"].shape == (2, 0, 3)
    assert records["stages"].tolist() == ["visually_found", "chain_success"]
    np.testing.assert_allclose(records["chain_success_rate"], [0.8, 1.0])
    np.testing.assert_allclose(records["found_and_delivered_rate"], [0.8, 1.0])
    np.testing.assert_allclose(records["visually_found_rate"], [1.0, 1.0])
    np.testing.assert_allclose(records["final_chain_length"], [0.0, 0.0])


def test_eval_csv_round_trips_chain_length_and_obstacle_layouts(tmp_path):
    obstacle_min = np.asarray([[[1, 2, 3]], [[4, 5, 6]]], dtype=np.float32)
    obstacle_max = obstacle_min + 1
    path = save_eval_info_csv(
        tmp_path / "obstacles.csv",
        target_positions=np.asarray([[8, 9, 10], [11, 12, 13]], dtype=np.float32),
        base_positions=np.zeros((2, 3), dtype=np.float32),
        successes=[True, True], delivered=[True, True], visually_found=[True, True],
        final_chain_lengths=[17.5, 19.25],
        obstacle_min=obstacle_min, obstacle_max=obstacle_max,
    )
    records = load_eval_info_csv(path)
    np.testing.assert_allclose(records["final_chain_length"], [17.5, 19.25])
    np.testing.assert_allclose(records["obstacle_min"], obstacle_min)
    np.testing.assert_allclose(records["obstacle_max"], obstacle_max)


def test_confidence_selects_highest_cumulative_stage_that_meets_threshold():
    rates = {
        "chain_success_rate": 0.6,
        "found_and_delivered_rate": 0.8,
        "visually_found_rate": 1.0,
    }

    assert evaluation_stage_from_rates(**rates, confidence=1.0) == "visually_found"
    assert (
        evaluation_stage_from_rates(**rates, confidence=0.8)
        == "found_and_delivered"
    )
    assert evaluation_stage_from_rates(**rates, confidence=0.5) == "chain_success"


def test_robust_eval_csv_rejects_non_cumulative_rates(tmp_path):
    with pytest.raises(ValueError, match="chain <= delivered <= visually_found"):
        save_eval_info_csv(
            tmp_path / "bad.csv",
            target_positions=np.asarray([[1.0, 2.0, 3.0]]),
            base_positions=np.zeros((1, 3)),
            stage_rates={
                "chain_success": np.asarray([1.0]),
                "found_and_delivered": np.asarray([0.8]),
                "visually_found": np.asarray([1.0]),
            },
        )


# Replay serialization

def test_replay_round_trip_and_atomic_manifest(tmp_path):
    building = load_building(
        "src/swarmecho/curriculum_config/maps/M00_no_maze_open_cuboid.yaml"
    )
    cfg = EnvConfig()
    reset, step, _, _ = make_env_fns(building, cfg)
    state = reset(jax.random.PRNGKey(3))
    states = [state]
    for _ in range(3):
        state = step(state, jnp.zeros((cfg.num_agents, 3)))
        states.append(state)

    data_path, manifest_path = write_replay(
        tmp_path / "rollout",
        states,
        map_name="M00_no_maze_open_cuboid",
        dt=cfg.dt,
        building=building,
        reward_terms=np.zeros((4, cfg.num_agents, 2), dtype=np.float32),
    )
    assert data_path.exists()
    assert manifest_path.exists()
    assert not list(tmp_path.glob("*.tmp"))

    metadata, arrays = load_replay(manifest_path)
    assert metadata["frames"] == 4
    assert arrays["position"].shape == (4, cfg.num_agents, 3)
    assert arrays["exploration_reward"].shape == (4, 4, 4, 4)
    assert arrays["reward_terms"].shape == (4, cfg.num_agents, 2)
    assert arrays["obstacle_min"].shape == (4, 0, 3)
    np.testing.assert_array_equal(arrays["base_target_known"], [s.base_target_known for s in states])
    assert metadata["building_snapshot"]["geometry"]["tiles"] == np.argwhere(building.tiles).tolist()


def test_replay_rejects_unknown_format(tmp_path):
    manifest = tmp_path / "bad.json"
    manifest.write_text(json.dumps({"format": "unknown"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported replay format"):
        load_replay(manifest)


# Required-agent replays

def replay_fixture(monkeypatch, *, actual=None):
    cfg = SimpleNamespace(geometric_agent_budget=7, max_steps=700, dt=0.1,
                          coverage_voxel_size=2.5, comm_radius=15,
                          comm_radius_base=15, visual_radius=5,
                          success_condition="chain_held")
    level = SimpleNamespace(env=cfg, building_name="random_eval_0000",
                            building=SimpleNamespace(cell_size_m=5),
                            reward=SimpleNamespace(allow_redundancy_reward=True))
    required = np.array([2, 1, 2, 7])
    targets = np.array([[i, 0, 0] for i in range(4)], dtype=np.float32)
    info = {"required_agents": required, "target_positions": targets}
    calls, outputs = [], []

    def collect(model, level, **kwargs):
        lane = int(kwargs["target_position"][0])
        calls.append(lane)
        return [SimpleNamespace(target_required_agents=required[lane] if actual is None else actual)], None

    monkeypatch.setattr("swarmecho.training.train.evaluate_model", collect)
    monkeypatch.setattr(evaluation, "planner_metadata", lambda cfg: {})
    monkeypatch.setattr(evaluation, "write_replay", lambda path, states, **kwargs: outputs.append((path, kwargs)))
    return level, info, calls, outputs


def test_one_replay_per_observed_required_agent_class(tmp_path, monkeypatch, capsys):
    level, info, calls, outputs = replay_fixture(monkeypatch)
    evaluation._write_required_agent_replays(None, level, info, tmp_path, "u001050_s00315M", {})
    assert calls == [1, 0, 3]
    assert [item[1]["metadata"]["required_agents"] for item in outputs] == [1, 2, 7]
    assert len({item[0] for item in outputs}) == 3
    assert "no sampled target requiring 3 agent(s)" in capsys.readouterr().out


def test_replay_class_must_match_evaluated_class(tmp_path, monkeypatch):
    level, info, calls, outputs = replay_fixture(monkeypatch, actual=2)
    with pytest.raises(ValueError, match="class changed from 1 to 2"):
        evaluation._write_required_agent_replays(None, level, info, tmp_path, "test", {})
    assert outputs == []


def test_reject_unclassified_targets(tmp_path, monkeypatch):
    level, info, calls, outputs = replay_fixture(monkeypatch)
    info["required_agents"][0] = 0
    with pytest.raises(ValueError, match="classified targets"):
        evaluation._write_required_agent_replays(None, level, info, tmp_path, "test", {})
    assert calls == []


def test_map_suite_requests_classes_and_uses_class_replays(tmp_path, monkeypatch):
    level, info, calls, outputs = replay_fixture(monkeypatch)
    level.env.num_agents = 7
    level.building.world_size_m = np.array([15, 15, 15])
    level.evaluation = SimpleNamespace(map_suite_replay_limit=None,
                                       target_sampling="independent", training_heatmap_creation=False)
    level.logging = SimpleNamespace(individual_map_diagnostics=False)
    level.map_source_name = None
    level.effective_eval_parallel_envs = 4
    suite_calls, replay_calls = [], []

    def evaluate(model, level, **kwargs):
        suite_calls.append(kwargs)
        return {"eval_success": 0.5, "eval_success_episode_count": 2,
                "eval_success_episode_length": 20}, info

    monkeypatch.setattr("swarmecho.training.train.evaluate_suite", evaluate)
    monkeypatch.setattr(evaluation, "resolve_evaluation_level", lambda level: level)
    monkeypatch.setattr(evaluation, "building_snapshot", lambda building: {})
    monkeypatch.setattr(evaluation, "_write_required_agent_replays",
                        lambda *args: replay_calls.append(args))
    metrics = evaluation.evaluate_map_suite(None, [level], tmp_path, update=1050,
                                            steps=315_000_000, replay_due=True,
                                            replay_by_required_agents=True)
    assert suite_calls[0]["return_required_agents"] is True
    assert len(replay_calls) == 1
    assert replay_calls[0][2] is info
    assert calls == []  # The ordinary random replay path must not also run.
    assert metrics["eval_success"] == 0.5


# Success and population metrics

@pytest.mark.parametrize("success, expected", [([True, False, True], 2.), ([False] * 3, None)])
def test_validation_useless_agents_excludes_failures(success, expected):
    outcomes = {name: np.ones(3) for name in
                ("return", "length", "population", "target_found", "visually_found")}
    outcomes.update(success=np.array(success), useless_agents=np.array([1, 99, 3]))
    metrics, _ = summarize_outcomes(np.zeros(3, dtype=int), outcomes, ["map"], 100)
    assert metrics.get("useless_agents_at_success") == expected


@pytest.mark.parametrize("counts", [(0, 1, 3), (1, 0, 3), (0, 0, 0)])
def test_robust_evaluation_pools_successful_agent_counts(monkeypatch, counts):
    level = SimpleNamespace(effective_eval_parallel_envs=3,
        evaluation=SimpleNamespace(eval_robustness_runs=3, eval_action_noise_max=.01),
        training=SimpleNamespace(seed=7), env=SimpleNamespace(success_condition="chain_held", max_steps=100),
        reward=SimpleNamespace(target_found_requires_delivery=True))

    def batch(model, level, *, action_noise_seed, **kwargs):
        run = action_noise_seed - 20_007
        success = np.arange(3) < counts[run]
        metrics = {"eval_success": float(success.mean()), "eval_success_episode_count": counts[run]}
        if counts[run]:
            metrics["eval_useless_agents_at_success"] = [7., 1., 3.][run]
        return metrics, dict(target_positions=np.zeros((3, 3)), base_positions=np.zeros((3, 3)),
            final_chain_lengths=np.ones(3), obstacle_min=np.empty((3, 0, 3)), obstacle_max=np.empty((3, 0, 3)),
            successes=success, lengths=np.ones(3), delivered=success, visually_found=success)

    monkeypatch.setattr(evaluate, "resolve_evaluation_level", lambda level: level)
    monkeypatch.setattr(evaluate, "evaluate_suite", batch)
    metrics, _ = evaluate.collect_robust_evaluation(None, level, on_run=lambda *args: None)
    if sum(counts):
        assert metrics["eval_useless_agents_at_success"] == pytest.approx(
            np.dot(counts, [7., 1., 3.]) / sum(counts))
    else:
        assert "eval_useless_agents_at_success" not in metrics


@pytest.mark.parametrize("counts", [(0, 1, 3), (1, 0, 3), (0, 0, 0)])
def test_map_suite_pools_successful_agent_counts(tmp_path, monkeypatch, counts):
    from swarmecho.training import train

    levels = [SimpleNamespace(
        building_name=f"static_eval_{i:04d}", map_source_name=None,
        env=SimpleNamespace(num_agents=3, max_steps=100), effective_eval_parallel_envs=3,
        evaluation=SimpleNamespace(map_suite_replay_limit=None, target_sampling="independent",
                                   training_heatmap_creation=False),
        logging=SimpleNamespace(individual_map_diagnostics=False),
        building=SimpleNamespace(world_size_m=np.ones(3)), index=i) for i in range(3)]

    def batch(model, level, **kwargs):
        metrics = dict(eval_success=counts[level.index] / 3,
                       eval_success_episode_count=counts[level.index], eval_success_episode_length=10.)
        if counts[level.index]:
            metrics["eval_useless_agents_at_success"] = [7., 1., 3.][level.index]
        return metrics, {}

    monkeypatch.setattr(train, "evaluate_suite", batch)
    monkeypatch.setattr(random_evaluation, "resolve_evaluation_level", lambda level: level)
    monkeypatch.setattr(random_evaluation, "building_snapshot", lambda building: {})
    metrics = random_evaluation.evaluate_map_suite(None, levels, tmp_path, update=1, steps=10)
    if sum(counts):
        assert metrics["eval_useless_agents_at_success"] == pytest.approx(
            np.dot(counts, [7., 1., 3.]) / sum(counts))
    else:
        assert "eval_useless_agents_at_success" not in metrics
