from pathlib import Path
from types import SimpleNamespace

import numpy as np

from swarmecho.training.artifacts import load_eval_info_csv
from swarmecho.training.artifacts import save_eval_info_csv
from swarmecho.training.evaluate import (
    diverse_success_lanes,
    run_parallel_evaluation,
    select_eval_replay_lanes,
    select_eval_target_with_lane,
)


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
        ),
        training=SimpleNamespace(seed=7, num_envs=4000, num_steps=100),
        building_name="test_map",
        building=SimpleNamespace(world_size_m=np.asarray([20.0, 20.0, 30.0])),
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
        return {"eval_success": float(np.mean(successes))}, {
            "target_positions": targets.copy(),
            "base_positions": bases.copy(),
            "successes": successes,
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
