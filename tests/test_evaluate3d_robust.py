from pathlib import Path
from types import SimpleNamespace

import numpy as np

from swarmecho.training.artifacts import load_eval_info_csv
from swarmecho.training.artifacts import save_eval_info_csv
from swarmecho.training.evaluate3d import (
    diverse_success_lanes,
    run_parallel_evaluation_3d,
    select_eval_target_with_lane,
)


def test_replay_only_cli_has_progress_logging_and_no_subprocess_duplication():
    source = Path("src/swarmecho/training/evaluate3d.py").read_text()
    assert "def render_csv_replays(" in source
    assert "[REPLAY {replay_number + 1}/{replay_count}]" in source
    assert "subprocess.run" not in source


def test_diverse_successes_balance_chain_length_and_spatial_separation():
    records = {
        "positions": np.asarray([[0, 0, 0], [0.1, 0, 0], [10, 0, 0], [0, 10, 0]]),
        "final_chain_length": np.asarray([100.0, 99.0, 95.0, 94.0]),
    }
    selected = diverse_success_lanes(records, np.arange(4))
    assert selected[0] == 0
    assert set(selected[:3]) == {0, 2, 3}


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
        "swarmecho.training.evaluate3d.evaluate_suite_3d",
        fake_evaluate_suite,
    )
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoints" / "ckpt_000001"

    path = run_parallel_evaluation_3d(object(), level, checkpoint, run_dir)
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
