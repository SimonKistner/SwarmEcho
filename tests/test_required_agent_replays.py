from types import SimpleNamespace

import numpy as np
import pytest

from swarmecho.training import random_evaluation as evaluation


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
