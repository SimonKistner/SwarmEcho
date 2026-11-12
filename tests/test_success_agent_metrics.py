from types import SimpleNamespace

import numpy as np
import pytest

from swarmecho.training.validation import summarize_outcomes
from swarmecho.training import evaluate, random_evaluation


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
