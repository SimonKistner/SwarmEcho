"""Information-source attribution, episode pooling and rolling eviction."""
from collections import deque
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from swarmecho.env.knowledge import information_counts, new_information_counts
from swarmecho.training.information_metrics import (
    information_percentages, evaluation_information_metrics, pool_evaluation_information,
)


def test_first_acquisitions_prioritize_sight_then_base_and_exclude_inactive():
    # Existing knower; direct sight + base; base + peer; peer; inactive; unknown.
    counts = jax.jit(new_information_counts)(
        jnp.array([1, 0, 0, 0, 0, 0], bool),
        jnp.array([1, 1, 1, 1, 1, 0], bool),
        jnp.array([1, 1, 1, 1, 0, 1], bool),
        jnp.array([0, 1, 0, 0, 0, 0], bool),
        jnp.array([1, 1, 1, 0, 1, 0], bool))
    np.testing.assert_array_equal(counts, [3, 1, 1])


def test_reset_sightings_count_once_and_reused_slots_can_learn_again():
    previous = SimpleNamespace(step=0, active=jnp.array([1, 1], bool),
                               target_known=jnp.array([1, 0], bool))
    current = SimpleNamespace(new_information_counts=jnp.array([1, 0, 1]))
    np.testing.assert_array_equal(information_counts(previous, current), [2, 0, 1])
    previous.step = 1
    np.testing.assert_array_equal(information_counts(previous, current), [1, 0, 1])
    # Respawning clears knowledge; the new occupant counts on acquisition.
    np.testing.assert_array_equal(new_information_counts(
        jnp.array([0], bool), jnp.array([1], bool), jnp.array([1], bool),
        jnp.array([0], bool), jnp.array([1], bool)), [1, 0, 1])


def test_completed_episode_window_pools_agents_and_evicts_old_episodes():
    window = deque(maxlen=2)
    window.extend(([10, 9, 0], [2, 0, 1]))
    assert information_percentages(window) == {"peer-informed-pct": 75., "base-informed-pct": 100/12}
    window.append([0, 0, 0])  # Zero-information episodes still occupy the window.
    assert information_percentages(window) == {"peer-informed-pct": 0., "base-informed-pct": 50.}
    window.append([0, 0, 0])
    assert information_percentages(window) == {}


@pytest.mark.parametrize("counts", [([0, 0, 0], [10, 9, 1]), ([2, 0, 1], [10, 9, 0])])
def test_map_and_robustness_pooling_uses_counts_even_if_first_pass_has_no_information(counts):
    result = pool_evaluation_information([evaluation_information_metrics(row) for row in counts])
    assert result == evaluation_information_metrics(counts)
    assert result["eval_peer-informed-pct"] == pytest.approx(900 / sum(row[0] for row in counts))


def test_environment_emits_source_counts_with_real_radio_cadence():
    from dataclasses import replace
    from pathlib import Path
    from swarmecho.core.config import EnvConfig
    from swarmecho.env.buildings import load_building
    from swarmecho.env.environment import make_env_fns

    building = load_building(Path("src/swarmecho/curriculum_config/maps/M00_no_maze_open_cuboid.yaml"))
    cfg = replace(EnvConfig(), num_agents=3, spawn_delay=0, comm_radius=1.5,
                  comm_radius_base=1.5, visual_radius=1., max_steps=20,
                  no_movement_termination_steps=20, hold_chain_for=20)
    reset, step, _, _ = make_env_fns(building, cfg, memory_comm_every_k_steps=3)
    state = reset(jax.random.PRNGKey(91), target_pos=jnp.array([17., 17., 12.]))._replace(
        pos=jnp.array([[3., 2., 2.], [4., 2., 2.], [5., 2., 2.]]),
        vel=jnp.zeros((3, 3)), base_pos=jnp.array([2., 2., 2.]),
        active=jnp.ones(3, bool), target_known=jnp.array([1, 0, 0], bool),
        base_target_known=jnp.bool_(True), step=jnp.int32(3))
    # Recipient 1 receives a peer message; recipient 2 waits for a later tick.
    action = jnp.zeros((3, 3))
    current = step(state, action)
    np.testing.assert_array_equal(information_counts(state, current), [1, 1, 0])
    state = current
    current = step(state, action)
    np.testing.assert_array_equal(information_counts(state, current), [0, 0, 0])
    # Both base and informed peer are in range: base takes precedence.
    state = state._replace(pos=state.pos.at[2].set(jnp.array([3.2, 2., 2.])), step=jnp.int32(6))
    current = step(state, action)
    np.testing.assert_array_equal(information_counts(state, current), [1, 0, 1])


def test_validation_counts_only_its_episodes_excluding_batch_padding(tmp_path, monkeypatch):
    from dataclasses import replace
    from pathlib import Path
    import json
    from swarmecho.core.config import load_level
    from swarmecho.training import train
    from swarmecho.training.validation import ValidationRunner

    level = load_level("G03_CLU_random_buildings")
    level = replace(level, validation=replace(level.validation, num_maps=1,
        num_episodes=3, parallel_envs=2, fixed_targets=False))
    runner = ValidationRunner(level, tmp_path,
        [Path("src/swarmecho/curriculum_config/maps/M00_no_maze_open_cuboid.yaml")])
    monkeypatch.setattr(runner, "_prepare", lambda: (None, None, None, None))
    calls = []

    def batch(*args, **kwargs):
        assert kwargs["return_information"]
        values = [np.zeros(2) for _ in range(18)]
        values[8] = np.zeros((2, 3))
        values[12] = np.ones(2)  # Valid target samples.
        values[14] = np.ones(2, bool)  # Valid decommission zones.
        values[17] = np.array([[4, 1, 2], [2, 1, 1]] if not calls else
                              [[1, 1, 0], [999, 0, 999]])
        calls.append(True)
        return tuple(values)

    monkeypatch.setattr(train, "_run_parallel_evaluation_jit", batch)
    metrics = runner.run(SimpleNamespace(), update=1, steps=100)
    assert metrics["peer-informed-pct"] == pytest.approx(300/7)
    assert metrics["base-informed-pct"] == pytest.approx(300/7)
    result = json.loads((tmp_path / "validation/results_u000001.json").read_text())
    assert result["information_counts"] == [7, 3, 3]


def test_real_training_update_reports_sources_from_completed_episodes(tmp_path):
    from dataclasses import replace
    from test_adapt_size import tiny_level
    from swarmecho.training.train import train

    level = tiny_level()
    level = replace(level,
        env=replace(level.env, visual_radius=100.),
        training=replace(level.training, num_envs=2, num_steps=6,
                         total_timesteps=12, num_minibatches=1),
        evaluation=replace(level.evaluation, save_model=False, eval_video=False,
                           eval_parallel_envs=2),
        logging=replace(level.logging, wandb_mode="disabled"))
    _, metrics = train(level, output_dir=tmp_path)
    assert metrics["completed_episodes"] >= 2
    # Every deployed drone sees the target directly in this tiny environment.
    assert metrics["peer-informed-pct"] == 0
    assert metrics["base-informed-pct"] == 0
