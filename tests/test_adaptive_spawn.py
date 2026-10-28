"""Focused curriculum calculation and JAX reset integration checks.

Run in the project's WSL environment; these checks do not train a policy.
"""
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from swarmecho.core.config import AdaptiveTargetSpawnConfig, EnvConfig, RewardConfig
from swarmecho.env.buildings import load_building, target_spawn_boxes
from swarmecho.env.environment import make_env_fns, make_autoreset_fns
from swarmecho.training.adaptive_spawn import AdaptiveTargetSpawnController


BUILDING = load_building(Path("src/swarmecho/curriculum_config/maps/B02c_static_map_noSpawn.yaml"))
ENV = replace(EnvConfig(), num_agents=1, success_condition="chain_held", max_steps=3)


def controller(config=None):
    return AdaptiveTargetSpawnController(BUILDING, ENV, config or AdaptiveTargetSpawnConfig(enabled=True))


def cell_points(curriculum):
    lower, upper, _ = target_spawn_boxes(BUILDING, ENV)
    centres = (lower + upper) / 2
    return np.asarray([centres[np.flatnonzero(curriculum.box_cell_indices == i)[0]]
                       for i in range(len(curriculum.cells))])


def assert_same_tree(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
        np.testing.assert_array_equal(a, b)


def test_volume_baseline_floor_redistribution_and_two_smoothing_stages():
    c = controller()
    points = cell_points(c)
    _, _, volumes = target_spawn_boxes(BUILDING, ENV)
    assert len(c.cells) == 23
    np.testing.assert_allclose(c.box_weights, volumes / volumes.sum(), rtol=2e-7)

    # A half-success cell loses probability, while untouched cells stay at zero success.
    c.observe(np.repeat(points[:1], 10, axis=0), [1] * 5 + [0] * 5)
    assert c.rates[0] == .5 and c.rates[1] == 0
    c.observe(np.empty((0, 3)), [])
    assert c.rates[0] == .5
    c.advance(1, 100)
    assert c.probabilities[0] < c.baseline[0]

    c = controller()
    c.observe(np.repeat(points, 500, axis=0), np.repeat([0] + [1] * 22, 500))
    expected = .2 * c.baseline.copy()
    expected[0] += .8
    start = c.probabilities.copy()
    for update in range(1, 4):
        c.advance(update, update * 100)
    np.testing.assert_allclose(c.probabilities, (start + expected) / 2, atol=1e-12)

    # Replace just the difficult cell's last 500 outcomes: other windows are unchanged.
    c.observe(np.repeat(points[:1], 500, axis=0), [0] * 250 + [1] * 250)
    expected = .6 * c.baseline.copy()
    expected[0] += .4
    start = c.probabilities.copy()
    for update in range(4, 7):
        row, _ = c.advance(update, update * 100)
        assert np.all(np.asarray(row["probabilities"]) >= .2 * c.baseline - 1e-12)
    np.testing.assert_allclose(c.probabilities, (start + expected) / 2, atol=1e-12)
    np.testing.assert_allclose(np.bincount(c.box_cell_indices, weights=c.box_weights),
                               c.probabilities, rtol=2e-7)

    c.observe(np.repeat(points[:1], 500, axis=0), [1] * 500)
    start = c.probabilities.copy()
    for update in range(7, 10):
        c.advance(update, update * 100)
    np.testing.assert_allclose(c.probabilities, (start + c.baseline) / 2, atol=1e-12)


def test_dynamic_weights_reach_jitted_resets_without_retracing():
    c = controller(replace(AdaptiveTargetSpawnConfig(enabled=True), min_probability_fraction=0.,
                           probability_smoothing_half_life_updates=1e-6))
    reset, _, _, _ = make_env_fns(BUILDING, ENV, plan_geodesic=False)
    traces = []

    def draw(keys, weights):
        traces.append(True)
        return jax.vmap(lambda key: reset(key, target_spawn_weights=weights).target_pos)(keys)

    draw_jit = jax.jit(draw)
    keys = jax.random.split(jax.random.PRNGKey(117), 64)
    for update, selected in enumerate((0, len(c.cells) - 1), start=1):
        # Two limiting distributions from the real controller, using the same compiled sampler.
        c.rates[:] = 1.
        c.rates[selected] = 0.
        c.advance(update, update * 100)
        targets = draw_jit(keys, jnp.asarray(c.box_weights))
        cells = np.floor(np.asarray(targets) / c.cell_size).astype(int)
        np.testing.assert_array_equal(cells, np.repeat(c.cells[selected:selected + 1], 64, axis=0))
    assert len(traces) == 1


def test_disabled_reset_keeps_the_legacy_keyed_volume_draw():
    reset, _, _, _ = make_env_fns(BUILDING, ENV, plan_geodesic=False)
    lower, upper, volumes = map(jnp.asarray, target_spawn_boxes(BUILDING, ENV))

    def legacy_first_draw(key):
        # Freeze the pre-feature PRNG contract independently of reset/autoreset wiring.
        target_key = jax.random.split(key, 3)[1]
        first_key = jax.random.split(target_key)[1]
        box_key, point_key = jax.random.split(first_key)
        box = jax.random.categorical(box_key, jnp.log(volumes))
        return jax.random.uniform(point_key, (3,), minval=lower[box], maxval=upper[box])

    keys = jax.random.split(jax.random.PRNGKey(883), 64)
    default = jax.jit(jax.vmap(reset))(keys)
    explicit_none = jax.jit(jax.vmap(lambda key: reset(key, target_spawn_weights=None)))(keys)
    assert_same_tree(default, explicit_none)
    np.testing.assert_array_equal(default.target_pos, jax.jit(jax.vmap(legacy_first_draw))(keys))


@pytest.mark.parametrize("sparse", [False, True])
def test_weights_reach_dense_and_sparse_autoresets_and_disabled_results_match(sparse):
    c = controller()
    reset, step, _, _ = make_autoreset_fns(BUILDING, ENV, RewardConfig(chain_reward_system="euclidean"))
    states = jax.jit(jax.vmap(reset))(jax.random.split(jax.random.PRNGKey(227), 4))
    # A 50-step chain hold cannot finish here: only selected lanes reach the time limit.
    steps = jnp.asarray([2, 0, 0, 0] if sparse else [2, 2, 2, 2], dtype=jnp.int32)
    states = states._replace(step=steps)
    actions = jnp.zeros((4, ENV.num_agents, 3))
    default = jax.jit(step.batched)(states, actions)
    explicit_none = jax.jit(lambda s, a: step.batched(s, a, None))(states, actions)
    scalar_reference = jax.jit(jax.vmap(step))(states, actions)
    assert_same_tree(default, explicit_none)
    assert_same_tree(default, scalar_reference)

    selected = len(c.cells) - 1
    weights = np.where(c.box_cell_indices == selected,
                       c.box_volumes / c.cell_volumes[selected], 0.).astype(np.float32)
    batched = jax.jit(step.batched)
    updated, _, done, _ = batched(states, actions, jnp.asarray(weights))
    finished = np.asarray(done, dtype=bool)
    assert finished.sum() == (1 if sparse else 4)
    cells = np.floor(np.asarray(updated.target_pos)[finished] / c.cell_size).astype(int)
    np.testing.assert_array_equal(cells, np.repeat(c.cells[selected:selected + 1], finished.sum(), axis=0))
    np.testing.assert_array_equal(np.asarray(updated.target_pos)[~finished], np.asarray(states.target_pos)[~finished])


def test_controller_checkpoint_restores_windows_and_next_update(tmp_path):
    original = controller()
    original.observe(cell_points(original)[[0, 1]], [1, 0])
    original.advance(1, 100)
    original.save(tmp_path)
    restored = controller()
    assert restored.restore(tmp_path)
    assert original.advance(2, 200) == restored.advance(2, 200)
    np.testing.assert_array_equal(original.box_weights, restored.box_weights)
