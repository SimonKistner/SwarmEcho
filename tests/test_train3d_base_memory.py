from dataclasses import replace

import jax
import jax.numpy as jnp

from swarmecho.core.config import load_level
from swarmecho.env.environment import make_env_fns
from swarmecho.training.train import _communication_inputs, _update_base_memory


def test_base_stores_first_reporter_replays_and_clears_on_reset():
    level = load_level()
    reset, _, _, _ = make_env_fns(level.building, level.env)
    state = reset(jax.random.PRNGKey(0))
    state = state._replace(
        active=jnp.ones_like(state.active),
        target_known=state.target_known.at[2].set(True),
    )
    states = jax.tree_util.tree_map(lambda item: item[None], state)
    valid = jnp.asarray([False])
    saved_signature = jnp.zeros((1, 4))
    saved_value = jnp.zeros((1, 3))
    emitted_signature = jnp.arange(16, dtype=jnp.float32).reshape(1, 4, 4)
    emitted_value = jnp.arange(12, dtype=jnp.float32).reshape(1, 4, 3)

    _, in_base, masks, _, _ = _communication_inputs(
        states,
        level.env,
        valid,
        saved_signature,
        saved_value,
    )
    assert not masks.any()
    valid, saved_signature, saved_value = _update_base_memory(
        states,
        in_base,
        emitted_signature,
        emitted_value,
        valid,
        saved_signature,
        saved_value,
        jnp.asarray([False]),
    )
    assert valid[0]
    assert jnp.array_equal(saved_signature[0], emitted_signature[0, 2])
    assert jnp.array_equal(saved_value[0], emitted_value[0, 2])

    _, _, masks, _, _ = _communication_inputs(
        states,
        level.env,
        valid,
        saved_signature,
        saved_value,
    )
    assert masks[0, 0]
    assert not masks[0, 2]

    valid, saved_signature, saved_value = _update_base_memory(
        states,
        in_base,
        emitted_signature,
        emitted_value,
        valid,
        saved_signature,
        saved_value,
        jnp.asarray([True]),
    )
    assert not valid[0]
    assert not saved_signature.any()
    assert not saved_value.any()


def test_obstacle_blocks_tarmac_reporting_and_base_memory_replay():
    level = load_level()
    cfg = replace(level.env, num_obstacles=1, comm_radius=10.0, comm_radius_base=10.0)
    reset, _, _, _ = make_env_fns(level.building, level.env)
    state = reset(jax.random.PRNGKey(1))._replace(
        pos=jnp.asarray([
            [2.0, 0.0, 1.0],
            [8.0, 0.0, 1.0],
            [2.0, 3.0, 1.0],
            [2.0, 6.0, 1.0],
        ]),
        base_pos=jnp.asarray([0.0, 0.0, 1.0]),
        active=jnp.ones(4, dtype=jnp.bool_),
        target_known=jnp.asarray([False, True, False, False]),
        obstacle_min=jnp.asarray([[4.0, -1.0, 0.0]]),
        obstacle_max=jnp.asarray([[6.0, 1.0, 2.0]]),
    )
    states = jax.tree_util.tree_map(lambda item: item[None], state)
    valid = jnp.asarray([False])
    saved_signature = jnp.zeros((1, 4))
    saved_value = jnp.zeros((1, 3))

    communication = jax.jit(
        lambda batched, base_valid: _communication_inputs(
            batched, cfg, base_valid, saved_signature, saved_value
        )
    )
    comm_mask, in_base, memory_mask, _, _ = communication(states, valid)
    assert not comm_mask[0, 0, 1]  # In range, but the cuboid blocks TarMAC.
    assert comm_mask[0, 0, 2]  # An unobstructed in-range peer remains reachable.
    assert not in_base[0, 1]  # The informed drone cannot report through the cuboid.

    emitted_signature = jnp.arange(16, dtype=jnp.float32).reshape(1, 4, 4)
    emitted_value = jnp.arange(12, dtype=jnp.float32).reshape(1, 4, 3)
    valid, _, _ = _update_base_memory(
        states, in_base, emitted_signature, emitted_value,
        valid, saved_signature, saved_value, jnp.asarray([False]),
    )
    assert not valid[0]

    valid = jnp.asarray([True])
    state = state._replace(target_known=jnp.zeros(4, dtype=jnp.bool_))
    states = jax.tree_util.tree_map(lambda item: item[None], state)
    _, _, memory_mask, _, _ = communication(states, valid)
    assert not memory_mask[0, 1]  # Base memory cannot replay through the cuboid.
    assert memory_mask[0, 2]  # An unobstructed drone can receive base memory.
