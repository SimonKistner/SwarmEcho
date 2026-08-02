import jax
import jax.numpy as jnp

from swarmecho.core.config import load_level_3d
from swarmecho.env.baseline3d import make_baseline_3d_fns
from swarmecho.training.train3d import _communication_inputs, _update_base_memory


def test_base_stores_first_reporter_replays_and_clears_on_reset():
    level = load_level_3d()
    reset, _, _, _ = make_baseline_3d_fns(level.building, level.env)
    state = reset(jax.random.PRNGKey(0))
    state = state._replace(target_known=state.target_known.at[2].set(True))
    states = jax.tree_util.tree_map(lambda item: item[None], state)
    valid = jnp.asarray([False])
    saved_signature = jnp.zeros((1, 4))
    saved_value = jnp.zeros((1, 3))
    emitted_signature = jnp.arange(20, dtype=jnp.float32).reshape(1, 5, 4)
    emitted_value = jnp.arange(15, dtype=jnp.float32).reshape(1, 5, 3)

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
