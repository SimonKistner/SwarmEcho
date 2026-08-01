import gc
import inspect

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from omegaconf import OmegaConf

from swarmecho.env.critic_state import (
    make_privileged_critic_state_fn,
    privileged_critic_dim,
)
from swarmecho.env.state import (
    CommunicationState,
    EnvState,
    ExplorationState,
    PhysicsState,
    RelayTaskState,
    StepDiagnostics,
)
from swarmecho.models.critic import (
    RecurrentAgentCentricCritic,
    RecurrentPrivilegedAgentCentricCritic,
)
from swarmecho.models.mappo import MAPPOModel


@pytest.fixture(autouse=True)
def _release_jax_compilations():
    yield
    jax.clear_caches()
    gc.collect()


def _state() -> EnvState:
    return EnvState(
        physics=PhysicsState(
            pos=jnp.array([[1.0, 1.0], [2.0, 2.0]]),
            vel=jnp.zeros((2, 2)),
            base_pos=jnp.array([0.0, 0.0]),
            target_pos=jnp.array([4.0, 4.0]),
            step=jnp.int32(5),
            key=jnp.zeros((2,), dtype=jnp.uint32),
            active=jnp.ones((2,), dtype=bool),
            box_width=jnp.float32(4),
            box_height=jnp.float32(4),
        ),
        communication=CommunicationState(
            target_known=jnp.array([True, False]),
            base_target_known=jnp.bool_(False),
            is_conn_base=jnp.array([True, False]),
            is_conn_target=jnp.array([False, True]),
            directly_sees_target=jnp.array([False, True]),
            adj_matrix=jnp.array([
                [False, True, True], [True, False, False], [True, False, False]
            ]),
        ),
        exploration=ExplorationState(coverage_grid=jnp.eye(4, dtype=bool)),
        relay=RelayTaskState(chain_held_steps=jnp.int32(2)),
        diagnostics=StepDiagnostics(
            collides=jnp.array([False, True]), last_cov_delta=jnp.array([1, 0])
        ),
    )


def test_privileged_state_contains_full_map_and_graph_rows():
    cfg = OmegaConf.create({
        "env": {
            "num_agents": 2, "max_steps": 10, "hold_chain_for": 4,
            "max_speed": 2.0, "cell_size": 1.0,
        },
        "network": {"critic_map_resolution": 64},
    })
    fn, width, wall_map = make_privileged_critic_state_fn(
        cfg, 4.0, 4.0, jnp.zeros((4, 4), dtype=bool)
    )
    features, coverage_map = fn(_state())

    assert features.shape == (2, privileged_critic_dim(2)) == (2, width)
    assert wall_map.shape == coverage_map.shape == (4, 4)
    assert jnp.all(jnp.isfinite(features))
    assert jnp.array_equal(features[:, 17:19], jnp.array([[0, 1], [1, 0]]))


def test_legacy_critic_remains_the_default():
    parameter = inspect.signature(MAPPOModel.__init__).parameters["critic_type"]
    assert parameter.default == "observation"
    assert RecurrentAgentCentricCritic is not RecurrentPrivilegedAgentCentricCritic


def test_privileged_cnn_outputs_per_agent_values():
    model = MAPPOModel(
        obs_dim=5, act_dim=2, num_agents=2, hidden_dim=16, num_layers=1,
        actor_num_layers=1, critic_memory=True, critic_type="privileged",
        critic_input_dim=22, critic_wall_map=jnp.zeros((16, 16)),
        rngs=nnx.Rngs(1),
    )
    tokens = jnp.zeros((2, 22)).at[:, :2].set(
        jnp.array([[0.25, 0.5], [0.75, 0.5]])
    )
    tokens = tokens.at[0, 9].set(1.0)
    semantic = model.critic._semantic_image(tokens, jnp.zeros((16, 16)))
    _, values = model.get_value_recurrent(
        jnp.zeros((2, 5)), model.initial_critic_hidden(),
        jnp.zeros((2,), dtype=bool), critic_obs=tokens,
        critic_map=jnp.eye(16),
    )
    assert values.shape == (2,)
    assert jnp.all(jnp.isfinite(values))
    assert type(model.critic) is RecurrentPrivilegedAgentCentricCritic
    assert semantic.shape == (16, 16, 5)
    assert jnp.sum(semantic[..., 2]) == 1.0
