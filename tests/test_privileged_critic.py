import jax.numpy as jnp
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


def test_privileged_state_is_compact_and_contains_graph_rows():
    cfg = OmegaConf.create({"env": {
        "num_agents": 2, "max_steps": 10, "hold_chain_for": 4,
        "max_speed": 2.0,
    }})
    fn, width = make_privileged_critic_state_fn(cfg, 4.0, 4.0)
    features = fn(_state())

    assert features.shape == (2, privileged_critic_dim(2)) == (2, width)
    assert width == 27
    assert jnp.all(jnp.isfinite(features))
    assert jnp.array_equal(features[:, 17:19], jnp.array([[0, 1], [1, 0]]))


def test_critic_selection_leaves_legacy_class_unchanged():
    common = dict(
        obs_dim=5, act_dim=2, num_agents=2, hidden_dim=8, num_layers=1,
        actor_num_layers=1, actor_memory=False, critic_memory=True,
    )
    legacy = MAPPOModel(**common, rngs=nnx.Rngs(0))
    privileged = MAPPOModel(
        **common, critic_type="privileged", critic_input_dim=36, rngs=nnx.Rngs(0)
    )

    assert type(legacy.critic) is RecurrentAgentCentricCritic
    assert type(privileged.critic) is RecurrentPrivilegedAgentCentricCritic
