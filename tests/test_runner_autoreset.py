import unittest

import jax.numpy as jnp
import numpy as np

from swarmecho.env.state import (
    CommunicationState,
    EnvState,
    ExplorationState,
    PhysicsState,
    RelayTaskState,
    StepDiagnostics,
)
from swarmecho.training.runner import _make_autoreset_step


def _state(*, step: int = 0, held_steps: int = 0, num_agents: int = 3) -> EnvState:
    return EnvState(
        physics=PhysicsState(
            pos=jnp.zeros((num_agents, 2), dtype=jnp.float32),
            vel=jnp.zeros((num_agents, 2), dtype=jnp.float32),
            base_pos=jnp.zeros((2,), dtype=jnp.float32),
            target_pos=jnp.ones((2,), dtype=jnp.float32),
            step=jnp.int32(step),
            key=jnp.array([0, 1], dtype=jnp.uint32),
            active=jnp.ones((num_agents,), dtype=jnp.bool_),
            box_width=jnp.float32(10.0),
            box_height=jnp.float32(10.0),
        ),
        communication=CommunicationState(
            target_known=jnp.zeros((num_agents,), dtype=jnp.bool_),
            base_target_known=jnp.bool_(False),
            is_conn_base=jnp.zeros((num_agents,), dtype=jnp.bool_),
            is_conn_target=jnp.zeros((num_agents,), dtype=jnp.bool_),
            directly_sees_target=jnp.zeros((num_agents,), dtype=jnp.bool_),
            adj_matrix=jnp.zeros((num_agents + 1, num_agents + 1), dtype=jnp.bool_),
        ),
        exploration=ExplorationState(
            coverage_grid=jnp.zeros((1, 1), dtype=jnp.bool_),
        ),
        relay=RelayTaskState(chain_held_steps=jnp.int32(held_steps)),
        diagnostics=StepDiagnostics(
            collides=jnp.zeros((num_agents,), dtype=jnp.bool_),
            last_cov_delta=jnp.zeros((num_agents,), dtype=jnp.int32),
        ),
    )


class AutoresetRewardTest(unittest.TestCase):
    def test_success_bonus_does_not_recompute_reward(self):
        reward_calls = []
        initial_state = _state()

        def env_step(state, _actions):
            return state.replace(
                physics=state.physics.replace(step=state.physics.step + 1),
            )

        def reset(_key):
            return initial_state

        def reward(_old_state, _new_state, is_done):
            reward_calls.append(bool(is_done))
            return jnp.full((3,), 2.0, dtype=jnp.float32), {
                "fully_connected": jnp.float32(1.0),
                "global_target_found": jnp.float32(0.0),
            }

        step = _make_autoreset_step(
            env_step,
            reset,
            reward,
            max_steps=10,
            hold_chain_for=0,
            success_bonus=12.0,
        )

        _next_state, rewards, done, _info = step(
            initial_state,
            jnp.zeros((3, 2), dtype=jnp.float32),
        )

        self.assertEqual(reward_calls, [False])
        self.assertTrue(bool(done))
        np.testing.assert_allclose(rewards, [6.0, 6.0, 6.0])


if __name__ == "__main__":
    unittest.main()
