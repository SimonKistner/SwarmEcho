import unittest

import numpy as np

from swarmecho.training.mappo_buffer import MAPPORolloutBuffer


class ComputeGAETest(unittest.TestCase):
    def _buffer(self, *, gae_lambda: float) -> MAPPORolloutBuffer:
        buffer = MAPPORolloutBuffer(
            num_steps=3,
            num_envs=1,
            num_agents=1,
            obs_dim=1,
            act_dim=1,
            gamma=1.0,
            gae_lambda=gae_lambda,
        )
        buffer._rewards[:, 0, 0] = [0.0, 1.0, 0.0]
        buffer._values[:, 0, 0] = [10.0, 20.0, 30.0]
        buffer._dones[:, 0] = [0.0, 1.0, 0.0]
        return buffer

    def test_td_residual_uses_current_transition_done(self):
        buffer = self._buffer(gae_lambda=0.0)

        advantages, returns = buffer.compute_gae(
            last_value=np.array([[40.0]], dtype=np.float32),
            last_done=np.array([0.0], dtype=np.float32),
        )

        np.testing.assert_allclose(advantages[:, 0, 0], [10.0, -19.0, 10.0])
        np.testing.assert_allclose(returns[:, 0, 0], [20.0, 1.0, 40.0])

    def test_advantage_does_not_cross_autoreset_boundary(self):
        buffer = self._buffer(gae_lambda=1.0)

        advantages, _ = buffer.compute_gae(
            last_value=np.array([[40.0]], dtype=np.float32),
            last_done=np.array([0.0], dtype=np.float32),
        )

        # The terminal transition at t=1 cannot inherit the t=2 advantage from
        # the new episode.  The preceding transition may still inherit t=1.
        np.testing.assert_allclose(advantages[:, 0, 0], [-9.0, -19.0, 10.0])

    def test_final_transition_uses_its_stored_done_mask(self):
        buffer = self._buffer(gae_lambda=1.0)
        buffer._dones[:, 0] = [0.0, 0.0, 1.0]

        advantages, returns = buffer.compute_gae(
            last_value=np.array([[999.0]], dtype=np.float32),
            # The caller's legacy argument must not override the transition
            # done stored alongside the final reward.
            last_done=np.array([0.0], dtype=np.float32),
        )

        np.testing.assert_allclose(advantages[:, 0, 0], [-9.0, -19.0, -30.0])
        np.testing.assert_allclose(returns[:, 0, 0], [1.0, 1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
