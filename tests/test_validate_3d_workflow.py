import math

from swarmecho.training.validate_3d_workflow import validate_3d_update


def test_real_3d_rollout_completes_mappo_gradient_update():
    stats = validate_3d_update(
        num_envs=2,
        num_steps=2,
        hidden_dim=16,
        recurrent=True,
    )
    assert stats
    assert all(math.isfinite(value) for value in stats.values())
