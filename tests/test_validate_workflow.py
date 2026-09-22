import math

from swarmecho.training.validate_workflow import validate_update


def test_real_rollout_completes_mappo_gradient_update():
    stats = validate_update(
        num_envs=2,
        num_steps=2,
        hidden_dim=16,
        recurrent=True,
    )
    assert stats
    assert all(math.isfinite(value) for value in stats.values())
