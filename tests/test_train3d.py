import json
from dataclasses import replace

from swarmecho.core.config3d import load_level_3d
from swarmecho.training.checkpoints import restore_model_checkpoint
from swarmecho.training.train3d import build_model_3d, evaluate_model_3d, train_3d
from swarmecho.visualize.replay3d import load_replay


def test_tiny_3d_training_creates_checkpoint_metrics_and_replay(tmp_path):
    level = load_level_3d()
    level = replace(
        level,
        training=replace(
            level.training,
            updates=1,
            num_envs=2,
            num_steps=2,
            num_epochs=1,
            num_minibatches=1,
            hidden_dim=16,
        ),
    )
    checkpoint, stats = train_3d(level, output_dir=tmp_path)
    assert checkpoint.exists()
    assert stats
    assert json.loads((tmp_path / "metrics.json").read_text())["total_loss"] == stats["total_loss"]
    metadata, arrays = load_replay(tmp_path / "replays/latest.json")
    assert metadata["frames"] > 1
    assert arrays["position"].shape[-1] == 3

    restored = build_model_3d(level, hidden_dim=16)
    restore_model_checkpoint(restored, checkpoint)
    states, rewards = evaluate_model_3d(restored, level, max_steps=1)
    assert len(states) == 2
    assert rewards.shape == (2, level.env.num_agents, 1)
