import json
from dataclasses import replace

from swarmecho.core.config import load_level
from swarmecho.training.checkpoints import restore_model_checkpoint
from swarmecho.training.train import build_model, evaluate_model, train
from swarmecho.visualize.replay import load_replay


def test_tiny_3d_training_creates_checkpoint_metrics_and_replay(tmp_path):
    level = load_level()
    level = replace(
        level,
        training=replace(
            level.training,
            total_timesteps=4,
            num_envs=2,
            num_steps=2,
            num_epochs=1,
            num_minibatches=1,
        ),
        network=replace(level.network, hidden_dim=16, tarmac_sig_dim=4, tarmac_val_dim=3),
        env=replace(level.env, max_steps=2),
        evaluation=replace(level.evaluation, eval_parallel_envs=1),
        logging=replace(level.logging, wandb_mode="disabled"),
    )
    checkpoint, stats = train(level, output_dir=tmp_path)
    assert checkpoint.exists()
    assert stats
    assert json.loads((tmp_path / "metrics.json").read_text())["total_loss"] == stats["total_loss"]
    replays = list(
        (tmp_path / "artifacts/eval/ckpt_u000001_s00004").glob(
            "eval_*/replays/*.json"
        )
    )
    assert len(replays) == 1
    replay = replays[0]
    metadata, arrays = load_replay(replay)
    assert metadata["artifact_scope"] == "eval"
    assert metadata["checkpoint"] == str(checkpoint)
    assert metadata["frames"] > 1
    assert arrays["position"].shape[-1] == 3

    restored = build_model(level, hidden_dim=16)
    restore_model_checkpoint(restored, checkpoint)
    states, rewards = evaluate_model(restored, level, max_steps=1)
    assert len(states) == 2
    assert rewards.shape == (2, level.env.num_agents, 1)
