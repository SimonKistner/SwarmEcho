"""Local, offline real-environment smoke; intentionally outside normal pytest.

Run: .venv/bin/python ablations/path-probe/runtime_smoke.py
Uses two lanes, a tiny network, two updates and the existing authored G01 map.
No W&B or cluster access. Temporary artifacts are removed on success or failure.
"""
import json
from pathlib import Path
import tempfile

from swarmecho.core.config import load_level
from swarmecho.training.train import train


def main():
    level = load_level("G01_static", [
        "training.num_envs=2", "training.num_steps=4", "training.total_timesteps=16",
        "training.num_minibatches=1", "training.num_epochs=1",
        "training.optimizer_microbatches=2", "training.rollout_batch_size=1",
        "training.max_training_seconds=null", "training.max_wall_seconds=null",
        "training.adaptive_target_spawn.enabled=false",
        "training.adaptive_target_spawn.curriculum_start_agents=null",
        "training.geo_probe_auxiliary.enabled=true",
        "training.geo_probe_auxiliary.head.hidden_sizes=[8]",
        "training.geo_probe_auxiliary.sampling.snapshots_per_agent=2",
        "training.geo_probe_auxiliary.schedule.warmup_updates=0",
        "training.geo_probe_auxiliary.schedule.ramp_updates=0",
        "env.max_steps=4", "network.hidden_dim=8", "network.num_layers=1",
        "network.actor_num_layers=1", "network.tarmac_sig_dim=2", "network.tarmac_val_dim=3",
        "inspection_evaluation.early_exit=false", "inspection_evaluation.target_sampling=independent",
        "inspection_evaluation.save_model=false", "inspection_evaluation.eval_freq=1000000",
        "inspection_evaluation.eval_parallel_envs=2",
        "inspection_evaluation.eval_offset=1000000", "inspection_evaluation.eval_video=false",
        "logging.wandb_mode=disabled", "logging.terminal_logging_frequency=1",
        "validation.num_episodes=2", "validation.parallel_envs=1",
        "validation.val_on_finalization=true", "validation.save_best_checkpoint=false",
    ])
    with tempfile.TemporaryDirectory(prefix="swarmecho_geo_auxiliary_") as directory:
        _, stats = train(level, output_dir=directory)
        history = [json.loads(row) for row in (Path(directory)/"training_history.jsonl").read_text().splitlines()]
        assert [row["update"] for row in history] == [1,2]
        assert stats["global_step"] == 16
        validation = list((Path(directory)/"validation").glob("results_*.json"))
        assert validation
        assert (Path(directory)/"geo_probe_auxiliary/labels.json").is_file()
        print("PASS: real rollout, chunking, auxiliary replay, PPO accumulation, validation and finalization")


if __name__ == "__main__":
    main()
