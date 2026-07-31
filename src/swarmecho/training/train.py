"""
training/train.py
================
CLI entry point for SwarmEcho MAPPO training.

Usage
-----
    # Python defaults
    uv run swarmecho-train

    # Override any config value via CLI
    uv run swarmecho-train training.num_envs=512 env.num_agents=6

    # Quick smoke test (fast, small)
    uv run swarmecho-train training.total_timesteps=20000 training.num_envs=32 logging.wandb_mode=disabled
"""

from swarmecho.core.config import load_config, validate_config
from swarmecho.training.runner import train


def main() -> None:
    cfg = load_config(cli_overrides=True)
    validate_config(cfg)
    train(cfg)

if __name__ == "__main__":
    main()


