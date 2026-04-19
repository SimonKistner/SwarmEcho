"""
training/train.py
================
CLI entry point for SwarmEcho IPPO training.

Usage
-----
    # Default config (src/curriculum_config/default.yaml)
    uv run python training/train.py

    # Override any config value via CLI
    uv run python training/train.py training.num_envs=512 env.num_agents=6

    # Quick smoke test (fast, small)
    uv run python training/train.py training.total_timesteps=20000 training.num_envs=32 logging.wandb_mode=disabled
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import load_config, validate_config
from training.runner import train

if __name__ == "__main__":
    cfg = load_config(cli_overrides=True)
    validate_config(cfg)
    train(cfg)


