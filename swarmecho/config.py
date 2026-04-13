"""
swarmecho/config.py
====================
Configuration management for SwarmEcho.

Loads configs/default.yaml via OmegaConf, optionally merges CLI overrides,
and exposes a clean typed interface via Python dataclasses.

Usage
-----
From Python:
    from swarmecho.config import load_config
    cfg = load_config()                # loads default.yaml
    cfg = load_config("configs/my_experiment.yaml")  # custom yaml
    print(cfg.env.num_agents)

From CLI (pass as extra args to any script):
    uv run python scripts/train.py env.num_agents=16 training.lr=1e-4
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf

# ---------------------------------------------------------------------------
# Default config path
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "default.yaml"


# ---------------------------------------------------------------------------
# Typed dataclasses — mirrors configs/default.yaml structure.
# These are used purely for IDE type-checking and documentation.
# The canonical config at runtime is always the OmegaConf DictConfig object.
# ---------------------------------------------------------------------------

@dataclass
class EnvConfig:
    num_agents: int = 8
    box_width: float = 100.0
    box_height: float = 100.0
    dt: float = 0.1
    visual_radius: float = 15.0
    comm_radius: float = 25.0
    max_speed: float = 10.0
    drag: float = 0.85
    max_force: float = 5.0
    max_steps: int = 500
    grid_resolution: int = 50
    base_x: float = 5.0
    base_y: float = 50.0
    target_x: float = 95.0
    target_y: float = 50.0
    k_max_neighbors: int = 4


@dataclass
class RewardConfig:
    time_penalty: float = -0.01
    exploration_bonus: float = 0.5
    chain_gap_penalty: float = 0.1
    success_bonus: float = 100.0


@dataclass
class TrainingConfig:
    seed: int = 42
    num_envs: int = 1024
    num_steps: int = 128
    num_epochs: int = 4
    num_minibatches: int = 8
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    total_timesteps: int = 50_000_000


@dataclass
class NetworkConfig:
    hidden_dim: int = 256
    num_layers: int = 3


@dataclass
class LoggingConfig:
    project: str = "SwarmEcho"
    run_name: str = "ppo_run"
    log_freq: int = 10
    video_freq: int = 50
    save_model: bool = True
    checkpoint_dir: str = "outputs/checkpoints"


@dataclass
class VisualizeConfig:
    dpi: int = 150               # 150 gives sharp high-def text; pixel sizes scale automatically
    comm_color: str = "match_drone"
    comm_fill_alpha: float = 0.15
    comm_edge_alpha: float = 0.70
    vis_color: str = "match_drone"
    vis_fill_alpha: float = 0.30
    vis_edge_alpha: float = 0.85


@dataclass
class SwarmEchoConfig:
    """Root config. All sub-configs accessible as attributes."""
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    visualize: VisualizeConfig = field(default_factory=VisualizeConfig)


# ---------------------------------------------------------------------------
# Derived / computed values
# ---------------------------------------------------------------------------

def compute_obs_dim(cfg: DictConfig) -> int:
    """
    Compute the total observation vector dimension from config.

    Observation layout (per agent):
        vel[i]               : 2
        rel_base_pos         : 2
        self_seen_target     : 1
        self_not_seen_target : 1
        comm_seen_target     : 1
        comm_not_seen_target : 1
        rel_target_pos       : 2   (zero if self_seen_target == 0)
        neighbor_data        : k_max_neighbors * 4
            (rel_pos_x, rel_pos_y, their_self_seen, their_comm_seen)
        agent_id_onehot      : num_agents
    """
    k = cfg.env.k_max_neighbors
    n = cfg.env.num_agents
    return 2 + 2 + 1 + 1 + 1 + 1 + 2 + k * 4 + n


def compute_action_dim(_cfg: DictConfig) -> int:
    """Each agent outputs a 2D continuous force vector."""
    return 2


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(
    config_path: str | Path | None = None,
    cli_overrides: bool = True,
) -> DictConfig:
    """
    Load the SwarmEcho configuration.

    Parameters
    ----------
    config_path:
        Path to a YAML config file. Defaults to configs/default.yaml.
    cli_overrides:
        If True, merge any extra CLI arguments (key=value pairs) on top of
        the loaded config. Useful when scripts are called with overrides:
            uv run python scripts/train.py training.lr=1e-4

    Returns
    -------
    OmegaConf DictConfig — use dot notation: cfg.env.num_agents
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"Expected default at: {_DEFAULT_CONFIG}"
        )

    cfg = OmegaConf.load(path)

    if cli_overrides:
        # Merge CLI key=value overrides (ignores unknown keys gracefully)
        cli_cfg = OmegaConf.from_cli()
        cfg = OmegaConf.merge(cfg, cli_cfg)

    # Make read-only at runtime to prevent accidental mutation
    OmegaConf.set_readonly(cfg, True)

    return cfg


# ---------------------------------------------------------------------------
# Quick validation on load
# ---------------------------------------------------------------------------

def validate_config(cfg: DictConfig) -> None:
    """
    Run basic sanity checks on the config.
    Raises ValueError with a clear message on any issue.
    """
    assert cfg.env.num_agents >= 2, "Need at least 2 agents for a chain."
    assert cfg.env.comm_radius > 0, "comm_radius must be positive."
    assert cfg.env.visual_radius > 0, "visual_radius must be positive."
    assert cfg.env.dt > 0, "dt must be positive."
    assert cfg.training.num_envs > 0
    assert cfg.training.num_steps > 0
    assert 0 < cfg.training.gamma <= 1.0
    assert 0 < cfg.training.gae_lambda <= 1.0
    assert cfg.env.k_max_neighbors >= 1, "k_max_neighbors must be >= 1."


if __name__ == "__main__":
    # Quick self-test: load and print the resolved config.
    cfg = load_config(cli_overrides=False)
    validate_config(cfg)
    print(OmegaConf.to_yaml(cfg))
    print(f"\nObs dim  : {compute_obs_dim(cfg)}")
    print(f"Action dim: {compute_action_dim(cfg)}")
