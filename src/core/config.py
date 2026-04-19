"""
swarmecho/config.py
====================
Configuration management for 

Loads src/curriculum_config/default.yaml via OmegaConf, optionally merges CLI overrides,
and exposes a clean typed interface via Python dataclasses.

Usage
-----
From Python:
    from core.config import load_config
    cfg = load_config()                # loads default.yaml
    cfg = load_config("src/curriculum_config/my_experiment.yaml")  # custom yaml
    print(cfg.env.num_agents)

From CLI (pass as extra args to any script):
    uv run python training/train.py env.num_agents=16 training.lr=1e-4
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

_SRC_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _SRC_ROOT / "curriculum_config" / "base_params.yaml"
MAP_DIR = _SRC_ROOT / "curriculum_config" / "maps"


# ---------------------------------------------------------------------------
# Typed dataclasses — mirrors src/curriculum_config/default.yaml structure.
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
    wall_restitution: float = 0.2
    max_force: float = 5.0
    max_steps: int = 500
    grid_cell_size: float = 1.0  # meter resolution per grid cell
    max_grid_width:  int = 500   # maximum width in cells for static JAX sizing
    max_grid_height: int = 150   # maximum height in cells for static JAX sizing
    base_x: float = 5.0
    base_y: float = 50.0
    target_x: float = 95.0
    target_y: float = 50.0
    k_max_neighbors: int = 3  # kept for any legacy references; no longer in obs
    radar_bins: int = 8        # B — angular radar slices
    spawn_delay: int = 10      # steps between drone activations (0 = all at once)
    map_names: list[str] = field(default_factory=list) # if set, samples from these maps
    num_targets: int = 1       # 0 = exploration focus, 1 = find target goal
    num_bases: int = 1         # 0 = exploration focus, 1 = tethered relay goal
    exploration_sampling_radius: float = 20.0 # meter offset for local coverage grid sampling


@dataclass
class RewardConfig:
    exploration_bonus: float = 0.5
    target_found_bonus: float = 50.0
    max_gap_penalty: float = 2.0       # absolute penalty when gap is at its maximum
    collision_penalty: float = 0.1    # absolute penalty per agent per step in collision
    success_bonus: float = 100.0


@dataclass
class TrainingConfig:
    seed: int = 42
    use_mappo: bool = True
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
    checkpoint_path: Optional[str] = None  # if set, resumes training from this path


@dataclass
class NetworkConfig:
    hidden_dim: int = 256
    num_layers: int = 3


@dataclass
class LoggingConfig:
    project: str = "SwarmEcho"
    run_name: str = "ppo_run"
    log_dir: str = "outputs"
    wandb_mode: str = "disabled"   # "online", "offline", or "disabled"
    wandb_project: str = "SwarmEcho"
    wandb_entity: Optional[str] = None
    log_freq: int = 10
    video_freq: int = 50
    save_model: bool = True
    checkpoint_dir: str = "outputs/checkpoints"
    suppress_xla_warnings: bool = True


@dataclass
class VisualizeConfig:
    renderer: str = "fast"       # "fast" for OpenCV, "slow" for Matplotlib
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

    Observation layout per agent (see observations.py for full docs):

        Self state (9)
        ├─ vel_i / v_max                         (2)
        ├─ (base_pos - pos_i) / max_dim          (2)
        ├─ is_connected_to_base                  (1)   multi-hop graph
        ├─ is_connected_to_target                (1)   multi-hop graph
        ├─ target_known_flag                     (1)   explicit 0/1 flag
        └─ (target_pos - pos_i) / max_dim × mask (2)   masked until target_known

        Radar (B × 6)  — per angular bin:
        ├─ inv_dist_wall                          (1)   ray-cast, norm by vis_r
        ├─ inv_dist_drone                         (1)   any active drone, norm by comm_r
        ├─ inv_dist_target_conn_drone             (1)   target-chain drones, norm by comm_r
        ├─ inv_dist_base_conn_drone               (1)   base-chain drones, norm by comm_r
        ├─ inv_dist_base_station                  (1)   fixed beacon, norm by comm_r
        └─ inv_dist_target                        (1)   target point, norm by vis_r (0 if unseen)

        Local Coverage (8) — per fixed direction (N, NE, E, SE, S, SW, W, NW):
        └─ is_cell_covered                        (1)   0/1 flag at sampling distance

    Total: 9 + 8 + B * 6
    """
    B = cfg.env.radar_bins
    return 9 + 8 + B * 6


def compute_action_dim(_cfg: DictConfig) -> int:
    """Each agent outputs a 2D continuous force vector."""
    return 2


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(
    config_path: str | Path | None = None,
    cli_overrides: bool = True,
    overrides: list[str] | None = None,
) -> DictConfig:
    """
    Load the SwarmEcho configuration.

    Parameters
    ----------
    config_path:
        Path to a YAML config file. Defaults to src/curriculum_config/default.yaml.
    cli_overrides:
        If True, merge any extra CLI arguments (key=value pairs) on top of
        the loaded config. Useful when scripts are called with overrides:
            uv run python training/train.py training.lr=1e-4
    overrides:
        Optional list of string arguments to parse instead of sys.argv.

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

    # Start with the structured base from the Dataclasses (ensures all fields exist)
    base_cfg = OmegaConf.structured(SwarmEchoConfig)
    
    # Load and merge YAML
    yaml_cfg = OmegaConf.load(path)
    cfg = OmegaConf.merge(base_cfg, yaml_cfg)

    if cli_overrides:
        cli_cfg = OmegaConf.from_cli(overrides)
        
        # Check for level-based curriculum loading
        # uv run python training/train.py level=01
        level = cli_cfg.get("level")
        if level:
            # Search for a file starting with the level ID (e.g., 00)
            level_dir = _SRC_ROOT / "curriculum_config" / "levels"
            matches = list(level_dir.glob(f"{level}*.yaml")) + list(level_dir.glob(f"{level}*.yml"))
            if not matches:
                raise FileNotFoundError(f"Level config starting with '{level}' not found in {level_dir}")
            level_path = matches[0]
            level_cfg = OmegaConf.load(level_path)
            cfg = OmegaConf.merge(cfg, level_cfg)

        # Merge standard CLI key=value overrides
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
    assert cfg.env.radar_bins >= 4, "radar_bins must be >= 4."
    assert cfg.env.spawn_delay >= 0, "spawn_delay must be >= 0."
    assert 0.0 <= cfg.env.wall_restitution <= 1.0, "wall_restitution must be [0, 1]."
    assert cfg.training.num_envs > 0
    assert cfg.training.num_steps > 0
    assert 0 < cfg.training.gamma <= 1.0
    assert 0 < cfg.training.gae_lambda <= 1.0


if __name__ == "__main__":
    # Quick self-test: load and print the resolved config.
    cfg = load_config(cli_overrides=False)
    validate_config(cfg)
    print(OmegaConf.to_yaml(cfg))
    print(f"\nObs dim  : {compute_obs_dim(cfg)}")
    print(f"Action dim: {compute_action_dim(cfg)}")


