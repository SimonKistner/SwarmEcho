"""
swarmecho/config.py
====================
Configuration management for 

Loads SwarmEcho configuration via OmegaConf, optionally merges mission-specific YAMLs and CLI overrides.
Exposes a clean typed interface via Python dataclasses.

Usage
-----
From Python:
    from core.config import load_config
    cfg = load_config()                        # loads Python defaults
    cfg = load_config("levels/01_warehouse.yaml") # loads warehouse mission
    print(cfg.env.num_agents)

From CLI:
    uv run python training/train.py level=01 env.num_agents=16
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
MAP_DIR = _SRC_ROOT / "curriculum_config" / "maps"


# ---------------------------------------------------------------------------
# Typed dataclasses — mirrors src/curriculum_config/default.yaml structure.
# These are used purely for IDE type-checking and documentation.
# The canonical config at runtime is always the OmegaConf DictConfig object.
# ---------------------------------------------------------------------------

@dataclass
class EnvConfig:
    num_agents: int = 8
    box_width: Optional[float] = None
    box_height: Optional[float] = None
    dt: float = 0.1
    visual_radius: float = 5.0
    comm_radius: float = 10.0
    comm_radius_base: float = 15.0  # comm radius for base-station first hop (defaults to comm_radius)
    max_speed: float = 10.0
    drag: float = 0.85
    wall_restitution: float = 0.2
    max_force: float = 50.0
    max_steps: int = 1000
    radar_bins: int = 8        # B — angular radar slices
    spawn_delay: int = 0      # steps between drone activations (0 = all at once)
    map_names: list[str] = field(default_factory=list) # if set, samples from these maps
    num_targets: int = 1       # 0 = exploration focus, 1 = find target goal
    num_bases: int = 1         # 0 = exploration focus, 1 = tethered relay goal
    exploration_sampling_radius: float = 6.0 # meter offset for local coverage grid sampling
    # --- Spawn overrides (B-series curriculum) ---
    use_random_base_spawn: bool = True    # if False, base always spawns at map centre
    use_random_drone_spawn: bool = True   # if False, drones spawn stacked at base_pos
    target_spawn_method: str = "map_defined"  # "map_defined", "ring", "outside_base"
    target_spawn_radius: float = 0.0      # if > 0, target spawns in a circle of this max radius around base (for "ring")
    target_spawn_radius_min: float = 0.0  # if > 0, target spawns in a ring (min to max radius) (for "ring")
    target_invalid_spawn_base_radius: float = 0.0 # if > 0, target cannot spawn within this radius of the base (for "outside_base")
    precover_base_comm: bool = False              # if True, cells in communication range of the base station are covered from reset
    hold_chain_for: int = 0                       # number of consecutive timesteps the chain must be held before success



@dataclass
class RewardConfig:
    # --- Local Rewards (Not divided by N) ---
    exploration_bonus: float = 0.05
    collision_penalty: float = 0.5
    proximity_penalty: float = 0.00
    finder_bonus: float = 50.0
    base_proximity_bonus: float = 0.000    # intuition drive toward base
    target_proximity_bonus: float = 0  # intuition drive toward target (if known)

    # --- Global Rewards (Divided by N) ---
    max_gap_penalty: float = 5.0       # absolute penalty when gap is at its maximum
    target_found_bonus: float = 100.0
    success_bonus: float = 500.0
    only_shortest_path_chain_reward: bool = False


@dataclass
class TrainingConfig:
    seed: int = 42
    num_envs: int = 1024
    num_steps: int = 256
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
    hidden_dim:       int = 256
    num_layers:       int = 3    # critic depth
    actor_num_layers: int = 3    # actor depth (lighter, separate)
    critic_type:      str = "agent_centric"  # "agent_centric" | "global_mean"


@dataclass
class DiversityConfig:
    enabled: bool = False
    method: str = "forward_history"  # currently: "forward_history"
    num_roles: int = 0                # 0 = one role per agent; otherwise agent i -> i % num_roles
    history_len: int = 4              # number of recent (obs, action) pairs used by predictors
    beta: float = 0.05                # intrinsic reward scale before GAE/PPO
    aux_coef: float = 1.0             # forward predictor loss weight
    l1_coef: float = 0.001            # role-adapter L1 loss weight
    reward_clip: float = 1.0          # clips normalized intrinsic reward to [-clip, clip]
    normalize_intrinsic: bool = True  # normalize intrinsic reward over valid agents each step
    adapter_scale: float = 1.0        # multiplier for role-specific actor residuals
    predictor_hidden_dim: int = 128
    predictor_num_layers: int = 2


@dataclass
class LoggingConfig:
    project: str = "SwarmEcho"
    run_name: Optional[str] = None
    use_timestamp_postfix: bool = False
    log_dir: str = "outputs"
    wandb_mode: str = "online"   # "online", "offline", or "disabled"
    wandb_project: str = "SwarmEcho"
    wandb_entity: Optional[str] = None
    log_freq: int = 10
    video_freq: int = 30
    async_video: bool = False   # If True, renders in background process; if False, blocks training to render
    num_checkpoints: int = 10   # Guaranteed number of checkpoints per run
    eval_episodes: int = 1
    save_model: bool = True
    checkpoint_dir: str = "outputs/checkpoints"
    suppress_xla_warnings: bool = True
    obs_log: bool = True


@dataclass
class VisualizeConfig:
    renderer: str = "slow"       # "fast" for OpenCV, "slow" for Matplotlib
    comm_color: str = "#03fbff"
    comm_fill_alpha: float = 0.02
    comm_edge_alpha: float = 0.50
    vis_color: str = "#03fbff"
    vis_fill_alpha: float = 0.10
    vis_edge_alpha: float = 0.50


@dataclass
class CurriculumConfig:
    success_threshold: Optional[float] = None


@dataclass
class SwarmEchoConfig:
    """Root config. All sub-configs accessible as attributes."""
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    diversity: DiversityConfig = field(default_factory=DiversityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    visualize: VisualizeConfig = field(default_factory=VisualizeConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)


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

        Local Coverage (16) — per circular direction (evenly spaced):
        ├─ is_cell_covered                        (1)   0/1 flag at sampling distance
        
        Radar (B × 4)  — per angular bin:
        ├─ inv_dist_wall                          (1)   ray-cast, norm by vis_r
        ├─ inv_dist_drone                         (1)   any active drone, norm by comm_r
        ├─ inv_dist_target_conn_drone             (1)   target-chain drones, norm by comm_r
        └─ inv_dist_base_conn_drone               (1)   base-chain drones, norm by comm_r

    Total: 9 + 16 + B * 4
    """
    B = cfg.env.radar_bins
    return 9 + 16 + B * 4


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
        Path to a YAML config file (e.g. levels/01_warehouse.yaml).
        If None, the pure Python defaults from config.py are used.
    cli_overrides:
        If True, merge any extra CLI arguments (key=value pairs) on top.
    overrides:
        Optional list of string arguments to parse instead of sys.argv.

    Returns
    -------
    OmegaConf DictConfig — use dot notation: cfg.env.num_agents
    """
    # Start with the structured base from the Dataclasses (the Source of Truth for defaults)
    cfg = OmegaConf.structured(SwarmEchoConfig)
    
    # Optional: Merge Mission YAML
    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        yaml_cfg = OmegaConf.load(path)
        cfg = OmegaConf.merge(cfg, yaml_cfg)

    if cli_overrides:
        cli_cfg = OmegaConf.from_cli(overrides)
        
        # Check for level-based curriculum loading
        # uv run python training/train.py level=A00
        # uv run python training/train.py level=B01
        level = cli_cfg.pop("level", None)
        if level is not None:
            # Treat the level as a raw string prefix (e.g. "A00", "B02", "00").
            # OmegaConf may parse purely numeric values as int — convert to str.
            # We do NOT zero-pad here so that alpha-prefixed levels like "A00" are
            # passed through verbatim.
            level_str = str(level)
            # For purely numeric tokens, preserve 2-digit zero-padding
            if level_str.isdigit():
                level_str = level_str.zfill(2)
            # Search for a file whose name starts with the level prefix followed by '_'
            level_dir = _SRC_ROOT / "curriculum_config" / "levels"
            matches = (
                list(level_dir.glob(f"{level_str}_*.yaml"))
                + list(level_dir.glob(f"{level_str}_*.yml"))
                # Fallback: any file starting with level_str
                + list(level_dir.glob(f"{level_str}*.yaml"))
                + list(level_dir.glob(f"{level_str}*.yml"))
            )
            # Deduplicate while preserving order
            seen = set()
            matches = [m for m in matches if not (m in seen or seen.add(m))]
            if not matches:
                raise FileNotFoundError(f"Level config starting with '{level_str}' not found in {level_dir}")
            level_path = matches[0]
            print(f"  Level Config     : {level_path.name}")
            level_cfg = OmegaConf.load(level_path)
            
            # Remove __meta__ block if it exists (it's for human-readability, not the dataclass)
            if hasattr(level_cfg, "__meta__"):
                level_cfg.pop("__meta__", None)
                
            cfg = OmegaConf.merge(cfg, level_cfg)

        # Merge standard CLI key=value overrides
        cfg = OmegaConf.merge(cfg, cli_cfg)

    if cfg.logging.run_name:
        print(f"  Run Name         : {cfg.logging.run_name}")

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
    assert cfg.env.num_agents >= 1, "Need at least 1 agent."
    assert cfg.env.comm_radius > 0, "comm_radius must be positive."
    assert cfg.env.comm_radius_base > 0, "comm_radius_base must be positive."
    assert cfg.env.visual_radius > 0, "visual_radius must be positive."
    assert cfg.env.dt > 0, "dt must be positive."
    assert cfg.env.radar_bins >= 4, "radar_bins must be >= 4."
    assert cfg.env.spawn_delay >= 0, "spawn_delay must be >= 0."
    assert cfg.env.target_spawn_radius >= 0.0, "target_spawn_radius must be non-negative."
    assert 0.0 <= cfg.env.wall_restitution <= 1.0, "wall_restitution must be [0, 1]."
    assert cfg.training.num_envs > 0
    assert cfg.training.num_steps > 0
    assert 0 < cfg.training.gamma <= 1.0
    assert 0 < cfg.training.gae_lambda <= 1.0
    if bool(cfg.diversity.enabled):
        if str(cfg.diversity.method) != "forward_history":
            raise ValueError("diversity.method must be 'forward_history'.")
        assert int(cfg.diversity.history_len) >= 1, "diversity.history_len must be >= 1."
        assert int(cfg.diversity.num_roles) >= 0, "diversity.num_roles must be >= 0."
        assert float(cfg.diversity.beta) >= 0.0, "diversity.beta must be non-negative."
        assert float(cfg.diversity.aux_coef) >= 0.0, "diversity.aux_coef must be non-negative."
        assert float(cfg.diversity.l1_coef) >= 0.0, "diversity.l1_coef must be non-negative."
        assert float(cfg.diversity.reward_clip) > 0.0, "diversity.reward_clip must be positive."
        assert float(cfg.diversity.adapter_scale) >= 0.0, "diversity.adapter_scale must be non-negative."
        assert int(cfg.diversity.predictor_hidden_dim) >= 1, "diversity.predictor_hidden_dim must be >= 1."
        assert int(cfg.diversity.predictor_num_layers) >= 1, "diversity.predictor_num_layers must be >= 1."


if __name__ == "__main__":
    # Quick self-test: load and print the resolved config.
    cfg = load_config(cli_overrides=False)
    validate_config(cfg)
    print(OmegaConf.to_yaml(cfg))
    print(f"\nObs dim  : {compute_obs_dim(cfg)}")
    print(f"Action dim: {compute_action_dim(cfg)}")
