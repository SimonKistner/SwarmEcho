"""
swarmecho/config.py
====================
Configuration management for SwarmEcho.

Loads SwarmEcho configuration via OmegaConf, optionally merges mission-specific YAMLs and CLI overrides.
Exposes a clean typed interface via Python dataclasses.

Usage
-----
From Python:
    from swarmecho.core.config import load_config
    cfg = load_config()                        # loads Python defaults
    cfg = load_config("src/swarmecho/curriculum_config/levels/M01_small_maze.yaml")
    print(cfg.env.num_agents)

From CLI:
    uv run swarmecho-train level=M01_small_maze
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf

# ---------------------------------------------------------------------------
# Default config path
# ---------------------------------------------------------------------------

_SRC_ROOT = Path(__file__).resolve().parent.parent
MAP_DIR = _SRC_ROOT / "curriculum_config" / "maps"
LEVEL_DIR = _SRC_ROOT / "curriculum_config" / "levels"


# ---------------------------------------------------------------------------
# Typed dataclasses mirror the bundled curriculum configuration structure.
# These are used purely for IDE type-checking and documentation.
# The canonical config at runtime is always the OmegaConf DictConfig object.
# ---------------------------------------------------------------------------

@dataclass
class EnvConfig:
    # --- World and swarm ---
    num_agents: int = 6
    box_width: Optional[float] = None
    box_height: Optional[float] = None
    dt: float = 0.1

    # --- Perception and communication ---
    visual_radius: float = 5.0
    comm_radius: float = 10.0
    comm_radius_base: float = 15.0  # comm radius for base-station first hop (defaults to comm_radius)
    radar_bins: int = 8        # B — angular radar slices

    # --- Movement and physics ---
    max_speed: float = 10.0
    drag: float = 0.85
    wall_restitution: float = 0.2
    max_force: float = 50.0

    # --- Episode and map ---
    max_steps: int = 700
    spawn_delay: int = 5      # steps between drone activations (0 = all at once)
    map_names: list[str] = field(default_factory=lambda: ["M01_small_maze"])
    hold_chain_for: int = 50                       # number of consecutive timesteps the chain must be held before success
    terminate_on_target_found: bool = False       # if True, terminate episode immediately after target is found/delivered

    # --- Observation state ---
    observe_target_vector: bool = False            # if False, remove target odometry vector from actor observations
    observe_base_vector: bool = False              # if False, remove base odometry vector from actor observations
    observe_coverage_probe: bool = False            # if False, remove local coverage probe observations (reward calculations still use coverage)

@dataclass
class RewardConfig:
    # --- Local Rewards (Not divided by N) ---
    exploration_bonus: float = 0.25
    collision_penalty: float = 0.5
    finder_bonus: float = 50.0

    # --- Global Rewards (Divided by N) ---
    max_gap_penalty: float = 5.0       # absolute penalty when gap is at its maximum
    target_found_bonus: float = 100.0
    success_bonus: float = 500.0
    target_found_requires_delivery: bool = True
    chain_reward_system: str = "euclidean"  # "euclidean" | "discrete_finders_path"


@dataclass
class TrainingConfig:
    # --- Training budget and rollout ---
    total_timesteps: int = 250_000_000
    seed: int = 42
    num_envs: int = 4000
    num_steps: int = 100
    num_epochs: int = 4
    num_minibatches: int = 20

    # --- PPO optimization ---
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5

    # --- Checkpoint loading and resume behavior ---
    checkpoint_path: Optional[str] = None  # if set, resumes training from this path
    checkpoint_step_offset: Optional[int] = None  # if set, starts W&B step reporting at this offset (otherwise auto-detected from checkpoint)
    ckpt_loading_mode: str = "branch"  # "resume" (continue update count and WandB run) or "branch" (start update 0 and new WandB run)

@dataclass
class EvalConfig:
    # --- Evaluation schedule and training gate ---
    eval_freq: int = 20
    eval_offset: int = 1
    eval_min_train_success: float = 0.0

    # --- Parallel evaluation ---
    eval_parallel_envs: int = 4000
    eval_broadcast_on_curriculum_early_stop: bool = False

    # --- Evaluation-success early exit ---
    early_exit: bool = False
    early_exit_threshold: float = 0.99

    # --- Evaluation video ---
    eval_video: bool = True
    eval_video_freq: int = 20
    eval_video_offset: int = 1

    # --- Evaluation data and heatmaps ---
    eval_failed_chain_heatmap: bool = True
    eval_not_delivered_or_visually_found_heatmap: bool = True
    eval_not_deliv_not_visual_splitt_in_two: bool = False

    # --- Checkpoint saving ---
    save_model: bool = True
    checkpoint_freq: int = 50
    checkpoint_offset: int = 0
    checkpoint_dir: Optional[str] = None


@dataclass
class NetworkConfig:
    # --- Actor and critic ---
    hidden_dim:       int = 256
    num_layers:       int = 3    # critic depth
    actor_num_layers: int = 3    # actor depth (lighter, separate)
    actor_memory:     bool = True  # if True, actor uses per-agent GRU memory
    critic_memory:    bool = True  # if True, agent-centric critic uses per-agent GRU memory
    critic_type:      str = "observation"  # "observation" (legacy) or "privileged"

    # --- Recurrent communication ---
    memory_comm_enabled: bool = True
    memory_comm_every_k_steps: int = 5
    tarmac_sig_dim: int = 16
    tarmac_val_dim: int = 32
    tarmac_include_self: bool = False


@dataclass
class LoggingConfig:
    # --- Project & Directories ---
    run_name: Optional[str] = None
    use_timestamp_postfix: bool = False
    log_dir: str = "outputs"

    # --- WandB Logging ---
    wandb_mode: str = "online"   # "online", "offline", or "disabled"
    wandb_project: str = "SwarmEcho"
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None

    # --- Diagnostics ---
    suppress_xla_warnings: bool = True


@dataclass
class VisualizeConfig:
    # --- Communication overlay ---
    comm_color: str = "#03fbff"
    comm_fill_alpha: float = 0.02
    comm_edge_alpha: float = 0.50

    # --- Visibility overlay ---
    vis_color: str = "#03fbff"
    vis_fill_alpha: float = 0.10
    vis_edge_alpha: float = 0.50

    # --- Debug overlays ---
    render_conn_matrix: bool = True       # if True, render the connections matrix in the legend
    render_finders_path_debug: bool = False  # if True, render the finders path list in the legend when valid


@dataclass
class SwarmEchoConfig:
    """Root config. All sub-configs accessible as attributes."""
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
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

        Local Coverage (16, optional) — per circular direction (evenly spaced):
        ├─ is_cell_covered                        (1)   0/1 flag at sampling distance
        
        Radar (B × 4)  — per angular bin:
        ├─ inv_dist_wall                          (1)   ray-cast, norm by vis_r
        ├─ inv_dist_drone                         (1)   any active drone, norm by comm_r
        ├─ inv_dist_target_conn_drone             (1)   target-chain drones, norm by comm_r
        └─ inv_dist_base_conn_drone               (1)   base-chain drones, norm by comm_r

    Total: 9 + 16 + B * 4 by default. The base and target odometry vectors
    can be removed independently with env.observe_base_vector and
    env.observe_target_vector; the target-known flag remains present. The
    16-dim local coverage probe block can be removed independently with
    env.observe_coverage_probe without disabling coverage/reward calculations.
    """
    B = cfg.env.radar_bins
    self_dim = 9
    if not bool(cfg.env.get("observe_base_vector", True)):
        self_dim -= 2
    if not bool(cfg.env.get("observe_target_vector", True)):
        self_dim -= 2
    coverage_dim = 16 if bool(cfg.env.get("observe_coverage_probe", True)) else 0
    return self_dim + coverage_dim + B * 4


def compute_action_dim(_cfg: DictConfig) -> int:
    """Each agent outputs a 2D continuous force vector."""
    return 2


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def resolve_level_config_path(level: str) -> Path:
    """Resolve an exact level name or one unambiguous filename prefix."""
    token = str(level).strip()
    if not token:
        raise ValueError("Level name cannot be empty.")

    level_files = sorted(
        [
            *LEVEL_DIR.glob("*.yaml"),
            *LEVEL_DIR.glob("*.yml"),
        ],
        key=lambda path: path.name,
    )
    exact_filename = [path for path in level_files if path.name == token]
    if exact_filename:
        return exact_filename[0]

    exact_stem = [path for path in level_files if path.stem == token]
    if len(exact_stem) == 1:
        return exact_stem[0]
    if len(exact_stem) > 1:
        match_names = ", ".join(path.name for path in exact_stem)
        raise ValueError(
            f"Level name '{token}' is ambiguous. Use the complete filename. "
            f"Matches: {match_names}"
        )

    prefix_matches = [
        path for path in level_files if path.stem.startswith(token)
    ]
    if not prefix_matches:
        raise FileNotFoundError(
            f"Level config '{token}' was not found in {LEVEL_DIR}."
        )
    if len(prefix_matches) > 1:
        match_names = ", ".join(path.name for path in prefix_matches)
        raise ValueError(
            f"Level prefix '{token}' is ambiguous. Use an exact level name. "
            f"Matches: {match_names}"
        )
    return prefix_matches[0]


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
        Path to a YAML level configuration file.
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
        
        # Resolve the optional level before applying ordinary CLI overrides.
        level = cli_cfg.pop("level", None)
        if level is not None:
            level_path = resolve_level_config_path(str(level))
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


def find_closest_divisors(num_envs: int, target_mb: int) -> tuple[Optional[int], Optional[int]]:
    # Find divisors below target
    below = None
    for i in range(target_mb - 1, 0, -1):
        if num_envs % i == 0:
            below = i
            break
            
    # Find divisors above target
    above = None
    for i in range(target_mb + 1, num_envs + 1):
        if num_envs % i == 0:
            above = i
            break
            
    return below, above


# ---------------------------------------------------------------------------
# Quick validation on load
# ---------------------------------------------------------------------------

def validate_config(cfg: DictConfig) -> None:
    """
    Run basic sanity checks on the config.
    Raises ValueError with a clear message on any issue.
    """
    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    require(cfg.env.num_agents >= 1, "env.num_agents must be at least 1.")
    require(cfg.env.comm_radius > 0, "env.comm_radius must be positive.")
    require(cfg.env.comm_radius_base > 0, "env.comm_radius_base must be positive.")
    require(cfg.env.visual_radius > 0, "env.visual_radius must be positive.")
    require(cfg.env.dt > 0, "env.dt must be positive.")
    require(cfg.env.radar_bins >= 4, "env.radar_bins must be at least 4.")
    require(cfg.env.spawn_delay >= 0, "env.spawn_delay must be non-negative.")
    require(
        0.0 <= cfg.env.wall_restitution <= 1.0,
        "env.wall_restitution must be in [0, 1].",
    )
    if str(cfg.reward.get("chain_reward_system", "euclidean")) not in ("euclidean", "discrete_finders_path"):
        raise ValueError("reward.chain_reward_system must be 'euclidean' or 'discrete_finders_path'.")
    require(cfg.training.num_envs > 0, "training.num_envs must be positive.")
    require(cfg.training.num_steps > 0, "training.num_steps must be positive.")
    require(
        cfg.training.num_minibatches > 0,
        "training.num_minibatches must be positive.",
    )
    require(0 < cfg.training.gamma <= 1.0, "training.gamma must be in (0, 1].")
    require(
        0 < cfg.training.gae_lambda <= 1.0,
        "training.gae_lambda must be in (0, 1].",
    )
    require(
        int(cfg.evaluation.eval_freq) > 0,
        "evaluation.eval_freq must be positive.",
    )
    require(
        int(cfg.evaluation.eval_parallel_envs) > 0,
        "evaluation.eval_parallel_envs must be positive.",
    )
    require(
        int(cfg.evaluation.eval_video_freq) > 0,
        "evaluation.eval_video_freq must be positive.",
    )
    require(
        int(cfg.evaluation.checkpoint_freq) > 0,
        "evaluation.checkpoint_freq must be positive.",
    )
    require(
        0.0 <= float(cfg.evaluation.eval_min_train_success) <= 1.0,
        "evaluation.eval_min_train_success must be in [0, 1].",
    )
    require(
        0.0 <= float(cfg.evaluation.early_exit_threshold) <= 1.0,
        "evaluation.early_exit_threshold must be in [0, 1].",
    )

    if bool(cfg.network.memory_comm_enabled) and not bool(cfg.network.actor_memory):
        raise ValueError("network.memory_comm_enabled=true requires network.actor_memory=true.")
    if str(cfg.network.get("critic_type", "observation")) not in {"observation", "privileged"}:
        raise ValueError("network.critic_type must be 'observation' or 'privileged'.")
    if int(cfg.network.memory_comm_every_k_steps) < 1:
        raise ValueError("network.memory_comm_every_k_steps must be >= 1.")
    if int(cfg.network.tarmac_sig_dim) < 1:
        raise ValueError("network.tarmac_sig_dim must be >= 1.")
    if int(cfg.network.tarmac_val_dim) < 1:
        raise ValueError("network.tarmac_val_dim must be >= 1.")
    if (bool(cfg.network.actor_memory) or bool(cfg.network.critic_memory)):
        num_envs = int(cfg.training.num_envs)
        mb = int(cfg.training.num_minibatches)
        if num_envs % mb != 0:
            below, above = find_closest_divisors(num_envs, mb)
            suggestions = []
            if below is not None:
                suggestions.append(str(below))
            if above is not None:
                suggestions.append(str(above))
            sugg_str = " or ".join(suggestions)
            sugg_msg = f"\n\n ⚠️  Suggested valid choices close to {mb}: {sugg_str}. ⚠️" if suggestions else ""
            raise ValueError(
                f"Recurrent MAPPO requires training.num_envs ({num_envs}) divisible by training.num_minibatches ({mb}).{sugg_msg}"
            )


if __name__ == "__main__":
    # Quick self-test: load and print the resolved config.
    cfg = load_config(cli_overrides=False)
    validate_config(cfg)
    print(OmegaConf.to_yaml(cfg))
    print(f"\nObs dim  : {compute_obs_dim(cfg)}")
    print(f"Action dim: {compute_action_dim(cfg)}")

# ---------------------------------------------------------------------------
# 3D environment adapter configuration
# ---------------------------------------------------------------------------
# Training, evaluation, network, logging, and CLI semantics remain owned by
# this canonical config module. These typed wrappers retain strict validation
# while the 3D environment adapter is being connected to the common runner.

@dataclass(frozen=True)
class Network3DConfig:
    hidden_dim: int = 256
    num_layers: int = 3
    actor_num_layers: int = 3
    actor_memory: bool = True
    critic_memory: bool = True
    critic_type: str = "observation"
    memory_comm_enabled: bool = True
    memory_comm_every_k_steps: int = 5
    tarmac_sig_dim: int = 16
    tarmac_val_dim: int = 32
    tarmac_include_self: bool = False


@dataclass(frozen=True)
class Training3DConfig:
    total_timesteps: int = 250_000_000
    seed: int = 42
    num_envs: int = 4000
    num_steps: int = 100
    num_epochs: int = 4
    num_minibatches: int = 20
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    checkpoint_path: str | None = None
    checkpoint_step_offset: int | None = None
    ckpt_loading_mode: str = "branch"


@dataclass(frozen=True)
class Evaluation3DConfig:
    eval_freq: int = 20
    eval_offset: int = 1
    eval_min_train_success: float = 0.0
    eval_parallel_envs: int = 4000
    eval_broadcast_on_curriculum_early_stop: bool = False
    early_exit: bool = False
    early_exit_threshold: float = 0.99
    eval_video: bool = True
    eval_video_freq: int = 20
    eval_video_offset: int = 1
    eval_failed_chain_heatmap: bool = True
    eval_not_delivered_or_visually_found_heatmap: bool = True
    eval_not_deliv_not_visual_splitt_in_two: bool = False
    save_model: bool = True
    checkpoint_freq: int = 50
    checkpoint_offset: int = 0
    checkpoint_dir: str | None = None


@dataclass(frozen=True)
class Logging3DConfig:
    run_name: str | None = "M00_no_maze_open_cuboid_3D"
    use_timestamp_postfix: bool = False
    log_dir: str = "outputs"
    wandb_mode: str = "online"
    wandb_project: str = "SwarmEcho"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    suppress_xla_warnings: bool = True
    log_every: int = 1


@dataclass(frozen=True)
class Level3D:
    name: str
    map_names: list[str]
    building: object
    env: object
    reward: object
    training: Training3DConfig
    network: Network3DConfig
    evaluation: Evaluation3DConfig
    logging: Logging3DConfig

    @property
    def building_name(self) -> str:
        return self.map_names[0]

    @property
    def ideal_chain_margin_m(self) -> float:
        from swarmecho.env.baseline3d import maximum_chain_distance
        return maximum_chain_distance(self.env) - self.building.max_base_to_top_corner_m

    @property
    def num_updates(self) -> int:
        return self.training.total_timesteps // (self.training.num_envs * self.training.num_steps)


def _strict_3d_dataclass(cls, values: object, label: str):
    if not isinstance(values, dict):
        raise ValueError(f"{label} must be a mapping.")
    unknown = set(values) - set(cls.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown {label} fields: {', '.join(sorted(unknown))}.")
    return cls(**values)


def load_level_3d(name_or_path: str | Path = "M00_no_maze_open_cuboid_3D", overrides: list[str] | None = None) -> Level3D:
    """Load a strict 3D adapter level through the canonical config module."""
    from swarmecho.env.baseline3d import Baseline3DConfig, Baseline3DRewardConfig
    from swarmecho.env.buildings import load_building

    source = Path(name_or_path)
    if not source.exists():
        source = LEVEL_DIR / f"{source.stem}.yaml"
    if not source.exists():
        raise FileNotFoundError(f"3D level not found: {name_or_path}")
    data = OmegaConf.to_container(OmegaConf.load(source), resolve=True)
    if overrides:
        data = OmegaConf.to_container(OmegaConf.merge(OmegaConf.create(data), OmegaConf.from_dotlist(overrides)), resolve=True)
    allowed = {"env", "reward", "training", "network", "evaluation", "logging"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown 3D config sections: {', '.join(sorted(unknown))}.")
    env_data = dict(data.get("env", {}))
    map_names = env_data.pop("map_names", None)
    if not isinstance(map_names, list) or len(map_names) != 1:
        raise ValueError("3D env.map_names must select exactly one map.")
    level = Level3D(
        name=source.stem,
        map_names=[str(map_names[0])],
        building=load_building(MAP_DIR / f"{Path(map_names[0]).stem}.yaml"),
        env=_strict_3d_dataclass(Baseline3DConfig, env_data, "env"),
        reward=_strict_3d_dataclass(Baseline3DRewardConfig, data.get("reward", {}), "reward"),
        training=_strict_3d_dataclass(Training3DConfig, data.get("training", {}), "training"),
        network=_strict_3d_dataclass(Network3DConfig, data.get("network", {}), "network"),
        evaluation=_strict_3d_dataclass(Evaluation3DConfig, data.get("evaluation", {}), "evaluation"),
        logging=_strict_3d_dataclass(Logging3DConfig, data.get("logging", {}), "logging"),
    )
    if level.ideal_chain_margin_m < 0:
        raise ValueError(f"3D level {level.name!r} is geometrically unsolvable: ideal chain margin is {level.ideal_chain_margin_m:.3f} m.")
    if level.training.num_envs % level.training.num_minibatches:
        raise ValueError("Recurrent training requires num_envs divisible by num_minibatches.")
    if level.num_updates < 1:
        raise ValueError("training.total_timesteps must cover at least one rollout.")
    if level.logging.wandb_mode not in {"disabled", "offline", "online"}:
        raise ValueError("logging.wandb_mode must be disabled, offline, or online.")
    if level.reward.chain_reward_system != "euclidean":
        raise ValueError("3D reward.chain_reward_system currently supports only euclidean.")
    return level


def load_level_3d_cli(arguments: list[str]) -> Level3D:
    """Load a 3D level with the same key=value CLI contract as training."""
    level_name = "M00_no_maze_open_cuboid_3D"
    overrides = []
    for argument in arguments:
        if "=" not in argument:
            raise ValueError(f"Unexpected argument {argument!r}; use key=value overrides.")
        key, value = argument.split("=", 1)
        if key == "level":
            level_name = value
        else:
            overrides.append(argument)
    return load_level_3d(level_name, overrides)
