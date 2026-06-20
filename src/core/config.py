"""
swarmecho/config.py
====================
Configuration management for SwarmEcho.

Loads SwarmEcho configuration via OmegaConf, optionally merges mission-specific YAMLs and CLI overrides.
Exposes a clean typed interface via Python dataclasses.

Usage
-----
From Python:
    from core.config import load_config
    cfg = load_config()                        # loads Python defaults
    cfg = load_config("src/curriculum_config/levels/A01_warehouse.yaml") # loads warehouse mission
    print(cfg.env.num_agents)

From CLI:
    uv run python src/training/train.py level=A01 env.num_agents=16
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
    # --- Spawn overrides (B-series curriculum) ---
    use_random_base_spawn: bool = True    # if False, base always spawns at map centre
    use_random_drone_spawn: bool = True   # if False, drones spawn stacked at base_pos
    target_spawn_method: str = "map_defined"  # "map_defined", "ring", "outside_base"
    target_spawn_radius: float = 0.0      # if > 0, target spawns in a circle of this max radius around base (for "ring")
    target_spawn_radius_min: float = 0.0  # if > 0, target spawns in a ring (min to max radius) (for "ring")
    target_invalid_spawn_base_radius: float = 0.0 # if > 0, target cannot spawn within this radius of the base (for "outside_base")
    precover_base_comm: bool = False              # if True, cells in communication range of the base station are covered from reset
    hold_chain_for: int = 0                       # number of consecutive timesteps the chain must be held before success
    mem_test_mask_nonlocal_obs: bool = False      # MEM_T8-only: zero non-local observation channels to prevent T identity leaks
    observe_target_vector: bool = True            # if False, remove target odometry vector from actor observations
    observe_base_vector: bool = True              # if False, remove base odometry vector from actor observations
    log_adjacency_matrix: bool = False            # if True, log direct connection matrix in EnvState (can be costly in training)
    terminate_on_target_found: bool = False       # if True, terminate episode immediately after target is found/delivered
    experimental_setup: bool = False              # if True, disable normal task-only machinery such as chain/finder-path rewards



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
    target_found_requires_delivery: bool = True
    back_to_target_after_delivery: bool = False
    chain_reward_system: str = "euclidean"  # "euclidean" | "discrete_finders_path"
    only_reward_chain_from_target: bool = False
    only_shortest_path_chain_reward: bool = False
    reward_single_shortest_path: bool = True
    only_explor_individual: bool = False  # keep exploration/safety local; share chain-related rewards
    every_reward_global: bool = False     # share every reward/penalty equally across agents



@dataclass
class TrainingConfig:
    seed: int = 42
    num_envs: int = 1024
    num_steps: int = 256
    num_epochs: int = 4
    num_minibatches: Optional[int] = 8
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    total_timesteps: int = 50_000_000
    checkpoint_path: Optional[str] = None  # if set, resumes training from this path
    checkpoint_step_offset: Optional[int] = None  # if set, starts W&B step reporting at this offset (otherwise auto-detected from checkpoint)
    ckpt_loading_mode: str = "branch"  # "resume" (continue update count and WandB run) or "branch" (start update 0 and new WandB run)
    resume_update: Optional[bool] = None  # Deprecated legacy parameter (use ckpt_loading_mode instead)
    warn_vram_limit: bool = False
    abort_on_vram_limit: bool = False
    vram_limit_gb: float = 20.0
    eval_parallel: bool = False
    eval_parallel_envs: int = 4000
    eval_parallel_early_exit_threshold: Optional[float] = None



@dataclass
class NetworkConfig:
    hidden_dim:       int = 256
    num_layers:       int = 3    # critic depth
    actor_num_layers: int = 3    # actor depth (lighter, separate)
    critic_type:      str = "agent_centric"  # "agent_centric" | "global_mean"
    actor_memory:     bool = False  # if True, actor uses per-agent GRU memory
    critic_memory:    bool = False  # if True, agent-centric critic uses per-agent GRU memory
    memory_comm_enabled: bool = False
    memory_comm_every_k_steps: int = 5
    tarmac_sig_dim: int = 64
    tarmac_val_dim: int = 128
    tarmac_include_self: bool = True


@dataclass
class LoggingConfig:
    # --- Project & Directories ---
    project: str = "SwarmEcho"
    run_name: Optional[str] = None
    use_timestamp_postfix: bool = False
    log_dir: str = "outputs"

    # --- WandB Logging ---
    wandb_mode: str = "online"   # "online", "offline", or "disabled"
    wandb_project: str = "SwarmEcho"
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None

    # --- Frequencies ---
    log_freq: int = 10
    eval_freq: int = 50           # Run evaluation and heatmap generation every N updates
    eval_offset: int = 1          # Offset for eval_freq modulo scheduling
    eval_video_freq: Optional[int] = 50 # Run video rendering evaluation every N updates. If None, defaults to eval_freq.
    eval_video_offset: int = 1     # Offset for eval_video_freq modulo scheduling

    # --- Model Checkpointing ---
    save_model: bool = True
    checkpoint_freq: int = 50     # Save model checkpoint every N updates
    checkpoint_offset: int = 0    # Offset for checkpoint_freq modulo scheduling
    checkpoint_dir: str = "outputs/checkpoints"

    # --- Diagnostics & Details ---
    suppress_xla_warnings: bool = True
    obs_log: bool = False
    memory_diagnostic_probe: bool = False  # Train a linear probe on base memory to predict target cell

    # --- Mid-run Evaluation Toggles ---
    eval_video: bool = True       # Render rollout video for evaluation episodes
    eval_failed_chain_heatmap: bool = False  # Generate heatmap of target positions for failed chain deliveries from sliding window
    eval_not_delivered_or_visually_found_heatmap: bool = False  # Generate heatmap of target positions not delivered/visually found
    eval_not_deliv_not_visual_splitt_in_two: bool = False      # If true, split the not-delivered/not-visual heatmap into two separate files

    # --- Deprecated / Legacy parameters (kept for backward compatibility with older runs) ---
    video_freq: Optional[int] = None # legacy
    eval_episodes: Optional[int] = None
    async_video: Optional[bool] = None


@dataclass
class VisualizeConfig:
    renderer: str = "fast"              # LEGACY fallback; prefer explicit params below
    train_eval_renderer: str = "fast"   # renderer used for mid-training single-episode videos
    final_eval_renderer: str = "fast"   # renderer used for final eval / standalone evaluate.py

    # --- Eval rendering control ---
    selective_eval_render: bool = False  # False = legacy mode; True = selective bucket mode

    # if selective_eval_render=False:
    eval_render_videos: int = 1          # episodes to compute AND render immediately (legacy)

    # if selective_eval_render=True:
    eval_max_compute_episodes: int = 100  # hard ceiling on episodes simulated
    eval_render_successes: int = 0       # SUCCESS_ bucket target  (0 = skip success renders)
    eval_render_failures:  int = 3       # FAIL_ bucket target     (0 = skip fail renders)
    # Note: both buckets=0 is valid → runs eval_max_compute_episodes, prints full stats, no videos.

    comm_color: str = "#03fbff"
    comm_fill_alpha: float = 0.02
    comm_edge_alpha: float = 0.50
    vis_color: str = "#03fbff"
    vis_fill_alpha: float = 0.10
    vis_edge_alpha: float = 0.50
    render_conn_matrix: bool = True       # if True, render the connections matrix in the legend
    render_finders_path_debug: bool = False  # if True, render the finders path list in the legend when valid


@dataclass
class CurriculumConfig:
    success_threshold: Optional[float] = None
    metric: str = "success"             # "success" or "target_found"
    mode: str = "train"                 # "train" or "eval"



@dataclass
class SwarmEchoConfig:
    """Root config. All sub-configs accessible as attributes."""
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
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

    Total: 9 + 16 + B * 4 by default. The base and target odometry vectors
    can be removed independently with env.observe_base_vector and
    env.observe_target_vector; the target-known flag remains present.
    """
    B = cfg.env.radar_bins
    self_dim = 9
    if not bool(cfg.env.get("observe_base_vector", True)):
        self_dim -= 2
    if not bool(cfg.env.get("observe_target_vector", True)):
        self_dim -= 2
    return self_dim + 16 + B * 4


def compute_action_dim(_cfg: DictConfig) -> int:
    """Each agent outputs a 2D continuous force vector."""
    return 2


def calculate_optimal_minibatches(
    num_envs: int,
    num_steps: int,
    recurrent: bool,
    max_transitions_per_mb: int = 30000
) -> int:
    """
    Calculate the optimal number of minibatches to keep transitions per minibatch
    under the specified VRAM safety limit, ensuring divisibility requirements are met.
    """
    import math
    total_batch_size = num_envs * num_steps
    
    if recurrent:
        min_mb = math.ceil(total_batch_size / max_transitions_per_mb)
        mb = max(1, min_mb)
        if mb > num_envs:
            mb = num_envs
        while num_envs % mb != 0:
            mb += 1
    else:
        min_mb = math.ceil(total_batch_size / max_transitions_per_mb)
        mb = max(1, min_mb)
        while total_batch_size % mb != 0:
            mb += 1
            
    return mb


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

    # ---------------------------------------------------------------------------
    # Dynamic Minibatch Calculation & VRAM warnings
    # ---------------------------------------------------------------------------
    if cfg.training.num_minibatches is None:
        actor_memory = bool(cfg.network.actor_memory)
        critic_memory = bool(cfg.network.critic_memory)
        recurrent = actor_memory or critic_memory
        
        # Calculate safety target based on vram_limit_gb (e.g. 1500 transitions/GB -> 30,000 transitions for 20GB)
        safety_target = int(cfg.training.vram_limit_gb * 1500)
        
        mb = calculate_optimal_minibatches(
            num_envs=int(cfg.training.num_envs),
            num_steps=int(cfg.training.num_steps),
            recurrent=recurrent,
            max_transitions_per_mb=safety_target
        )
        
        # Temporarily allow modifying
        OmegaConf.set_readonly(cfg, False)
        cfg.training.num_minibatches = mb
        
        total_batch_size = int(cfg.training.num_envs) * int(cfg.training.num_steps)
        transitions_per_mb = total_batch_size // mb
        print(f"  [config] Auto-calculated training.num_minibatches = {mb} "
              f"({transitions_per_mb:,} transitions/mb, safety target: {safety_target:,})")

    elif cfg.training.warn_vram_limit:
        mb = int(cfg.training.num_minibatches)
        num_envs = int(cfg.training.num_envs)
        num_steps = int(cfg.training.num_steps)
        total_batch_size = num_envs * num_steps
        transitions_per_mb = total_batch_size // mb
        
        # Safety warning threshold: e.g. 1550 transitions/GB -> 31,000 transitions for 20GB
        limit = int(cfg.training.vram_limit_gb * 1550)
        
        if transitions_per_mb > limit:
            actor_memory = bool(cfg.network.actor_memory)
            critic_memory = bool(cfg.network.critic_memory)
            recurrent = actor_memory or critic_memory
            
            safety_target = int(cfg.training.vram_limit_gb * 1500)
            safe_mb = calculate_optimal_minibatches(
                num_envs=num_envs,
                num_steps=num_steps,
                recurrent=recurrent,
                max_transitions_per_mb=safety_target
            )
            
            print("\n" + "!" * 80)
            print(f"⚠️  WARNING: VRAM Safety Limit Exceeded check failed!")
            print(f"   Current configuration: num_minibatches = {mb}")
            print(f"   This results in {transitions_per_mb:,} transitions per minibatch.")
            print(f"   This exceeds the safety limit of {limit:,} transitions per minibatch ({cfg.training.vram_limit_gb} GiB usable VRAM constraint).")
            print(f"   It is highly likely to cause an Out-Of-Memory (OOM) error on RTX 4090.")
            print(f"👉  Recommended safe choice: training.num_minibatches={safe_mb} "
                  f"({total_batch_size // safe_mb:,} transitions/mb)")
            print("!" * 80 + "\n")

            if cfg.training.abort_on_vram_limit:
                import sys
                sys.exit(1)

    if cfg.logging.run_name:
        print(f"  Run Name         : {cfg.logging.run_name}")

    if cfg.reward.every_reward_global:
        cfg.reward.only_explor_individual = False
        cfg.reward.only_shortest_path_chain_reward = False
    elif cfg.reward.only_explor_individual:
        cfg.reward.only_shortest_path_chain_reward = False

    if bool(cfg.network.memory_comm_enabled):
        # Memory communication already requires the direct adjacency matrix to build
        # sender/receiver masks, so expose it in EnvState/logging as well.
        OmegaConf.set_readonly(cfg, False)
        cfg.env.log_adjacency_matrix = True

    # Automatically enable adjacency matrix logging for render, evaluate, and test_physics scripts
    # (since the matrix is needed for visuals/diagnostics in those entry points)
    import sys
    if sys.argv and len(sys.argv[0]) > 0:
        script_name = Path(sys.argv[0]).name
        if any(word in script_name for word in ["evaluate", "render", "preview", "test_physics"]):
            # Respect explicit disabled settings in overrides
            explicit_false = False
            if overrides is not None:
                for o in overrides:
                    if "env.log_adjacency_matrix=false" in o.lower():
                        explicit_false = True
            if not explicit_false:
                OmegaConf.set_readonly(cfg, False)
                cfg.env.log_adjacency_matrix = True

    # Auto-false diagnostic probe if memory or memory communication is false/off
    if not (bool(cfg.network.actor_memory) and bool(cfg.network.memory_comm_enabled)):
        OmegaConf.set_readonly(cfg, False)
        cfg.logging.memory_diagnostic_probe = False

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
    assert cfg.env.num_agents >= 1, "Need at least 1 agent."
    assert cfg.env.comm_radius > 0, "comm_radius must be positive."
    assert cfg.env.comm_radius_base > 0, "comm_radius_base must be positive."
    assert cfg.env.visual_radius > 0, "visual_radius must be positive."
    assert cfg.env.dt > 0, "dt must be positive."
    assert cfg.env.radar_bins >= 4, "radar_bins must be >= 4."
    assert cfg.env.spawn_delay >= 0, "spawn_delay must be >= 0."
    assert cfg.env.target_spawn_radius >= 0.0, "target_spawn_radius must be non-negative."
    assert 0.0 <= cfg.env.wall_restitution <= 1.0, "wall_restitution must be [0, 1]."
    if str(cfg.reward.get("chain_reward_system", "euclidean")) not in ("euclidean", "discrete_finders_path"):
        raise ValueError("reward.chain_reward_system must be 'euclidean' or 'discrete_finders_path'.")
    assert cfg.training.num_envs > 0
    assert cfg.training.num_steps > 0
    assert 0 < cfg.training.gamma <= 1.0
    assert 0 < cfg.training.gae_lambda <= 1.0
    # Visualize / eval rendering
    assert str(cfg.visualize.train_eval_renderer) in ("fast", "slow"), \
        "visualize.train_eval_renderer must be 'fast' or 'slow'"
    assert str(cfg.visualize.final_eval_renderer) in ("fast", "slow"), \
        "visualize.final_eval_renderer must be 'fast' or 'slow'"
    assert int(cfg.visualize.eval_render_videos) >= 1, \
        "visualize.eval_render_videos must be >= 1"
    if bool(cfg.visualize.selective_eval_render):
        assert int(cfg.visualize.eval_max_compute_episodes) > 0, \
            "visualize.eval_max_compute_episodes must be > 0 when selective_eval_render=True"
        assert int(cfg.visualize.eval_render_successes) >= 0, \
            "visualize.eval_render_successes must be >= 0"
        assert int(cfg.visualize.eval_render_failures) >= 0, \
            "visualize.eval_render_failures must be >= 0"
        # Both buckets=0 is valid: compute episodes, print stats, render nothing.
    if hasattr(cfg, "curriculum") and cfg.curriculum is not None:
        if cfg.curriculum.get("metric", None) is not None:
            valid_metrics = ("success", "target_found")
            if str(cfg.curriculum.metric) not in valid_metrics:
                raise ValueError(f"curriculum.metric must be one of {valid_metrics}")
        if cfg.curriculum.get("mode", None) is not None:
            valid_modes = ("train", "eval")
            if str(cfg.curriculum.mode) not in valid_modes:
                raise ValueError(f"curriculum.mode must be one of {valid_modes}")

    if bool(cfg.network.critic_memory) and str(cfg.network.critic_type) != "agent_centric":
        raise ValueError("network.critic_memory=true requires network.critic_type='agent_centric'.")
    if bool(cfg.network.memory_comm_enabled) and not bool(cfg.network.actor_memory):
        raise ValueError("network.memory_comm_enabled=true requires network.actor_memory=true.")
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
