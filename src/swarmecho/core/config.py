"""User-facing level settings, defaults, and key=value CLI overrides."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import math
import warnings
from swarmecho.core.compatibility import (
    DEFAULT_LEVEL_NAME,
    REWARD_NAME_ALIASES,
    canonical_level_name,
    canonical_map_name,
    resolve_level_file,
    resolve_map_file,
)

_SRC_ROOT = Path(__file__).resolve().parent.parent
MAP_DIR = _SRC_ROOT / "curriculum_config" / "maps" / "custom_maps"
LEVEL_DIR = _SRC_ROOT / "curriculum_config" / "levels"

# Fixed across maps, population stages and available slots; not a deployment cap.
DEPLOYED_AGENTS_OBS_REF = 10.0

CHAIN_SUCCESS_CONDITIONS = frozenset({"chain_held", "chain_held_and_cleand_up"})
SUCCESS_CONDITIONS = CHAIN_SUCCESS_CONDITIONS

# Adaptive population control is opt-in; disabled levels retain their old shapes.
@dataclass(frozen=True)
class DecommissioningConfig:
    enabled: bool = False
    decommission_radius_m: float = 1.0
    decommission_hold_steps: int = 15


# env
@dataclass(frozen=True)
class EnvConfig:
    num_agents: int = 5
    decommissioning: DecommissioningConfig = DecommissioningConfig()

    dt: float = 0.1  # Simulated time per step; scales velocity and position updates.
    max_force: float = 15.0
    max_speed: float = 5.0
    drag: float = 0.85
    drone_radius: float = 0.25

    comm_radius_base: float = 6.0
    comm_radius: float = 5.0
    visual_radius: float = 4.0
    wall_sensing_rays: int = 16  # fibonacci: any integer >= 4; xyz: 14 | 18 | 26
    wall_sensing_ray_max_distance: float = 8.0  # Metres; independent of target visual_radius.
    drone_radar_bins: int = 8
    radar_mode: str = "max_proximity"  # Baseline radar; band_sep_counting is a retained ablation.
    # Settings used only by the band_sep_counting ablation.
    distance_bands: int = 1  # Equal-width bands; 0 is an alias for 1.
    radar_count_cap: int = 3  # Saturation count per band and peer group.

    target_wall_buffer_fraction: float = 0.1
    spawn_delay: int = 5  # Minimum steps between automatic launches.

    max_steps: int = 700
    hold_chain_for: int = 50
    movement_epsilon: float = 1e-3
    success_condition: str = "chain_held"  # chain_held | chain_held_and_cleand_up

    # Debugging-only legacy observation; never treat as a design decision or candidate.
    observe_visually_gated_target_displacement: bool = False  # Visible target XYZ offset / visual_radius.
    # Debugging-only legacy observation; never treat as a design decision or candidate.
    observe_chain_contributor: bool = False
    observe_decommission_zone: bool = True  # Inside the valid decommission sphere; adaptive size only.
    observe_base_population: bool = False  # Active population / DEPLOYED_AGENTS_OBS_REF, while base-connected.
    # Debugging-only legacy observation; never treat as a design decision or candidate.
    coverage_voxel_size: float | None = None # None -> use the building’s cell size i.e. 5m

    obstacle_planning_clearance_m: float = 0.1
    obstacle_layout_version: str = "three_aabb_v1"

    roadmap_node_density: float = 1.0
    roadmap_merge_wall_end_nodes: bool = False


def validate_decommissioning(cfg: EnvConfig) -> None:
    a = cfg.decommissioning
    if type(a.enabled) is not bool:
        raise ValueError("decommissioning.enabled must be boolean.")
    if type(a.decommission_hold_steps) is not int or a.decommission_hold_steps < 1:
        raise ValueError("decommissioning.decommission_hold_steps must be a positive integer.")
    if not math.isfinite(a.decommission_radius_m) or a.decommission_radius_m <= 0:
        raise ValueError("decommissioning.decommission_radius_m must be positive and finite.")
    if a.enabled and cfg.success_condition not in CHAIN_SUCCESS_CONDITIONS:
        raise ValueError("Decommissioning requires chain_held or chain_held_and_cleand_up success.")


def validate_radar_config(cfg: EnvConfig) -> None:
    """Validate radar settings for both loaded levels and direct environments."""
    for name in ("observe_visually_gated_target_displacement",
                 "observe_base_population",
                 "observe_decommission_zone"):
        if type(getattr(cfg, name)) is not bool:
            raise ValueError(f"env.{name} must be boolean.")
    for name in ("wall_sensing_rays", "drone_radar_bins"):
        value = getattr(cfg, name)
        if type(value) is not int or value < 4:
            raise ValueError(f"env.{name} must be an integer of at least 4.")
    if cfg.radar_mode not in {"max_proximity", "band_sep_counting"}:
        raise ValueError("env.radar_mode must be max_proximity or band_sep_counting.")
    if type(cfg.distance_bands) is not int or cfg.distance_bands < 0:
        raise ValueError("env.distance_bands must be a nonnegative integer (0 means 1).")
    if type(cfg.radar_count_cap) is not int or cfg.radar_count_cap < 1:
        raise ValueError("env.radar_count_cap must be a positive integer.")
    for name in ("comm_radius", "visual_radius", "wall_sensing_ray_max_distance"):
        value = getattr(cfg, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"env.{name} must be finite and positive.")


# random_buildings
@dataclass(frozen=True)
class RandomBuildingConfig:
    enabled: bool = False

    length_m: float = 30.0
    width_m: float = 30.0
    stories: int = 3
    cell_size_m: float = 5.0
    wall_thickness_m: float = 0.25
    tile_thickness_m: float = 0.25

    rooms_per_story_min: int = 4
    rooms_per_story_max: int = 8
    extra_door_probability: float = 0.15
    window_probability: float = 0.2
    staircase_max: int = 1

    advanced_exclustion_base_LOS: bool = False

    save_training_maps: bool = False
    load_maps_from_bank: bool = False


# reward
@dataclass(frozen=True)
class RewardConfig:
    # Signed individual amounts: negative values are costs; zero disables a term.
    # Numeric defaults follow G02, using the former four-agent split as reference.
    # Independent, opt-in surplus-agent rewards during an intact chain hold.
    non_contributor_return_reward_per_meter: float = 0.0
    non_contributor_departure_reward_per_meter: float = 0.0
    non_contributor_decommission_reward: float = 0.0  # One-shot surplus-removal amount; G03 matches its success reward.
    all_decommissioned_penalty: float = -1000.0  # Full signed amount to each last acting drone.
    chain_efficiency_reward: float = 0.0  # Per eligible drone times its path efficiency; no population divisor.

    exploration_reward: float = 2.0  # Per new voxel; simultaneous viewers split that voxel's credit.
    base_reporter_reward: float = 100.0  # Each direct base deliverer.
    target_delivery_reward: float = 125.0  # One-shot amount to each active drone when the target reaches base.
    success_reward: float = 3000.0  # Full amount to every active drone at first success; no population divisor.

    collision_reward: float = -0.5  # Per colliding drone per transition.
    time_presure_reward: float = -0.75  # Each acting drone every step, including its removal step.
    # Recurring amount per credited closed metre per step; retain local soft shares.
    closed_gap_reward_per_meter: float = 0.2  # Recurring per credited closed metre each step.

    @property
    def non_contributor_movement_enabled(self) -> bool:
        return (self.non_contributor_return_reward_per_meter != 0
                or self.non_contributor_departure_reward_per_meter != 0)

    @property
    def non_contributor_rewards_enabled(self) -> bool:
        return self.non_contributor_movement_enabled or self.non_contributor_decommission_reward != 0

def validate_reward_config(cfg: RewardConfig) -> None:
    """All reward coefficients are signed, finite amounts; no implicit negation."""
    for name in (
        "non_contributor_return_reward_per_meter", "non_contributor_departure_reward_per_meter",
        "non_contributor_decommission_reward",
        "all_decommissioned_penalty",
        "chain_efficiency_reward", "exploration_reward", "base_reporter_reward",
        "target_delivery_reward", "success_reward",
        "collision_reward", "time_presure_reward", "closed_gap_reward_per_meter",
    ):
        value = getattr(cfg, name)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"reward.{name} must be finite (positive, zero or negative).")


# training
# Keep training action perturbations aligned with the default used by the
# standalone robust checkpoint evaluator.


@dataclass(frozen=True)
class ForwardTargetCurriculumConfig:
    enabled: bool = False

@dataclass(frozen=True)
class TargetSpawnEligibilityConfig:
    enabled: bool = False
    samples_per_map: int = 800
    minimum_cell_fraction: float = 0.01


def validate_target_spawn_eligibility(cfg):
    if type(cfg.enabled) is not bool:
        raise ValueError("training.target_spawn_eligibility.enabled must be boolean.")
    if type(cfg.samples_per_map) is not int or cfg.samples_per_map < 1:
        raise ValueError("training.target_spawn_eligibility.samples_per_map must be a positive integer.")
    value = cfg.minimum_cell_fraction
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("training.target_spawn_eligibility.minimum_cell_fraction must be in [0, 1].")




@dataclass(frozen=True)
class AdaptiveMapSamplingConfig:
    enabled: bool = False
    min_probability_fraction: float = 0.30
    success_window_episodes_per_map: int = 500
    probability_smoothing_half_life_updates: float = 3.0
    eligibility_samples_per_map: int = 800
    held_out_maps: int = 10


def validate_adaptive_map_sampling(cfg: AdaptiveMapSamplingConfig) -> None:
    if type(cfg.enabled) is not bool:
        raise ValueError("training.adaptive_map_sampling.enabled must be boolean.")
    if not cfg.enabled:
        return  # Weighting is inactive; adaptive cells validate their support budget separately.
    for name in ("success_window_episodes_per_map", "eligibility_samples_per_map", "held_out_maps"):
        if type(getattr(cfg, name)) is not int or getattr(cfg, name) < 1:
            raise ValueError(f"training.adaptive_map_sampling.{name} must be a positive integer.")
    f = cfg.min_probability_fraction
    h = cfg.probability_smoothing_half_life_updates
    if type(f) not in (int, float) or not math.isfinite(f) or not 0 <= f <= 1:
        raise ValueError("training.adaptive_map_sampling.min_probability_fraction must be in [0, 1].")
    if type(h) not in (int, float) or not math.isfinite(h) or h <= 0:
        raise ValueError("Adaptive map probability half-life must be finite and positive.")


@dataclass(frozen=True)
class AdaptiveTargetSpawnConfig:
    enabled: bool = False
    initial_target_agent_budget: int | None = None  # Optional reset-time target budget; population stays fixed.
    min_probability_fraction: float = 0.20
    success_window_episodes_per_cell: int = 500
    probability_smoothing_half_life_updates: float = 3.0
    plateau_warning_enabled: bool = True
    plateau_warning_horizon_updates: int = 50
    plateau_warning_improvement_threshold_pp: float = 3.0


def validate_adaptive_target_spawn(cfg: AdaptiveTargetSpawnConfig) -> None:
    if cfg.initial_target_agent_budget is not None and (
            type(cfg.initial_target_agent_budget) is not int or cfg.initial_target_agent_budget < 1 or not cfg.enabled):
        raise ValueError("training.adaptive_target_spawn.initial_target_agent_budget must be a positive integer "
                         "with adaptive_target_spawn.enabled=true, or null.")
    for name in ("enabled", "plateau_warning_enabled"):
        if type(getattr(cfg, name)) is not bool:
            raise ValueError(f"training.adaptive_target_spawn.{name} must be boolean.")
    for name in ("success_window_episodes_per_cell", "plateau_warning_horizon_updates"):
        if type(getattr(cfg, name)) is not int or getattr(cfg, name) < 1:
            raise ValueError(f"training.adaptive_target_spawn.{name} must be a positive integer.")
    for name, lower, upper in (("min_probability_fraction", 0., 1.),
                               ("plateau_warning_improvement_threshold_pp", 0., 100.)):
        value = getattr(cfg, name)
        if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"training.adaptive_target_spawn.{name} must be in [{lower}, {upper}].")
    value = cfg.probability_smoothing_half_life_updates
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("training.adaptive_target_spawn.probability_smoothing_half_life_updates must be finite and positive.")


@dataclass(frozen=True)
class DynamicMaxGeoDistanceConfig:
    candidate_spacing_fraction: float = 1.0  # Random budget = ceil(world_volume / (cell_size * fraction)^3).
    max_candidates_per_map: int = 4096  # Explicit guard; candidates are never truncated.


def validate_dynamic_max_geo_distance(cfg):
    value = cfg.candidate_spacing_fraction
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("candidate_spacing_fraction must be finite and positive.")
    if type(cfg.max_candidates_per_map) is not int or cfg.max_candidates_per_map < 1:
        raise ValueError("max_candidates_per_map must be a positive integer.")


@dataclass(frozen=True)
class OOMRecoveryConfig:
    enabled: bool = False
    parameter: str = "num_envs"  # num_envs | num_steps | num_minibatches | optimizer_microbatches
    decrement: int = 500  # Used when factor is null; preserves the original envs -500 policy.
    factor: float | None = None  # <1 for envs/steps; >1 for smaller optimizer minibatches.
    max_retries: int | None = None  # null preserves retries until the shape is exhausted.


def validate_oom_recovery(cfg: OOMRecoveryConfig) -> None:
    if type(cfg.enabled) is not bool:
        raise ValueError("training.oom_recovery.enabled must be boolean.")
    if cfg.parameter not in {"num_envs", "num_steps", "num_minibatches", "optimizer_microbatches"}:
        raise ValueError("Unknown training.oom_recovery.parameter.")
    if type(cfg.decrement) is not int or cfg.decrement < 1:
        raise ValueError("training.oom_recovery.decrement must be a positive integer.")
    if cfg.factor is not None:
        if (type(cfg.factor) not in (int, float) or not math.isfinite(cfg.factor)
                or (cfg.factor <= 1 if cfg.parameter in {"num_minibatches", "optimizer_microbatches"} else not 0 < cfg.factor < 1)):
            raise ValueError("training.oom_recovery.factor must be finite and greater than 1 for "
                             "num_minibatches/optimizer_microbatches, or between 0 and 1 for num_envs/num_steps.")
    elif cfg.parameter in {"num_minibatches", "optimizer_microbatches"}:
        raise ValueError("OOM recovery for num_minibatches/optimizer_microbatches requires a factor greater than 1.")
    if cfg.max_retries is not None and (type(cfg.max_retries) is not int or cfg.max_retries < 0):
        raise ValueError("training.oom_recovery.max_retries must be null or a nonnegative integer.")


@dataclass(frozen=True)
class GeoAuxHeadConfig:
    hidden_sizes: tuple[int, ...] = (256, 128)
    learning_rate: float = 0.0003


@dataclass(frozen=True)
class GeoAuxSamplingConfig:
    every_updates: int = 1
    snapshots_per_agent: int = 1
    selection: str = "stratified_eligible_steps"


@dataclass(frozen=True)
class GeoAuxScheduleConfig:
    warmup_updates: int = 10
    ramp_updates: int = 100


DEFAULT_PATH_GRID_SHAPE = (5, 5, 5)


@dataclass(frozen=True)
class GeoProbeAuxiliaryConfig:
    lambda_target: float = 0.03
    metric_threshold: float = 0.8
    gradient_diagnostics: bool = True
    head: GeoAuxHeadConfig = GeoAuxHeadConfig()
    sampling: GeoAuxSamplingConfig = GeoAuxSamplingConfig()
    schedule: GeoAuxScheduleConfig = GeoAuxScheduleConfig()


@dataclass(frozen=True)
class TrainingConfig:
    total_timesteps: int = 250_000_000
    # Per invocation, including setup/JIT and periodic evaluation; final saving/media follow.
    max_wall_seconds: float | None = None
    # Per training attempt after setup, including first JIT/evaluation; final saving/media follow.
    max_training_seconds: float | None = None
    num_envs: int = 4000
    num_maps: int | None = None  # Persistent random training maps; null retains the legacy lane-count fallback.
    num_steps: int = 100
    num_epochs: int = 4
    num_minibatches: int = 20
    optimizer_microbatches: int = 1 # Execution chunks per PPO minibatch; gradients accumulate before one Adam step.
    rollout_batch_size: int | None = None # Execution-only chunking; null uses every lane. OOM recovery may reduce it.
    geo_probe_auxiliary: GeoProbeAuxiliaryConfig = GeoProbeAuxiliaryConfig()
    seed: int = 42
    seed_map_pool: bool = False  # Generate and reuse a persistent building pool keyed by training.seed.
    map_selection_seed: int | None = None  # None follows seed; explicit value fixes bank splits across training seeds.
    map_pool_path: str | None = None  # Optional exact precompiled pool; incompatible/missing pools fail instead of regenerating.

    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    actor_clip_eps: float = 0.2
    value_clip_eps: float = 0.2  # Measured in normalized critic units.
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5

    diagnostics_every: int = 1  # PPO updates between before/after diagnostic passes
    profile_timing: bool = False # debug diagnostic

    randomize_base: bool = False
    spawn_pair_max_attempts: int = 1024
    adaptive_target_spawn: AdaptiveTargetSpawnConfig = AdaptiveTargetSpawnConfig()
    forward_target_curriculum: ForwardTargetCurriculumConfig = ForwardTargetCurriculumConfig()
    target_spawn_eligibility: TargetSpawnEligibilityConfig = TargetSpawnEligibilityConfig()
    adaptive_map_sampling: AdaptiveMapSamplingConfig = AdaptiveMapSamplingConfig()
    dynamic_max_geo_distance: DynamicMaxGeoDistanceConfig = DynamicMaxGeoDistanceConfig()
    oom_recovery: OOMRecoveryConfig = OOMRecoveryConfig()

    checkpoint_path: str | None = None
    checkpoint_step_offset: int | None = None
    ckpt_loading_mode: str = "branch"  # "resume" continues counters; "branch" carries progress; "init" loads weights only

    @property
    def bank_selection_seed(self) -> int:
        return self.seed if self.map_selection_seed is None else self.map_selection_seed


# network
@dataclass(frozen=True)
class NetworkConfig:
    hidden_dim: int = 256
    num_layers: int = 3
    actor_num_layers: int = 3


    tarmac_sig_dim: int = 16
    tarmac_val_dim: int = 32


# evaluation
@dataclass(frozen=True)
class EarlyStoppingConfig:
    enabled: bool = False
    success_rate: float = 0.99
    final_success_rate: float | None = None
    min_success_length_reduction: float | None = None
    hold_evals: int = 0


@dataclass(frozen=True)
class EvaluationConfig:
    seed: int | None = None  # None follows training.seed; explicit value fixes inspection cases across training seeds.
    inspection_maps: int | None = None  # Random inspection subset; authored suites use their configured maps.
    num_agents: int | None = None  # None (YAML: null) uses env.num_agents from training; otherwise deploy this many for all evals/replays.
    episodes_per_map: int | None = 4000  # null follows the effective training.num_envs.
    eval_differes_from_training_map: bool = False
    eval_map: str | None = None
    # Optional handcrafted [min_x,min_y,min_z,max_x,max_y,max_z] cuboids.

    random_eval: bool = False
    random_eval_maps: tuple[str, ...] | None = None
    replay_targets_from_storey: int | None = None  # 1-based; None samples every valid storey.
    target_sampling: str = "independent"  # independent | training (adaptive weights) | curriculum (uniform, current budget).

    success_condition: str | None = None  # null follows env; also accepts chain_held_and_cleand_up
    eval_broadcast_on_curriculum_early_stop: bool = False

    save_model: bool = True
    checkpoint_freq: int = 50
    checkpoint_offset: int = 0
    checkpoint_dir: str | None = None
    eval_freq: int = 20
    eval_offset: int = 1
    eval_min_train_success: float = 0.0

    training_heatmap_creation: bool = False
    final_artifacts_only: bool = False # Keep intermediate metrics; create heatmaps/replays at final checkpoint only.

    eval_video: bool = True
    eval_video_freq: int = 20
    eval_video_offset: int = 1
    map_suite_replay_limit: int | None = None  # null replays all suite maps; metrics always evaluate every map.


@dataclass(frozen=True)
class ValidationConfig:
    enabled: bool = False
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()
    mode: str | None = None  # Generated maps use a separate validation split by default.
    success_condition: str | None = None  # null follows env, independently of inspection_evaluation.
    num_maps: int = 400
    num_episodes: int = 10000
    seed: int = 456789
    fixed_targets: bool = True
    parallel_envs: int = 500
    val_freq: int = 100
    val_on_finalization: bool = True
    save_per_map_stats: bool = True
    save_best_checkpoint: bool = True


def validate_validation(cfg: ValidationConfig) -> None:
    if cfg.success_condition is not None and (
            not isinstance(cfg.success_condition, str) or cfg.success_condition not in SUCCESS_CONDITIONS):
        raise ValueError(f"validation.success_condition must be one of {sorted(SUCCESS_CONDITIONS)} or null.")
    for name in ("enabled", "fixed_targets", "val_on_finalization", "save_per_map_stats", "save_best_checkpoint"):
        if type(getattr(cfg, name)) is not bool:
            raise ValueError(f"validation.{name} must be boolean.")
    if cfg.mode not in (None, "uniform_full_task"):
        raise ValueError("validation.mode must be null or uniform_full_task.")
    if not cfg.enabled:
        return
    for name in ("num_maps", "num_episodes", "parallel_envs", "val_freq"):
        if type(getattr(cfg, name)) is not int or getattr(cfg, name) < 1:
            raise ValueError(f"validation.{name} must be a positive integer.")
    if type(cfg.seed) is not int or not 0 <= cfg.seed < 2**32:
        raise ValueError("validation.seed must be an integer in [0, 2**32).")


def inspection_config_dict(config: EvaluationConfig) -> dict:
    """Write canonical inspection names; legacy names remain readable on input."""
    from dataclasses import asdict
    values = asdict(config)
    values.pop("random_eval")
    return values


def normalize_inspection_sections(data: dict) -> dict:
    """Normalize before CLI merge so overrides through either section name work."""
    data = dict(data)
    legacy = data.pop("evaluation", None)
    current = data.pop("inspection_evaluation", None)
    if legacy is not None and current is not None:
        raise ValueError("Specify inspection_evaluation or its legacy alias evaluation, not both.")
    values = current if current is not None else legacy
    if values is not None:
        if not isinstance(values, dict):
            raise ValueError("inspection_evaluation must be a mapping.")
        values = dict(values)
        data["evaluation"] = values
    return data


# logging
@dataclass(frozen=True)
class LoggingConfig:
    run_name: str | None = "no_name_provided"
    use_timestamp_postfix: bool = False
    log_dir: str = "outputs"

    wandb_mode: str = "online"
    wandb_project: str = "SwarmEcho"
    wandb_entity: str | None = None
    wandb_group: str | None = None

    # Reporting only; adaptive controller state is always retained in checkpoints.
    adaptive_cell_diagnostics: bool = True
    individual_map_diagnostics: bool = True

    suppress_xla_warnings: bool = True
    terminal_logging_frequency: int = 1
    terminal_log_warmup: bool = False
    terminal_log_init: bool = False
    # Independent completed-episode window for each exact target estimator count.
    required_agents_metrics_window_episodes: int = 500


@dataclass(frozen=True)
class Level:
    name: str
    map_names: list[str]
    building: object
    env: EnvConfig
    reward: RewardConfig
    training: TrainingConfig
    network: NetworkConfig
    evaluation: EvaluationConfig
    logging: LoggingConfig
    validation: ValidationConfig = ValidationConfig()
    random_buildings: RandomBuildingConfig = RandomBuildingConfig()
    map_paths: tuple[Path, ...] = ()
    map_source_name: str | None = None

    @property
    def decommissioning(self) -> DecommissioningConfig:
        return self.env.decommissioning

    @property
    def inspection_evaluation(self) -> EvaluationConfig:
        return self.evaluation

    @property
    def training_map_count(self) -> int:
        return self.training.num_envs if self.training.num_maps is None else self.training.num_maps

    @property
    def building_name(self) -> str:
        return self.map_names[0]

    @property
    def static_map_suite(self) -> bool:
        return not self.random_buildings.enabled and len(self.map_paths) > 1

    @property
    def ideal_chain_margin_m(self) -> float:
        return maximum_chain_distance(self.env) - self.building.max_base_to_top_corner_m

    @property
    def num_updates(self) -> int:
        return self.training.total_timesteps // (self.training.num_envs * self.training.num_steps)

    @property
    def effective_episodes_per_map(self) -> int:
        value = self.evaluation.episodes_per_map
        return self.training.num_envs if value is None else value


def maximum_chain_distance(cfg: EnvConfig) -> float:
    """Ideal straight-line reach; config loading must not initialize JAX."""
    return cfg.comm_radius_base + max(0, cfg.num_agents - 1) * cfg.comm_radius + cfg.visual_radius


def resolve_validation_level(level: Level) -> Level:
    """Apply the validation goal without inheriting inspection-only overrides."""
    if level.validation.success_condition is not None:
        level = replace(level, env=replace(level.env, success_condition=level.validation.success_condition))
    validate_decommissioning(level.env)
    return level


def resolve_evaluation_level(level: Level) -> Level:
    """Apply evaluation-only agent, goal, and map overrides without changing training."""
    evaluation = level.evaluation
    if evaluation.seed is not None:
        level = replace(level, training=replace(level.training, seed=evaluation.seed))
    num_agents = getattr(evaluation, "num_agents", None)
    if num_agents is not None:
        level = replace(level, env=replace(level.env, num_agents=num_agents))
    if evaluation.success_condition is not None:
        level = replace(level, env=replace(level.env, success_condition=evaluation.success_condition))
    validate_decommissioning(level.env)
    if not evaluation.eval_differes_from_training_map:
        return level
    if evaluation.eval_map is None:
        warnings.warn(
            "evaluation.eval_differes_from_training_map is enabled but "
            "evaluation.eval_map is unset; evaluation will use the training map.",
            RuntimeWarning,
            stacklevel=2,
        )
        return level

    from swarmecho.env.buildings import load_building

    source = resolve_map_file(evaluation.eval_map, MAP_DIR)
    if not source.exists():
        raise FileNotFoundError(
            f"Evaluation map not found: {evaluation.eval_map!r}. "
            f"Expected a map name in {MAP_DIR} or an existing path."
        )
    map_name = canonical_map_name(source.stem)
    if map_name == canonical_map_name(Path(level.building_name).stem):
        return level
    return replace(
        level,
        map_names=[map_name],
        building=load_building(source),
    )


def _strict_dataclass(cls, values: object, label: str):
    if not isinstance(values, dict):
        raise ValueError(f"{label} must be a mapping.")
    unknown = set(values) - set(cls.__dataclass_fields__)
    if unknown:
        if cls is RewardConfig:
            raise ValueError(
                f"Unsupported reward fields: {', '.join(sorted(unknown))}. "
                f"See {DEFAULT_LEVEL_NAME} for the current signed reward fields."
            )
        raise ValueError(f"Unknown {label} fields: {', '.join(sorted(unknown))}.")
    return cls(**values)


def normalize_renamed_settings(data: dict) -> dict:
    """Translate historical config keys before merging CLI overrides."""
    from copy import deepcopy

    data = normalize_inspection_sections(deepcopy(data))

    def rename(values, aliases):
        for old, new in aliases.items():
            if old in values:
                value = values.pop(old)
                if new in values and values[new] != value:
                    raise ValueError(f"Conflicting settings: {old} and {new}.")
                values[new] = value

    rename(data, {"adapt_size": "decommissioning"})
    env = data.get("env", {})
    rename(env, {"adapt_size": "decommissioning"})
    if "decommissioning" in env:
        data["decommissioning"] = {**env.pop("decommissioning"), **data.get("decommissioning", {})}
    if env.get("radar_mode") == "legacy":
        env["radar_mode"] = "max_proximity"
    reward = data.get("reward", {})
    rename(reward, REWARD_NAME_ALIASES)
    adaptive = data.get("training", {}).get("adaptive_target_spawn", {})
    rename(adaptive, {"curriculum_start_agents": "initial_target_agent_budget"})
    inspection = data.get("evaluation", {})
    rename(inspection, {"eval_parallel_envs": "episodes_per_map",
                        "random_eval_envs": "inspection_maps"})
    stopping = {new: inspection.pop(old) for old, new in {
        "early_exit": "enabled", "early_exit_success_rate": "success_rate",
        "final_early_exit_success_rate": "final_success_rate",
        "early_exit_min_success_length_reduction": "min_success_length_reduction",
        "early_exit_hold_evals": "hold_evals",
    }.items() if old in inspection}
    if stopping:
        validation = data.setdefault("validation", {})
        current = validation.get("early_stopping", {})
        for name, value in stopping.items():
            if name in current and current[name] != value:
                raise ValueError(f"Conflicting validation early-stopping setting: {name}.")
        validation["early_stopping"] = {**stopping, **current}
    return data


def validate_early_stopping(cfg: EarlyStoppingConfig) -> None:
    if type(cfg.enabled) is not bool:
        raise ValueError("validation.early_stopping.enabled must be boolean.")
    for name in ("success_rate", "final_success_rate", "min_success_length_reduction"):
        value = getattr(cfg, name)
        if value is not None and (type(value) not in (int, float)
                                  or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"validation.early_stopping.{name} must be in [0, 1] or null.")
    if cfg.success_rate is None:
        raise ValueError("validation.early_stopping.success_rate is required.")
    if type(cfg.hold_evals) is not int or cfg.hold_evals < 0:
        raise ValueError("validation.early_stopping.hold_evals must be nonnegative.")


def load_level(name_or_path: str | Path = DEFAULT_LEVEL_NAME, overrides: list[str] | None = None,
               *, config_data: dict | None = None) -> Level:
    """Load a strict level through the canonical config module."""
    from swarmecho.env.buildings import load_building

    if config_data is None or overrides:
        from omegaconf import OmegaConf

    source = resolve_level_file(name_or_path, LEVEL_DIR)
    saved_config = config_data is not None
    if not saved_config:
        if not source.exists():
            raise FileNotFoundError(f"level not found: {name_or_path}")
        config_data = OmegaConf.to_container(OmegaConf.load(source), resolve=True)
    data = normalize_renamed_settings(config_data)
    if saved_config:
        data.pop("name", None)
        data.pop("building", None)
        inspection = data.get("evaluation", {})
        for key in ("training_robustness", "final_robustness", "eval_robustness_runs", "eval_action_noise_max",
                    "eval_every_env_steps", "eval_not_deliv_not_visual_splitt_in_two", "early_exit_source",
                    "apply_trainings_min_geo_separation", "minimum_geodesic_separation", "dynamic_max_geo_distance",
                    "eval_fixed_obstacle_bounds"):
            inspection.pop(key, None)
        if isinstance(data.get("validation"), dict):
            validation_snapshot = dict(data["validation"])
            validation_snapshot.pop("val_every_env_steps", None)
            if validation_snapshot.get("mode") == "held_out_maps":
                validation_snapshot.pop("mode")
            data["validation"] = validation_snapshot
        if isinstance(data.get("network"), dict):
            network_snapshot = dict(data["network"])
            for key, value in (("critic_type", "observation"), ("actor_memory", True), ("critic_memory", True),
                               ("memory_comm_enabled", True), ("memory_comm_every_k_steps", 1),
                               ("tarmac_include_self", False)):
                if network_snapshot.get(key) == value:
                    network_snapshot.pop(key)
            data["network"] = network_snapshot
    if overrides:
        override_data = normalize_renamed_settings(OmegaConf.to_container(OmegaConf.from_dotlist(overrides), resolve=True))
        data = OmegaConf.to_container(OmegaConf.merge(OmegaConf.create(data), OmegaConf.create(override_data)), resolve=True)
    allowed = {"env", "reward", "training", "network", "evaluation", "logging", "random_buildings", "decommissioning", "validation"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown config sections: {', '.join(sorted(unknown))}.")
    env_data = dict(data.get("env", {}))
    if saved_config:
        for key in ("observe_target_vector", "observe_seeing_target", "observe_base_vector",
                    "observe_current_timestep", "observe_coverage_probe", "observe_base_vote_ratio",
                    "base_keeps_informing"):
            if env_data.get(key) is False:
                env_data.pop(key)
        if env_data.get("roadmap_approach") == "full":
            env_data.pop("roadmap_approach")
        if env_data.get("wall_sensing_layout") == "fibonacci":
            env_data.pop("wall_sensing_layout")
        env_data.pop("no_movement_termination_steps", None)
        env_data.pop("relay_agent_budget", None)
        for key in ("num_obstacles", "obstacle_size_min_m", "obstacle_size_max_m",
                    "obstacle_spawn_layer_min", "obstacle_spawn_layer_max", "obstacle_boundary_buffer_m",
                    "obstacle_target_buffer_m", "roadmap_corner_bonus_m"):
            env_data.pop(key, None)
    adapt_data = {**env_data.pop("decommissioning", {}), **data.get("decommissioning", {})}
    if saved_config:
        for key in ("initial_agents", "vote_mode", "vote_protocol", "vote_holding", "quorum_fraction",
                    "quorum_hold", "reward_by_reference_enabled", "reward_by_reference_fraction",
                    "reward_by_reference_steps", "recruitment_enabled"):
            adapt_data.pop(key, None)
    env_data["decommissioning"] = _strict_dataclass(DecommissioningConfig, adapt_data, "decommissioning")
    # Preserve frozen run configs, including the old coupling to visual_radius.
    # Explicit current fields take precedence over renamed/combined counts.
    legacy_wall_sensing = "wall_radar_bins" in env_data or "radar_bins" in env_data
    if "wall_radar_bins" in env_data:
        env_data.setdefault("wall_sensing_rays", env_data.pop("wall_radar_bins"))
    if "radar_bins" in env_data:
        old_bins = env_data.pop("radar_bins")
        env_data.setdefault("wall_sensing_rays", old_bins)
        env_data.setdefault("drone_radar_bins", old_bins)
    if legacy_wall_sensing:
        env_data.setdefault("wall_sensing_ray_max_distance", env_data.get("visual_radius", EnvConfig.visual_radius))
    map_names = env_data.pop("map_names", None)
    generation = _strict_dataclass(RandomBuildingConfig, data.get("random_buildings", {}), "random_buildings")
    validation_data = dict(data.get("validation", {}))
    validation_data["early_stopping"] = _strict_dataclass(
        EarlyStoppingConfig, validation_data.get("early_stopping", {}), "validation.early_stopping")
    validate_early_stopping(validation_data["early_stopping"])
    validation = _strict_dataclass(ValidationConfig, validation_data, "validation")
    validate_validation(validation)
    env = _strict_dataclass(EnvConfig, env_data, "env")
    validate_decommissioning(env)
    validate_radar_config(env)
    training_data = dict(data.get("training", {}))
    if saved_config:
        training_data.pop("agent_count_curriculum", None)
        for key in ("base_memory_replay", "entropy_mode", "value_normalization", "training_noise", "noise_level",
                    "minimum_geodesic_separation", "minimum_geodesic_separation_multiplier", "skip_on_no_pair_found"):
            training_data.pop(key, None)
    if saved_config and "target_curriculum" in training_data:
        legacy_curriculum = training_data.pop("target_curriculum")
        mode = legacy_curriculum.get("mode", "none")
        if mode not in ("none", "forward") or (mode == "forward" and (
                legacy_curriculum.get("start_agents", 2) != 2
                or legacy_curriculum.get("success_threshold", 0.75) != 0.75)):
            raise ValueError("Saved target curriculum uses an unsupported schedule.")
        training_data.setdefault("forward_target_curriculum", {"enabled": mode == "forward"})
    geo = dict(training_data.get("geo_probe_auxiliary", {}))
    if saved_config and geo.get("observe_prev_path_probe") is False:
        geo.pop("observe_prev_path_probe")
    if saved_config:
        geo.pop("enabled", None)
        geo.pop("grid_shape", None)
    head = dict(geo.get("head", {}))
    if "hidden_sizes" in head:
        head["hidden_sizes"] = tuple(head["hidden_sizes"])
    for key, cls in (("head", GeoAuxHeadConfig), ("sampling", GeoAuxSamplingConfig), ("schedule", GeoAuxScheduleConfig)):
        geo[key] = _strict_dataclass(cls, head if key == "head" else geo.get(key, {}), "training.geo_probe_auxiliary." + key)
    training_data["geo_probe_auxiliary"] = _strict_dataclass(GeoProbeAuxiliaryConfig, geo, "training.geo_probe_auxiliary")
    training_data["adaptive_target_spawn"] = _strict_dataclass(
        AdaptiveTargetSpawnConfig, training_data.get("adaptive_target_spawn", {}),
        "training.adaptive_target_spawn",
    )
    validate_adaptive_target_spawn(training_data["adaptive_target_spawn"])
    training_data["forward_target_curriculum"] = _strict_dataclass(
        ForwardTargetCurriculumConfig, training_data.get("forward_target_curriculum", {}),
        "training.forward_target_curriculum",
    )
    if type(training_data["forward_target_curriculum"].enabled) is not bool:
        raise ValueError("training.forward_target_curriculum.enabled must be boolean.")
    training_data["target_spawn_eligibility"] = _strict_dataclass(
        TargetSpawnEligibilityConfig, training_data.get("target_spawn_eligibility", {}), "training.target_spawn_eligibility")
    validate_target_spawn_eligibility(training_data["target_spawn_eligibility"])
    map_sampling_data = training_data.get("adaptive_map_sampling", {})
    if isinstance(map_sampling_data, dict):
        map_sampling_data = dict(map_sampling_data)
        # Legacy saved configs used a separate warmup threshold. Window size
        # now always determines readiness, irrespective of the old value.
        map_sampling_data.pop("warmup_episodes_per_map", None)
        if validation.enabled:
            # Validation owns the holdout; an unused legacy count must not affect resume identity.
            map_sampling_data.pop("held_out_maps", None)
    training_data["adaptive_map_sampling"] = _strict_dataclass(
        AdaptiveMapSamplingConfig, map_sampling_data,
        "training.adaptive_map_sampling")
    validate_adaptive_map_sampling(training_data["adaptive_map_sampling"])
    if saved_config and isinstance(training_data.get("dynamic_max_geo_distance"), dict):
        relay_snapshot = dict(training_data["dynamic_max_geo_distance"])
        relay_snapshot.pop("mode", None)
        relay_snapshot.pop("enabled", None)
        training_data["dynamic_max_geo_distance"] = relay_snapshot
    training_data["dynamic_max_geo_distance"] = _strict_dataclass(
        DynamicMaxGeoDistanceConfig, training_data.get("dynamic_max_geo_distance", {}), "training.dynamic_max_geo_distance")
    training_data["oom_recovery"] = _strict_dataclass(
        OOMRecoveryConfig, training_data.get("oom_recovery", {}), "training.oom_recovery")
    training = _strict_dataclass(TrainingConfig, training_data, "training")
    geo = training.geo_probe_auxiliary
    if type(geo.gradient_diagnostics) is not bool:
        raise ValueError("Geo auxiliary gradient_diagnostics must be boolean.")
    if not math.isfinite(geo.lambda_target) or geo.lambda_target < 0 or not 0 <= geo.metric_threshold <= 1:
        raise ValueError("Invalid geo auxiliary lambda/metric threshold.")
    if (not geo.head.hidden_sizes or any(type(n) is not int or n < 1 for n in geo.head.hidden_sizes)
            or not math.isfinite(geo.head.learning_rate) or geo.head.learning_rate <= 0):
        raise ValueError("Invalid geo auxiliary head configuration.")
    for value in (geo.sampling.every_updates, geo.sampling.snapshots_per_agent, training.optimizer_microbatches):
        if type(value) is not int or value < 1:
            raise ValueError("Geo sampling counts and optimizer_microbatches must be positive integers.")
    if geo.sampling.selection != "stratified_eligible_steps":
        raise ValueError("Unsupported geo auxiliary sampling selection.")
    if any(type(n) is not int or n < 0 for n in (geo.schedule.warmup_updates, geo.schedule.ramp_updates)):
        raise ValueError("Geo auxiliary warm-up/ramp must be nonnegative integers.")
    if training.optimizer_microbatches > 1 and (training.num_envs % training.num_minibatches or (training.num_envs // training.num_minibatches) % training.optimizer_microbatches):
        raise ValueError("PPO minibatches must divide environments and optimizer_microbatches must divide each minibatch.")
    if training.rollout_batch_size is not None and (type(training.rollout_batch_size) is not int or training.rollout_batch_size < 1):
        raise ValueError("training.rollout_batch_size must be null or positive.")
    validate_oom_recovery(training.oom_recovery)
    if type(training.seed_map_pool) is not bool:
        raise ValueError("training.seed_map_pool must be boolean.")
    if training.seed_map_pool and (
            not generation.enabled or not generation.load_maps_from_bank
            or type(training.seed) is not int or not 0 <= training.seed < 2**32):
        raise ValueError("training.seed_map_pool requires a persistent random-building bank and "
                         "training.seed in [0, 2**32).")
    if training.map_selection_seed is not None and (
            type(training.map_selection_seed) is not int or not 0 <= training.map_selection_seed < 2**32):
        raise ValueError("training.map_selection_seed must be null or an integer in [0, 2**32).")
    if training.map_pool_path is not None and (
            not isinstance(training.map_pool_path, str) or not training.map_pool_path.strip()
            or not generation.enabled or not generation.load_maps_from_bank):
        raise ValueError("training.map_pool_path requires a nonempty path and a persistent random-building bank.")
    for budget_name in ("max_wall_seconds", "max_training_seconds"):
        budget = getattr(training, budget_name)
        if budget is not None and (
            type(budget) not in (int, float) or not math.isfinite(budget) or budget <= 0
        ):
            raise ValueError(f"training.{budget_name} must be null or a positive finite number.")
    if training.num_maps is not None and (type(training.num_maps) is not int or training.num_maps < 1):
        raise ValueError("training.num_maps must be null or a positive integer.")
    if validation.enabled and validation.mode is None and not generation.enabled:
        raise ValueError("Validation requires persistent random buildings.")
    if validation.enabled and validation.mode == "uniform_full_task" and (
            generation.enabled or len(map_names or []) != 1 or validation.num_maps != 1
            or not validation.fixed_targets):
        raise ValueError("Uniform full-task validation requires one authored map, num_maps=1, "
                         "and fixed_targets=true (solvable targets only).")
    if validation.enabled and training.randomize_base:
        raise ValueError("Validation currently requires fixed bases.")
    if training.num_maps is not None and not generation.enabled:
        if training.num_maps != len(map_names or []):
            raise ValueError("Authored training.num_maps must match the configured map list, or be null.")
    validate_dynamic_max_geo_distance(training.dynamic_max_geo_distance)
    if training.target_spawn_eligibility.enabled:
        if (training.randomize_base
                or (not generation.enabled and len(map_names or []) != 1)):
            raise ValueError("Target spawn eligibility requires fixed-base, certified target tasks on random maps or one authored map.")
    if training.forward_target_curriculum.enabled:
        if (training.adaptive_target_spawn.enabled or training.adaptive_map_sampling.enabled
                or env.num_agents < 2
                or not validation.enabled):
            raise ValueError("Target curriculum requires uniform sampling, full population, dynamic feasibility and validation progression.")
    if training.adaptive_target_spawn.initial_target_agent_budget is not None:
        ev_data = data.get("evaluation", {})
        if ((not generation.enabled and not map_names)
                or training.adaptive_target_spawn.initial_target_agent_budget > env.num_agents
                or not validation.early_stopping.enabled
                or not validation.enabled):
            raise ValueError("Spawn curriculum requires authored or persistent random maps, fixed full population, "
                             "dynamic feasibility and validation-based early exit.")
    if training.adaptive_target_spawn.enabled and training.randomize_base:
        raise ValueError("Adaptive target spawning requires static target maps and a fixed base per map.")
    if training.adaptive_target_spawn.enabled and generation.enabled:
        # Per-map cell support is needed even when map choice is uniform.
        samples = training.adaptive_map_sampling.eligibility_samples_per_map
        if type(samples) is not int or samples < 1:
            raise ValueError("training.adaptive_map_sampling.eligibility_samples_per_map must be a positive integer "
                             "when adaptive target spawning uses random buildings.")
    if training.adaptive_map_sampling.enabled:
        if not generation.enabled and len(map_names or []) < 2:
            raise ValueError("Adaptive map sampling requires at least two static maps or persistent random buildings.")
        if generation.enabled and data.get("evaluation", {}).get("random_eval_maps") is not None:
            raise ValueError("Adaptive random-map sampling reserves its own held-out maps; omit evaluation.random_eval_maps.")
    if generation.load_maps_from_bank and not generation.enabled:
        raise ValueError("random_buildings.load_maps_from_bank requires random_buildings.enabled=true.")
    map_paths: tuple[Path, ...] = ()
    if generation.enabled:
        if saved_config and map_names == ["random_buildings"]:
            map_names = []
        if map_names not in (None, [], ["random"]):
            raise ValueError("random_buildings.enabled requires env.map_names: [] (or [random]); authored maps cannot be specified.")
        from swarmecho.env.random_buildings import generate_building, map_seed
        _, building = generate_building(generation, map_seed(training.seed, 0), name="map_0000",
                                        drone_clearance=env.drone_radius + env.obstacle_planning_clearance_m,
                                        env_cfg=env)
        map_names = ["random_buildings"]
    else:
        if not isinstance(map_names, list) or not map_names or any(
            not isinstance(name, str) or not name.strip() for name in map_names
        ):
            raise ValueError("env.map_names must contain at least one map name or YAML path.")
        paths = []
        for name in map_names:
            requested = Path(name)
            local = MAP_DIR / requested
            path = local if local.is_file() else resolve_map_file(name, MAP_DIR)
            if not path.is_file():
                raise FileNotFoundError(f"Map not found: {name!r} (resolved to {path}).")
            paths.append(path.resolve())
        if len(set(paths)) != len(paths):
            raise ValueError("env.map_names cannot contain the same map more than once.")
        map_paths = tuple(paths)
        building = load_building(map_paths[0])
    reward_data = dict(data.get("reward", {}))
    if saved_config:
        for key in ("decommission_reward", "useless_agent_at_success_reward",
                    "useless_agent_at_success_all_agents_reward", "useless_agent_at_success_all_agents_reward_cap",
                    "no_movement_termination_penalty", "success_reward_as_hold_record", "peer_informing_reward_enabled",
                    "peer_informing_reward", "gap_reward_uses_change", "gap_change_reward_per_meter",
                    "delivery_gap_vesting_enabled", "delivery_gap_vesting_steps", "priveleged_vote_reward",
                    "reinforcement_reward", "rejected_call_reward", "target_found_requires_delivery",
                    "chain_reward_system", "progress_based_cluster_tiebreaker_soft", "allow_redundancy_reward",
                    "enable_chain_efficiency_reward"):
            reward_data.pop(key, None)
    level = Level(
        name=canonical_level_name(source.stem),
        map_names=(["random_buildings"] if generation.enabled else
                   [canonical_map_name(Path(name).stem) for name in map_names]),
        building=building,
        env=env,
        reward=_strict_dataclass(RewardConfig, reward_data, "reward"),
        training=training,
        network=_strict_dataclass(NetworkConfig, data.get("network", {}), "network"),
        evaluation=_strict_dataclass(EvaluationConfig, data.get("evaluation", {}), "evaluation"),
        logging=_strict_dataclass(LoggingConfig, data.get("logging", {}), "logging"),
        validation=validation,
        random_buildings=generation,
        map_paths=map_paths,
    )
    if generation.enabled:
        count = level.evaluation.inspection_maps
        if count is None:
            count = (training.adaptive_map_sampling.held_out_maps if training.adaptive_map_sampling.enabled
                     else 5)
        if type(count) is not int or count < 1:
            raise ValueError("inspection_evaluation.inspection_maps must be null or a positive integer.")
        if validation.enabled and count > validation.num_maps:
            raise ValueError("Inspection map count cannot exceed validation.num_maps.")
        level = replace(level, evaluation=replace(level.evaluation, random_eval=True,
            inspection_maps=count, random_eval_maps=None if validation.enabled or training.adaptive_map_sampling.enabled
            else level.evaluation.random_eval_maps))
    if level.evaluation.inspection_maps is not None and (
            type(level.evaluation.inspection_maps) is not int or level.evaluation.inspection_maps < 1):
        raise ValueError("inspection_evaluation.inspection_maps must be null or a positive integer.")
    if validation.early_stopping.enabled and not validation.enabled:
        raise ValueError("Validation early exit requires validation.enabled=true.")
    validate_reward_config(level.reward)
    success_conditions = SUCCESS_CONDITIONS
    if level.env.success_condition not in success_conditions:
        raise ValueError(f"env.success_condition must be one of {sorted(success_conditions)}.")
    if (level.evaluation.success_condition is not None
            and level.evaluation.success_condition not in success_conditions):
        raise ValueError(f"evaluation.success_condition must be one of {sorted(success_conditions)} or null.")
    if validation.enabled:
        resolve_validation_level(level)
    if level.ideal_chain_margin_m < 0:
        raise ValueError(f"level {level.name!r} is geometrically unsolvable: ideal chain margin is {level.ideal_chain_margin_m:.3f} m.")
    if level.training.num_epochs < 1 or level.training.num_minibatches < 1:
        raise ValueError("PPO epoch and minibatch counts must be positive.")
    if level.training.num_envs % level.training.num_minibatches:
        raise ValueError("Recurrent training requires num_envs divisible by num_minibatches.")
    if level.training.actor_clip_eps <= 0:
        raise ValueError("training.actor_clip_eps must be positive.")
    if not math.isfinite(level.training.value_clip_eps) or level.training.value_clip_eps < 0:
        raise ValueError("training.value_clip_eps must be finite and nonnegative.")
    parallel_envs = level.evaluation.episodes_per_map
    if parallel_envs is not None and (type(parallel_envs) is not int or parallel_envs < 1):
        raise ValueError("evaluation.episodes_per_map must be null or a positive integer.")
    recovery = level.training.oom_recovery
    if (recovery.enabled and recovery.parameter == "num_envs" and recovery.factor is None
            and recovery.decrement % level.training.num_minibatches):
        raise ValueError("OOM recovery requires num_minibatches to divide the configured environment decrement.")
    ev = level.evaluation
    if ev.random_eval and ev.inspection_maps is None:
        ev = replace(ev, inspection_maps=5)
        level = replace(level, evaluation=ev)
    if ev.num_agents is not None and (type(ev.num_agents) is not int or ev.num_agents < 1):
        raise ValueError("evaluation.num_agents must be null or a positive integer.")
    if level.static_map_suite:
        if ev.random_eval or ev.random_eval_maps is not None:
            raise ValueError("Static multi-map evaluation uses env.map_names; disable random_eval and random_eval_maps.")
        if ev.eval_differes_from_training_map or ev.eval_map:
            raise ValueError("Static multi-map evaluation cannot use evaluation map overrides.")
    if ev.target_sampling not in {"independent", "training", "curriculum"}:
        raise ValueError("inspection_evaluation.target_sampling must be independent, training or curriculum.")
    if ev.target_sampling == "curriculum":
        if (not training.adaptive_target_spawn.enabled
                or training.adaptive_target_spawn.initial_target_agent_budget is None
                or ev.num_agents is not None or ev.eval_differes_from_training_map or ev.eval_map
                or not validation.enabled):
            raise ValueError("Curriculum inspection requires a virtual target curriculum, the full training "
                             "population, dynamic feasibility, deterministic inspection and full-task validation.")
    if ev.target_sampling == "training":
        if (generation.enabled or len(map_names or []) != 1 or not training.adaptive_target_spawn.enabled
                or ev.random_eval or ev.eval_differes_from_training_map or ev.eval_map
                or ev.num_agents is not None or ev.success_condition not in (None, env.success_condition)):
            raise ValueError("Training target sampling requires one authored training map with adaptive target "
                             "spawning and the training population, task and feasibility filter. "
                             "Inspection maps cannot reuse another map's adaptive cell weights.")
    if ev.seed is not None and (type(ev.seed) is not int or not 0 <= ev.seed < 2**32):
        raise ValueError("inspection_evaluation.seed must be null or an integer in [0, 2**32).")
    if type(ev.random_eval) is not bool:
        raise ValueError("random_eval must be boolean.")
    if ev.replay_targets_from_storey is not None and (
        type(ev.replay_targets_from_storey) is not int or ev.replay_targets_from_storey < 1
    ):
        raise ValueError("evaluation.replay_targets_from_storey must be null or a positive integer.")
    if ev.random_eval:
        if ev.eval_differes_from_training_map or ev.eval_map:
            raise ValueError("random_eval cannot be combined with eval_map overrides.")
        if ev.random_eval_maps is not None:
            if not isinstance(ev.random_eval_maps, (list, tuple)) or len(ev.random_eval_maps) != (ev.inspection_maps or 5):
                raise ValueError("random_eval_maps must contain exactly inspection_maps map paths.")
            for item in ev.random_eval_maps:
                if not isinstance(item, str):
                    raise ValueError("random_eval_maps entries must be map names or paths.")
                # Resolve during suite initialization, which first checks the
                # run's frozen copy. Replays must survive source-map removal.
        else:
            from swarmecho.env.random_buildings import random_building_grid
            random_building_grid(generation)
    if type(level.logging.terminal_logging_frequency) is not int or level.logging.terminal_logging_frequency < 1:
        raise ValueError("logging.terminal_logging_frequency must be a positive integer.")
    if type(level.logging.terminal_log_warmup) is not bool:
        raise ValueError("logging.terminal_log_warmup must be a boolean.")
    if type(level.logging.terminal_log_init) is not bool:
        raise ValueError("logging.terminal_log_init must be a boolean.")
    for name in ("eval_freq", "eval_video_freq", "checkpoint_freq"):
        value = getattr(level.evaluation, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"evaluation.{name} must be a positive integer.")
    if type(level.evaluation.final_artifacts_only) is not bool:
        raise ValueError("inspection_evaluation.final_artifacts_only must be boolean.")
    for name in ("eval_offset", "eval_video_offset", "checkpoint_offset"):
        value = getattr(level.evaluation, name)
        if type(value) is not int or value < 0:
            raise ValueError(f"evaluation.{name} must be a non-negative integer.")
    if not 0.0 <= level.evaluation.eval_min_train_success <= 1.0:
        raise ValueError("evaluation.eval_min_train_success must be in [0, 1].")
    if ev.map_suite_replay_limit is not None and (
        type(ev.map_suite_replay_limit) is not int or ev.map_suite_replay_limit < 1
    ):
        raise ValueError("evaluation.map_suite_replay_limit must be null or a positive integer.")
    if level.num_updates < 1:
        raise ValueError("training.total_timesteps must cover at least one rollout.")
    if level.logging.wandb_mode not in {"disabled", "offline", "online"}:
        raise ValueError("logging.wandb_mode must be disabled, offline, or online.")
    for name in ("adaptive_cell_diagnostics", "individual_map_diagnostics"):
        if type(getattr(level.logging, name)) is not bool:
            raise ValueError(f"logging.{name} must be a boolean.")
    return level


def load_level_cli(arguments: list[str]) -> Level:
    """Load a level with the same key=value CLI contract as training."""
    level_name = DEFAULT_LEVEL_NAME
    overrides = []
    for argument in arguments:
        if "=" not in argument:
            raise ValueError(f"Unexpected argument {argument!r}; use key=value overrides.")
        key, value = argument.split("=", 1)
        if key == "level":
            level_name = value
        else:
            overrides.append(argument)
    return load_level(level_name, overrides)
