"""User-facing level settings, defaults, and key=value CLI overrides."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import math
import warnings
from swarmecho.core.compatibility import canonical_level_name, canonical_map_name, resolve_level_file, resolve_map_file

_SRC_ROOT = Path(__file__).resolve().parent.parent
MAP_DIR = _SRC_ROOT / "curriculum_config" / "maps"
LEVEL_DIR = _SRC_ROOT / "curriculum_config" / "levels"

# Fixed across maps, population stages and available slots; not a deployment cap.
DEPLOYED_AGENTS_OBS_REF = 10.0

# Adaptive population control is opt-in; disabled levels retain their old shapes.
@dataclass(frozen=True)
class AdaptSizeConfig:
    enabled: bool = False
    initial_agents: int = 1
    vote_mode: str = "hold"  # hold | keep_yes_no
    vote_protocol: str = "credits"  # credits | latest_decision
    vote_holding: int = 1  # Delivered YES credits; latest_decision always needs one YES.
    quorum_fraction: float = 0.49  # Strictly greater below 1; 1 means unanimity.
    quorum_hold: int = 0  # Consecutive environment steps with sufficient votes.
    decommission_radius_m: float = 1.0
    decommission_hold_steps: int = 15
    # Each qualified caller receives a decaying share of the recruit's normal reward.
    reward_by_reference_enabled: bool = False
    reward_by_reference_fraction: float = 0.10
    reward_by_reference_steps: int = 200

    @property
    def approval_threshold(self) -> int:
        return 1 if self.vote_protocol == "latest_decision" else self.vote_holding


# env
@dataclass(frozen=True)
class EnvConfig:
    num_agents: int = 5
    adapt_size: AdaptSizeConfig = AdaptSizeConfig()

    dt: float = 0.1  # Simulated time per step; scales velocity and position updates.
    max_force: float = 15.0
    max_speed: float = 5.0
    drag: float = 0.85
    drone_radius: float = 0.25

    comm_radius_base: float = 6.0
    base_keeps_informing: bool = False
    comm_radius: float = 5.0
    visual_radius: float = 4.0
    wall_sensing_layout: str = "fibonacci"  # fibonacci (legacy) | xyz
    wall_sensing_rays: int = 16  # fibonacci: any integer >= 4; xyz: 14 | 18 | 26
    wall_sensing_ray_max_distance: float = 8.0  # Metres; independent of target visual_radius.
    drone_radar_bins: int = 8
    radar_mode: str = "legacy"  # legacy | band_sep_counting
    distance_bands: int = 1  # Equal-width bands; 0 is an alias for 1.
    radar_count_cap: int = 3  # Saturation count in band_sep_counting mode.

    target_wall_buffer_fraction: float = 0.1
    spawn_delay: int = 5

    max_steps: int = 700
    hold_chain_for: int = 50
    no_movement_termination_steps: int = 1000
    movement_epsilon: float = 1e-3
    success_condition: str = "chain_held"  # coverage | discovery | delivery | chain_held

    # Debugging-only legacy observation; never treat as a design decision or candidate.
    observe_target_vector: bool = False
    observe_seeing_target: bool = False  # Current direct visual sighting, not remembered knowledge.
    observe_visually_gated_target_displacement: bool = False  # Visible target XYZ offset / visual_radius.
    # Debugging-only legacy observation; never treat as a design decision or candidate.
    observe_base_vector: bool = False
    observe_chain_contributor: bool = False
    observe_base_vote_ratio: bool = False  # Qualified base approvals / active population, while base-connected.
    observe_base_population: bool = False  # Active population / DEPLOYED_AGENTS_OBS_REF, while base-connected.
    observe_current_timestep: bool = False  # Episode step / max_steps, in [0, 1].
    # Debugging-only legacy observation; never treat as a design decision or candidate.
    observe_coverage_probe: bool = False
    coverage_voxel_size: float | None = None # None -> use the building’s cell size i.e. 5m

    num_obstacles: int = 0
    obstacle_size_min_m: float = 2.0
    obstacle_size_max_m: float = 4.0
    obstacle_spawn_layer_min: int = 2
    obstacle_spawn_layer_max: int = 5
    obstacle_boundary_buffer_m: float = 0.5
    obstacle_target_buffer_m: float = 0.5
    obstacle_planning_clearance_m: float = 0.1
    obstacle_layout_version: str = "three_aabb_v1"

    roadmap_approach: str = "full"
    roadmap_node_density: float = 1.0
    roadmap_merge_wall_end_nodes: bool = False
    roadmap_corner_bonus_m: float = 1.0
    # Curriculum-generated target budget; null follows the deployable population.
    relay_agent_budget: int | None = None

    @property
    def geometric_agent_budget(self) -> int:
        return self.num_agents if self.relay_agent_budget is None else self.relay_agent_budget


def validate_adapt_size(cfg: EnvConfig) -> None:
    if (cfg.relay_agent_budget is not None and
            (type(cfg.relay_agent_budget) is not int
             or not 1 <= cfg.relay_agent_budget <= cfg.num_agents)):
        raise ValueError("env.relay_agent_budget must be null or an integer in [1, env.num_agents].")
    a = cfg.adapt_size
    if type(a.enabled) is not bool:
        raise ValueError("adapt_size.enabled must be boolean.")
    if type(a.reward_by_reference_enabled) is not bool:
        raise ValueError("adapt_size.reward_by_reference_enabled must be a boolean.")
    fraction = a.reward_by_reference_fraction
    if (type(fraction) not in (int, float) or not math.isfinite(fraction)
            or not 0 <= fraction <= 1):
        raise ValueError("adapt_size.reward_by_reference_fraction must be finite and in [0, 1].")
    if type(a.reward_by_reference_steps) is not int or a.reward_by_reference_steps < 1:
        raise ValueError("adapt_size.reward_by_reference_steps must be a positive integer.")
    if a.reward_by_reference_enabled and not a.enabled:
        raise ValueError("adapt_size.reward_by_reference_enabled requires adapt_size.enabled=true.")
    if type(a.initial_agents) is not int or not 1 <= a.initial_agents <= cfg.num_agents:
        raise ValueError("adapt_size.initial_agents must be in [1, env.num_agents].")
    if a.vote_mode not in {"hold", "keep_yes_no"}:
        raise ValueError("adapt_size.vote_mode must be hold or keep_yes_no.")
    if a.vote_protocol not in {"credits", "latest_decision"}:
        raise ValueError("adapt_size.vote_protocol must be credits or latest_decision.")
    for name in ("vote_holding", "decommission_hold_steps"):
        if type(getattr(a, name)) is not int or getattr(a, name) < 1:
            raise ValueError(f"adapt_size.{name} must be a positive integer.")
    if type(a.quorum_hold) is not int or a.quorum_hold < 0:
        raise ValueError("adapt_size.quorum_hold must be a nonnegative integer.")
    if not math.isfinite(a.quorum_fraction) or not 0 <= a.quorum_fraction <= 1:
        raise ValueError("adapt_size.quorum_fraction must be in [0, 1].")
    if not math.isfinite(a.decommission_radius_m) or a.decommission_radius_m <= 0:
        raise ValueError("adapt_size.decommission_radius_m must be positive and finite.")
    if a.enabled and cfg.success_condition != "chain_held":
        raise ValueError("Adaptive size currently requires success_condition=chain_held.")


def validate_radar_config(cfg: EnvConfig) -> None:
    """Validate radar settings for both loaded levels and direct environments."""
    for name in ("observe_visually_gated_target_displacement",
                 "observe_base_vote_ratio", "observe_base_population"):
        if type(getattr(cfg, name)) is not bool:
            raise ValueError(f"env.{name} must be boolean.")
    for name in ("wall_sensing_rays", "drone_radar_bins"):
        value = getattr(cfg, name)
        if type(value) is not int or value < 4:
            raise ValueError(f"env.{name} must be an integer of at least 4.")
    if cfg.wall_sensing_layout not in {"fibonacci", "xyz"}:
        raise ValueError("env.wall_sensing_layout must be fibonacci or xyz.")
    if cfg.wall_sensing_layout == "xyz" and cfg.wall_sensing_rays not in {14, 18, 26}:
        raise ValueError("env.wall_sensing_rays must be 14, 18, or 26 for the xyz layout.")
    if cfg.radar_mode not in {"legacy", "band_sep_counting"}:
        raise ValueError("env.radar_mode must be legacy or band_sep_counting.")
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

    save_training_maps: bool = False
    load_maps_from_bank: bool = False


# reward
@dataclass(frozen=True)
class PrivelegedVoteRewardConfig:
    # Estimator/observed-chain supervision; never changes the voting protocol.
    enabled: bool = False
    withheld_reinforcement_vote_reward: float = -1.0
    unnecessary_reinforcement_call_reward: float = -200.0


def validate_priveleged_vote_reward(cfg: PrivelegedVoteRewardConfig, env_cfg: EnvConfig | None = None) -> None:
    if not isinstance(cfg, PrivelegedVoteRewardConfig):
        raise ValueError("reward.priveleged_vote_reward must be a PrivelegedVoteRewardConfig.")
    if type(cfg.enabled) is not bool:
        raise ValueError("reward.priveleged_vote_reward.enabled must be boolean.")
    for name in ("withheld_reinforcement_vote_reward", "unnecessary_reinforcement_call_reward"):
        value = getattr(cfg, name)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"reward.priveleged_vote_reward.{name} must be finite (positive, zero or negative).")
    if cfg.enabled and env_cfg is not None and (
        not env_cfg.adapt_size.enabled or env_cfg.adapt_size.vote_protocol != "latest_decision"
    ):
        raise ValueError("Privileged vote rewards require adapt_size.enabled=true and vote_protocol=latest_decision.")


@dataclass(frozen=True)
class RewardConfig:
    # Signed individual amounts: negative values are costs; zero disables a term.
    # Numeric defaults follow G02, using the former four-agent split as reference.
    reinforcement_reward: float = 1.0  # Full amount to each qualified deployment voter.
    priveleged_vote_reward: PrivelegedVoteRewardConfig = PrivelegedVoteRewardConfig()
    decommission_reward: float = 0.0  # Full amount to each removed drone, on removal.
    rejected_call_reward: float = -200.0  # Each qualified YES voter on a rejected batch.
    all_decommissioned_penalty: float = -1000.0  # Full signed amount to each last acting drone.
    useless_agent_at_success_reward: float = -3000.0  # Each active drone outside the selected success chain.
    useless_agent_at_success_all_agents_reward: float = -750.0  # Each active drone per useless drone at terminal chain success; no population divisor.
    useless_agent_at_success_all_agents_reward_cap: float = 2000.0  # Maximum absolute adjustment per active drone, after counting useless drones.
    target_found_requires_delivery: bool = True
    chain_reward_system: str = "euclidean"
    progress_based_cluster_tiebreaker_soft: bool = False

    # Credit every drone on a simple base-target path, including alternate routes.
    allow_redundancy_reward: bool = False
    enable_chain_efficiency_reward: bool = False
    chain_efficiency_reward: float = 0.0  # Per eligible drone times its path efficiency; no population divisor.

    exploration_reward: float = 2.0  # Per new voxel; simultaneous viewers split that voxel's credit.
    finder_reward: float = 100.0  # Each direct base deliverer, or first visual finder without delivery gating.
    target_found_reward: float = 125.0  # Full one-shot amount to every active drone at delivery/discovery.
    success_reward: float = 3000.0  # Full amount to every active drone at first success; no population divisor.
    # Pay each qualifying holder success_reward / hold_chain_for per new record.
    success_reward_as_hold_record: bool = False
    # One newly informed peer's credit is split among its simultaneous senders.
    peer_informing_reward_enabled: bool = False
    peer_informing_reward: float = 50.0  # Per newly informed peer, split among its direct senders.

    collision_reward: float = -0.5  # Per colliding drone per transition.
    time_presure_reward: float = -0.75  # Each acting drone every step, including its removal step.
    # Recurring amount per credited closed metre per step; retain local soft shares.
    gap_progress_reward_per_meter: float = 0.2  # Recurring per credited closed metre each step.
    # Legacy diagnostic only: signed changes in per-drone credit. Retained for
    # debugging historical experiments, not a training design/tuning option.
    gap_reward_uses_change: bool = False
    gap_change_reward_per_meter: float = 3.0  # Legacy diagnostic change-mode coefficient.
    # Legacy diagnostic companion to change mode: vest already credited gap
    # progress after target delivery instead of discarding its gain.
    delivery_gap_vesting_enabled: bool = False
    delivery_gap_vesting_steps: int = 10
    no_movement_termination_penalty: float = 0.0  # Full signed amount to each active drone at idle failure.


def validate_reward_config(cfg: RewardConfig) -> None:
    """All reward coefficients are signed, finite amounts; no implicit negation."""
    validate_priveleged_vote_reward(cfg.priveleged_vote_reward)
    for name in (
        "reinforcement_reward", "decommission_reward", "rejected_call_reward",
        "all_decommissioned_penalty", "useless_agent_at_success_reward",
        "useless_agent_at_success_all_agents_reward",
        "chain_efficiency_reward", "exploration_reward", "finder_reward",
        "target_found_reward", "success_reward", "peer_informing_reward",
        "collision_reward", "time_presure_reward", "gap_progress_reward_per_meter",
        "gap_change_reward_per_meter", "no_movement_termination_penalty",
    ):
        value = getattr(cfg, name)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"reward.{name} must be finite (positive, zero or negative).")
    cap = cfg.useless_agent_at_success_all_agents_reward_cap
    if type(cap) not in (int, float) or not math.isfinite(cap) or cap < 0:
        raise ValueError("reward.useless_agent_at_success_all_agents_reward_cap must be finite and nonnegative.")


# training
# Keep training action perturbations aligned with the default used by the
# standalone robust checkpoint evaluator.
DEFAULT_ACTION_NOISE_LEVEL = 0.011

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
    min_probability_fraction: float = 0.20
    success_window_episodes_per_cell: int = 500
    probability_smoothing_half_life_updates: float = 3.0
    plateau_warning_enabled: bool = True
    plateau_warning_horizon_updates: int = 50
    plateau_warning_improvement_threshold_pp: float = 3.0


def validate_adaptive_target_spawn(cfg: AdaptiveTargetSpawnConfig) -> None:
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
    enabled: bool = False
    mode: str = "simplifier"  # simplifier | global_candidates
    candidate_spacing_fraction: float = 1.0  # Random budget = ceil(world_volume / (cell_size * fraction)^3).
    max_candidates_per_map: int = 4096  # Explicit guard; candidates are never truncated.


def validate_dynamic_max_geo_distance(cfg):
    if not isinstance(cfg.mode, str) or cfg.mode not in {"simplifier", "global_candidates"}:
        raise ValueError("training.dynamic_max_geo_distance.mode must be simplifier or global_candidates.")
    if cfg.mode == "global_candidates":
        value = cfg.candidate_spacing_fraction
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("candidate_spacing_fraction must be finite and positive.")
        if type(cfg.max_candidates_per_map) is not int or cfg.max_candidates_per_map < 1:
            raise ValueError("max_candidates_per_map must be a positive integer.")


@dataclass(frozen=True)
class AgentCountCurriculumConfig:
    enabled: bool = False
    start_agents: int = 2
    additional_agents: int = 0


@dataclass(frozen=True)
class OOMRecoveryConfig:
    enabled: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    total_timesteps: int = 250_000_000
    num_envs: int = 4000
    num_maps: int | None = None  # Persistent random training maps; null retains the legacy lane-count fallback.
    num_steps: int = 100
    num_epochs: int = 4
    num_minibatches: int = 20
    seed: int = 42

    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    actor_clip_eps: float = 0.2
    # Raw value units, or normalized units with value_normalization=running.
    # None (YAML: null) disables value clipping; zero does NOT disable it.
    value_clip_eps: float | None = 0.2
    entropy_mode: str = "squashed"  # legacy or squashed (reparameterized tanh entropy)
    value_normalization: str = "running"  # none or running; checkpointed critic units
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5

    diagnostics_every: int = 1  # PPO updates between before/after diagnostic passes
    profile_timing: bool = False # debug diagnostic

    # Perturb sampled pre-tanh actions before stepping training environments.
    # Evaluation uses its separate eval_action_noise_max configuration.
    training_noise: bool = False
    noise_level: float = DEFAULT_ACTION_NOISE_LEVEL

    randomize_base: bool = False
    minimum_geodesic_separation: bool = False
    minimum_geodesic_separation_multiplier: float = 2.0
    skip_on_no_pair_found: bool = False
    spawn_pair_max_attempts: int = 1024
    adaptive_target_spawn: AdaptiveTargetSpawnConfig = AdaptiveTargetSpawnConfig()
    adaptive_map_sampling: AdaptiveMapSamplingConfig = AdaptiveMapSamplingConfig()
    dynamic_max_geo_distance: DynamicMaxGeoDistanceConfig = DynamicMaxGeoDistanceConfig()
    agent_count_curriculum: AgentCountCurriculumConfig = AgentCountCurriculumConfig()
    oom_recovery: OOMRecoveryConfig = OOMRecoveryConfig()

    checkpoint_path: str | None = None
    checkpoint_step_offset: int | None = None
    ckpt_loading_mode: str = "branch"  # "resume" continues counters; "branch" carries progress; "init" loads weights only


# network
@dataclass(frozen=True)
class NetworkConfig:
    hidden_dim: int = 256
    num_layers: int = 3
    actor_num_layers: int = 3

    actor_memory: bool = True
    critic_memory: bool = True
    critic_type: str = "observation"  # Only observation is implemented; retained for a future critic experiment.

    memory_comm_enabled: bool = True
    memory_comm_every_k_steps: int = 5
    tarmac_sig_dim: int = 16
    tarmac_val_dim: int = 32
    tarmac_include_self: bool = False


# evaluation
@dataclass(frozen=True)
class EvaluationConfig:
    inspection_maps: int | None = None  # Random inspection subset; authored suites use their configured maps.
    num_agents: int | None = None  # None (YAML: null) uses env.num_agents from training; otherwise deploy this many for all evals/replays.
    eval_parallel_envs: int | None = 4000  # null follows the effective training.num_envs.
    eval_differes_from_training_map: bool = False
    eval_map: str | None = None
    # Optional handcrafted [min_x,min_y,min_z,max_x,max_y,max_z] cuboids.
    eval_fixed_obstacle_bounds: tuple[tuple[float, float, float, float, float, float], ...] | None = None

    random_eval: bool = False
    random_eval_envs: int = 5
    random_eval_maps: tuple[str, ...] | None = None
    replay_targets_from_storey: int | None = None  # 1-based; None samples every valid storey.
    apply_trainings_min_geo_separation: bool = False
    minimum_geodesic_separation: bool | None = None  # None follows apply_trainings_min_geo_separation.
    dynamic_max_geo_distance: bool | None = None  # None follows training's toggle.

    # Periodic metrics may use the same ensemble as the optional robust final eval.
    training_robustness: bool = False
    final_robustness: bool = True
    eval_robustness_runs: int = 5
    eval_action_noise_max: float = DEFAULT_ACTION_NOISE_LEVEL

    early_exit: bool = False
    early_exit_source: str = "evaluation"  # evaluation | training (fresh completed-episode windows per map).
    success_condition: str | None = None  # null follows env; coverage | discovery | delivery | chain_held
    early_exit_success_rate: float = 0.99
    final_early_exit_success_rate: float | None = None  # Override only the final agent-count curriculum stage.
    early_exit_min_success_length_reduction: float | None = None  # 0.25 -> mean completion by step 450 of 600
    early_exit_hold_evals: int = 0
    eval_broadcast_on_curriculum_early_stop: bool = False

    save_model: bool = True
    checkpoint_freq: int = 50
    checkpoint_offset: int = 0
    checkpoint_dir: str | None = None
    eval_freq: int = 20
    eval_offset: int = 1
    eval_min_train_success: float = 0.0

    training_heatmap_creation: bool = False

    eval_video: bool = True
    eval_video_freq: int = 20
    eval_video_offset: int = 1
    map_suite_replay_limit: int | None = None  # null replays all suite maps; metrics always evaluate every map.
    eval_not_deliv_not_visual_splitt_in_two: bool = False


@dataclass(frozen=True)
class ValidationConfig:
    enabled: bool = False
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
    for name in ("enabled", "fixed_targets", "val_on_finalization", "save_per_map_stats", "save_best_checkpoint"):
        if type(getattr(cfg, name)) is not bool:
            raise ValueError(f"validation.{name} must be boolean.")
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
    values.pop("random_eval_envs")
    if config.random_eval and config.inspection_maps is None:
        values["inspection_maps"] = config.random_eval_envs
    if values["early_exit_source"] == "evaluation":
        values["early_exit_source"] = "inspection_evaluation"
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
        if "random_eval_envs" in values and "inspection_maps" not in values:
            values["inspection_maps"] = values.pop("random_eval_envs")
        if values.get("early_exit_source") == "inspection_evaluation":
            values["early_exit_source"] = "evaluation"
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
    def adapt_size(self) -> AdaptSizeConfig:
        return self.env.adapt_size

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
    def effective_eval_parallel_envs(self) -> int:
        value = self.evaluation.eval_parallel_envs
        return self.training.num_envs if value is None else value


def maximum_chain_distance(cfg: EnvConfig) -> float:
    """Ideal straight-line reach; config loading must not initialize JAX."""
    return cfg.comm_radius_base + max(0, cfg.num_agents - 1) * cfg.comm_radius + cfg.visual_radius


def resolve_evaluation_level(level: Level) -> Level:
    """Apply evaluation-only agent, goal, and map overrides without changing training."""
    evaluation = level.evaluation
    num_agents = getattr(evaluation, "num_agents", None)
    if num_agents is not None:
        level = replace(level, env=replace(level.env, num_agents=num_agents))
    if evaluation.success_condition is not None:
        level = replace(level, env=replace(level.env, success_condition=evaluation.success_condition))
    validate_adapt_size(level.env)
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


def evaluation_minimum_geodesic_separation(level: Level) -> bool:
    """Select the evaluation target/base separation independently of coverage training."""
    if resolve_evaluation_level(level).env.success_condition == "coverage":
        return False
    override = level.evaluation.minimum_geodesic_separation
    if override is not None:
        return override
    return (level.evaluation.apply_trainings_min_geo_separation
            and level.training.minimum_geodesic_separation)


def evaluation_dynamic_max_geo_distance(level: Level) -> bool:
    override = level.evaluation.dynamic_max_geo_distance
    return level.training.dynamic_max_geo_distance.enabled if override is None else override


def _strict_dataclass(cls, values: object, label: str):
    if not isinstance(values, dict):
        raise ValueError(f"{label} must be a mapping.")
    unknown = set(values) - set(cls.__dataclass_fields__)
    if unknown:
        if cls is RewardConfig:
            raise ValueError(
                f"Unsupported reward fields: {', '.join(sorted(unknown))}. "
                "Use the signed individual reward settings in G02_static or "
                "the commented replacement beneath this level's reward block."
            )
        raise ValueError(f"Unknown {label} fields: {', '.join(sorted(unknown))}.")
    return cls(**values)


def load_level(name_or_path: str | Path = "M00_no_maze_open_cuboid", overrides: list[str] | None = None) -> Level:
    """Load a strict level through the canonical config module."""
    from omegaconf import OmegaConf
    from swarmecho.env.buildings import load_building

    source = resolve_level_file(name_or_path, LEVEL_DIR)
    if not source.exists():
        raise FileNotFoundError(f"level not found: {name_or_path}")
    data = normalize_inspection_sections(OmegaConf.to_container(OmegaConf.load(source), resolve=True))
    if overrides:
        override_data = normalize_inspection_sections(OmegaConf.to_container(OmegaConf.from_dotlist(overrides), resolve=True))
        data = OmegaConf.to_container(OmegaConf.merge(OmegaConf.create(data), OmegaConf.create(override_data)), resolve=True)
    allowed = {"env", "reward", "training", "network", "evaluation", "logging", "random_buildings", "adapt_size", "validation"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown config sections: {', '.join(sorted(unknown))}.")
    env_data = dict(data.get("env", {}))
    nested_adapt = env_data.pop("adapt_size", {})
    env_data["adapt_size"] = _strict_dataclass(
        AdaptSizeConfig, {**nested_adapt, **data.get("adapt_size", {})}, "adapt_size"
    )
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
    validation = _strict_dataclass(ValidationConfig, data.get("validation", {}), "validation")
    validate_validation(validation)
    env = _strict_dataclass(EnvConfig, env_data, "env")
    validate_adapt_size(env)
    validate_radar_config(env)
    training_data = dict(data.get("training", {}))
    training_data["adaptive_target_spawn"] = _strict_dataclass(
        AdaptiveTargetSpawnConfig, training_data.get("adaptive_target_spawn", {}),
        "training.adaptive_target_spawn",
    )
    validate_adaptive_target_spawn(training_data["adaptive_target_spawn"])
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
    for name, cls in (("dynamic_max_geo_distance", DynamicMaxGeoDistanceConfig),
                      ("agent_count_curriculum", AgentCountCurriculumConfig),
                      ("oom_recovery", OOMRecoveryConfig)):
        training_data[name] = _strict_dataclass(cls, training_data.get(name, {}), "training." + name)
        if type(training_data[name].enabled) is not bool:
            raise ValueError(f"training.{name}.enabled must be boolean.")
    training = _strict_dataclass(TrainingConfig, training_data, "training")
    if training.num_maps is not None and (type(training.num_maps) is not int or training.num_maps < 1):
        raise ValueError("training.num_maps must be null or a positive integer.")
    if validation.enabled and not generation.enabled:
        raise ValueError("Validation requires persistent random buildings.")
    if validation.enabled and (training.randomize_base or training.minimum_geodesic_separation):
        raise ValueError("Validation currently requires fixed bases and minimum_geodesic_separation=false.")
    if validation.enabled and env.success_condition == "coverage":
        raise ValueError("The held-out validation pipeline currently requires a target task.")
    if training.num_maps is not None and not generation.enabled:
        if training.num_maps != len(map_names or []):
            raise ValueError("Authored training.num_maps must match the configured map list, or be null.")
    validate_dynamic_max_geo_distance(training.dynamic_max_geo_distance)
    curriculum = training.agent_count_curriculum
    if type(curriculum.start_agents) is not int or curriculum.start_agents < 2:
        raise ValueError("training.agent_count_curriculum.start_agents must be an integer >= 2.")
    if type(curriculum.additional_agents) is not int or curriculum.additional_agents < 0:
        raise ValueError("training.agent_count_curriculum.additional_agents must be an integer >= 0.")
    if training.dynamic_max_geo_distance.enabled:
        if env.success_condition == "coverage" or env.num_obstacles:
            raise ValueError("Dynamic max distance requires a target task without generated obstacles.")
    if curriculum.enabled:
        if not training.dynamic_max_geo_distance.enabled or curriculum.start_agents > env.num_agents:
            raise ValueError("Agent curriculum requires dynamic_max_geo_distance and start_agents <= env.num_agents.")
        ev_data = data.get("evaluation", {})
        exit_source = ev_data.get("early_exit_source", EvaluationConfig.early_exit_source)
        if (not ev_data.get("early_exit", EvaluationConfig.early_exit)
                or exit_source not in {"evaluation", "validation"}
                or not ev_data.get("save_model", EvaluationConfig.save_model)):
            raise ValueError("Agent curriculum requires saved checkpoints and inspection- or validation-based early exit.")
        # Validation always uses the filtered training stage. Diagnostic
        # inspection overrides cannot change its curriculum progression gate.
        if exit_source == "evaluation" and (
                ev_data.get("dynamic_max_geo_distance") is False
                or ev_data.get("num_agents") is not None):
            raise ValueError("Inspection-based curriculum progression requires the filtered stage population "
                             "(inspection_evaluation.dynamic_max_geo_distance=null or true, num_agents=null).")
    if training.adaptive_target_spawn.enabled and (
        env.num_obstacles or training.randomize_base or training.minimum_geodesic_separation
        or env.success_condition == "coverage"
    ):
        raise ValueError("Adaptive target spawning requires static target maps, a fixed base per map, "
                         "no generated obstacles and minimum_geodesic_separation=false.")
    if training.adaptive_target_spawn.enabled and generation.enabled:
        # Per-map cell support is needed even when map choice is uniform.
        samples = training.adaptive_map_sampling.eligibility_samples_per_map
        if type(samples) is not int or samples < 1:
            raise ValueError("training.adaptive_map_sampling.eligibility_samples_per_map must be a positive integer "
                             "when adaptive target spawning uses random buildings.")
    if (generation.enabled and data.get("evaluation", {}).get("early_exit", EvaluationConfig.early_exit)
            and data.get("evaluation", {}).get("early_exit_source", EvaluationConfig.early_exit_source) == "training"):
        raise ValueError("Random-map episode resampling requires inspection- or validation-based early exit; "
                         "fixed-lane training gates do not apply.")
    if training.adaptive_map_sampling.enabled:
        if not generation.enabled and len(map_names or []) < 2:
            raise ValueError("Adaptive map sampling requires at least two static maps or persistent random buildings.")
        if data.get("evaluation", {}).get("early_exit_source", "evaluation") not in {"evaluation", "validation"}:
            raise ValueError("Adaptive map sampling requires evaluation-based early exit; fixed-lane training gates do not apply.")
        if generation.enabled and data.get("evaluation", {}).get("random_eval_maps") is not None:
            raise ValueError("Adaptive random-map sampling reserves its own held-out maps; omit evaluation.random_eval_maps.")
    if generation.load_maps_from_bank and not generation.enabled:
        raise ValueError("random_buildings.load_maps_from_bank requires random_buildings.enabled=true.")
    map_paths: tuple[Path, ...] = ()
    if generation.enabled:
        if map_names not in (None, [], ["random"]):
            raise ValueError("random_buildings.enabled requires env.map_names: [] (or [random]); authored maps cannot be specified.")
        if env.num_obstacles:
            raise ValueError("Random buildings and random obstacles are mutually exclusive.")
        from swarmecho.env.random_buildings import generate_building, map_seed
        _, building = generate_building(generation, map_seed(training.seed, 0), name="map_0000",
                                        drone_clearance=env.drone_radius + env.obstacle_planning_clearance_m)
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
    reward_data["priveleged_vote_reward"] = _strict_dataclass(
        PrivelegedVoteRewardConfig, reward_data.get("priveleged_vote_reward", {}),
        "reward.priveleged_vote_reward",
    )
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
                     else level.evaluation.random_eval_envs)
        if type(count) is not int or count < 1:
            raise ValueError("inspection_evaluation.inspection_maps must be null or a positive integer.")
        if validation.enabled and count > validation.num_maps:
            raise ValueError("Inspection map count cannot exceed validation.num_maps.")
        level = replace(level, evaluation=replace(level.evaluation, random_eval=True,
            random_eval_envs=count, random_eval_maps=None if validation.enabled or training.adaptive_map_sampling.enabled
            else level.evaluation.random_eval_maps))
    if level.evaluation.inspection_maps is not None and (
            type(level.evaluation.inspection_maps) is not int or level.evaluation.inspection_maps < 1):
        raise ValueError("inspection_evaluation.inspection_maps must be null or a positive integer.")
    if level.evaluation.early_exit_source == "validation" and not validation.enabled:
        raise ValueError("Validation early exit requires validation.enabled=true.")
    validate_reward_config(level.reward)
    validate_priveleged_vote_reward(level.reward.priveleged_vote_reward, level.env)
    success_conditions = {"coverage", "discovery", "delivery", "chain_held"}
    if level.env.success_condition not in success_conditions:
        raise ValueError(f"env.success_condition must be one of {sorted(success_conditions)}.")
    if (level.evaluation.success_condition is not None
            and level.evaluation.success_condition not in success_conditions):
        raise ValueError(f"evaluation.success_condition must be one of {sorted(success_conditions)} or null.")
    if level.network.critic_type != "observation":
        raise ValueError("network.critic_type currently supports only observation.")
    if level.env.success_condition == "coverage" and level.training.minimum_geodesic_separation:
        raise ValueError("Coverage training has no target for minimum geodesic separation.")
    if level.env.success_condition != "coverage" and level.ideal_chain_margin_m < 0:
        raise ValueError(f"level {level.name!r} is geometrically unsolvable: ideal chain margin is {level.ideal_chain_margin_m:.3f} m.")
    if level.training.num_epochs < 1 or level.training.num_minibatches < 1:
        raise ValueError("PPO epoch and minibatch counts must be positive.")
    if level.training.num_envs % level.training.num_minibatches:
        raise ValueError("Recurrent training requires num_envs divisible by num_minibatches.")
    if level.training.actor_clip_eps <= 0:
        raise ValueError("training.actor_clip_eps must be positive.")
    if level.training.value_clip_eps is not None and level.training.value_clip_eps < 0:
        raise ValueError("training.value_clip_eps must be nonnegative or null.")
    if level.training.entropy_mode not in {"legacy", "squashed"}:
        raise ValueError("training.entropy_mode must be legacy or squashed.")
    if level.training.value_normalization not in {"none", "running"}:
        raise ValueError("training.value_normalization must be none or running.")
    if level.training.diagnostics_every < 1:
        raise ValueError("training.diagnostics_every must be positive.")
    if (type(level.logging.required_agents_metrics_window_episodes) is not int
            or level.logging.required_agents_metrics_window_episodes < 1):
        raise ValueError("logging.required_agents_metrics_window_episodes must be a positive integer.")
    if level.network.memory_comm_every_k_steps < 1:
        raise ValueError("network.memory_comm_every_k_steps must be positive.")
    if type(env.base_keeps_informing) is not bool:
        raise ValueError("env.base_keeps_informing must be a boolean.")
    if level.training.noise_level < 0.0:
        raise ValueError("training.noise_level must be non-negative.")
    parallel_envs = level.evaluation.eval_parallel_envs
    if parallel_envs is not None and (type(parallel_envs) is not int or parallel_envs < 1):
        raise ValueError("evaluation.eval_parallel_envs must be null or a positive integer.")
    if level.training.oom_recovery.enabled and 500 % level.training.num_minibatches:
        raise ValueError("OOM recovery requires num_minibatches to divide the 500-environment decrement.")
    ev = level.evaluation
    if ev.num_agents is not None and (type(ev.num_agents) is not int or ev.num_agents < 1):
        raise ValueError("evaluation.num_agents must be null or a positive integer.")
    if level.static_map_suite:
        if ev.random_eval or ev.random_eval_maps is not None:
            raise ValueError("Static multi-map evaluation uses env.map_names; disable random_eval and random_eval_maps.")
        if ev.training_robustness or ev.eval_differes_from_training_map or ev.eval_map or ev.eval_fixed_obstacle_bounds:
            raise ValueError("Static multi-map evaluation cannot use robustness or evaluation map overrides.")
        if env.num_obstacles:
            raise ValueError("Static multi-map training cannot use generated obstacles.")
    if type(ev.apply_trainings_min_geo_separation) is not bool:
        raise ValueError("evaluation.apply_trainings_min_geo_separation must be a boolean.")
    if ev.minimum_geodesic_separation is not None and type(ev.minimum_geodesic_separation) is not bool:
        raise ValueError("evaluation.minimum_geodesic_separation must be boolean or null.")
    if ev.dynamic_max_geo_distance is not None and type(ev.dynamic_max_geo_distance) is not bool:
        raise ValueError("evaluation.dynamic_max_geo_distance must be boolean or null.")
    if ev.dynamic_max_geo_distance and (env.success_condition == "coverage" or env.num_obstacles):
        raise ValueError("Dynamic max distance evaluation requires a static-geometry target task.")
    if type(ev.random_eval) is not bool or type(ev.random_eval_envs) is not int or ev.random_eval_envs < 1:
        raise ValueError("random_eval must be boolean and random_eval_envs a positive integer.")
    if ev.replay_targets_from_storey is not None and (
        type(ev.replay_targets_from_storey) is not int or ev.replay_targets_from_storey < 1
    ):
        raise ValueError("evaluation.replay_targets_from_storey must be null or a positive integer.")
    if ev.random_eval:
        if ev.training_robustness:
            raise ValueError("random_eval requires training_robustness=false (robust evaluation off).")
        if ev.eval_differes_from_training_map or ev.eval_map or ev.eval_fixed_obstacle_bounds or env.num_obstacles:
            raise ValueError("random_eval cannot be combined with eval_map overrides or random/fixed obstacles.")
        if ev.random_eval_maps is not None:
            if not isinstance(ev.random_eval_maps, (list, tuple)) or len(ev.random_eval_maps) != ev.random_eval_envs:
                raise ValueError("random_eval_maps must contain exactly random_eval_envs map paths.")
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
    for name in ("eval_offset", "eval_video_offset", "checkpoint_offset"):
        value = getattr(level.evaluation, name)
        if type(value) is not int or value < 0:
            raise ValueError(f"evaluation.{name} must be a non-negative integer.")
    for name in ("early_exit_success_rate", "eval_min_train_success"):
        if not 0.0 <= getattr(level.evaluation, name) <= 1.0:
            raise ValueError(f"evaluation.{name} must be in [0, 1].")
    final_success_rate = level.evaluation.final_early_exit_success_rate
    if final_success_rate is not None and not 0.0 <= final_success_rate <= 1.0:
        raise ValueError("evaluation.final_early_exit_success_rate must be in [0, 1] or null.")
    speed_gate = level.evaluation.early_exit_min_success_length_reduction
    if speed_gate is not None and not 0.0 <= speed_gate <= 1.0:
        raise ValueError("evaluation.early_exit_min_success_length_reduction must be in [0, 1] or null.")
    if type(level.evaluation.training_robustness) is not bool:
        raise ValueError("evaluation.training_robustness must be a boolean.")
    if type(level.evaluation.final_robustness) is not bool:
        raise ValueError("evaluation.final_robustness must be a boolean.")
    if type(level.evaluation.eval_robustness_runs) is not int or level.evaluation.eval_robustness_runs < 2:
        raise ValueError("evaluation.eval_robustness_runs must be at least 2 (no 1/1 robust evaluations).")
    if type(level.evaluation.early_exit_hold_evals) is not int or level.evaluation.early_exit_hold_evals < 0:
        raise ValueError("evaluation.early_exit_hold_evals must be a non-negative integer.")
    if ev.early_exit_source not in {"evaluation", "training", "validation"}:
        raise ValueError("inspection_evaluation.early_exit_source must be inspection_evaluation, training or validation.")
    if ev.early_exit_source == "training" and generation.enabled:
        raise ValueError("Training-based early exit currently requires authored maps.")
    if ev.map_suite_replay_limit is not None and (
        type(ev.map_suite_replay_limit) is not int or ev.map_suite_replay_limit < 1
    ):
        raise ValueError("evaluation.map_suite_replay_limit must be null or a positive integer.")
    if level.evaluation.eval_action_noise_max < 0.0:
        raise ValueError("evaluation.eval_action_noise_max must be non-negative.")
    fixed_bounds = level.evaluation.eval_fixed_obstacle_bounds
    if fixed_bounds is not None:
        if len(fixed_bounds) != level.env.num_obstacles or any(
            len(bounds) != 6
            or any(bounds[index] >= bounds[index + 3] for index in range(3))
            for bounds in fixed_bounds
        ):
            raise ValueError(
                "evaluation.eval_fixed_obstacle_bounds must define ordered min/max "
                "coordinates for every obstacle."
            )
    if level.num_updates < 1:
        raise ValueError("training.total_timesteps must cover at least one rollout.")
    if level.logging.wandb_mode not in {"disabled", "offline", "online"}:
        raise ValueError("logging.wandb_mode must be disabled, offline, or online.")
    for name in ("adaptive_cell_diagnostics", "individual_map_diagnostics"):
        if type(getattr(level.logging, name)) is not bool:
            raise ValueError(f"logging.{name} must be a boolean.")
    if level.reward.chain_reward_system not in {"euclidean", "obstacle_geodesic"}:
        raise ValueError("reward.chain_reward_system must be euclidean or obstacle_geodesic.")
    for name in ("success_reward_as_hold_record",
                 "progress_based_cluster_tiebreaker_soft",
                 "gap_reward_uses_change",
                 "peer_informing_reward_enabled", "delivery_gap_vesting_enabled"):
        if type(getattr(level.reward, name)) is not bool:
            raise ValueError(f"reward.{name} must be a boolean.")
    if level.reward.progress_based_cluster_tiebreaker_soft and (
        level.env.success_condition == "coverage" or level.reward.allow_redundancy_reward
    ):
        raise ValueError("Soft cluster credit requires a target mission and allow_redundancy_reward=false.")
    if (type(level.reward.delivery_gap_vesting_steps) is not int
            or level.reward.delivery_gap_vesting_steps < 1):
        raise ValueError("reward.delivery_gap_vesting_steps must be a positive integer.")
    if level.reward.delivery_gap_vesting_enabled and not (
        level.reward.gap_reward_uses_change and level.reward.target_found_requires_delivery
    ):
        raise ValueError("Delivery gap vesting requires gap-change reward and target delivery.")
    if level.reward.success_reward_as_hold_record and level.env.success_condition != "chain_held":
        raise ValueError("reward.success_reward_as_hold_record requires env.success_condition=chain_held.")
    if (level.reward.success_reward_as_hold_record
            and level.evaluation.success_condition not in (None, "chain_held")):
        raise ValueError("Hold-record success reward requires chain_held evaluation success.")
    if level.env.success_condition == "coverage" and (
        level.reward.gap_reward_uses_change or level.reward.peer_informing_reward_enabled
    ):
        raise ValueError("Gap-change and peer-informing rewards require a target mission.")
    if level.env.num_obstacles and level.env.obstacle_spawn_layer_max <= level.env.obstacle_spawn_layer_min:
        raise ValueError("The obstacle spawn layer range must have positive height.")
    return level


def load_level_cli(arguments: list[str]) -> Level:
    """Load a level with the same key=value CLI contract as training."""
    level_name = "M00_no_maze_open_cuboid"
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
