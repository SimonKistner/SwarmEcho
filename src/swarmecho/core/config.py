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

# env
@dataclass(frozen=True)
class EnvConfig:
    num_agents: int = 5

    dt: float = 0.1  # Simulated time per step; scales velocity and position updates.
    max_force: float = 15.0
    max_speed: float = 5.0
    drag: float = 0.85
    drone_radius: float = 0.25

    comm_radius_base: float = 6.0
    base_keeps_informing: bool = False
    comm_radius: float = 5.0
    visual_radius: float = 4.0
    radar_bins: int = 8

    target_wall_buffer_fraction: float = 0.1
    spawn_delay: int = 5

    max_steps: int = 700
    hold_chain_for: int = 50
    no_movement_termination_steps: int = 50
    movement_epsilon: float = 1e-3
    success_condition: str = "chain_held"  # coverage | discovery | delivery | chain_held

    observe_target_vector: bool = False
    observe_base_vector: bool = False
    observe_chain_contributor: bool = False
    observe_current_timestep: bool = False  # Episode step / max_steps, in [0, 1].
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
class RewardConfig:
    target_found_requires_delivery: bool = True
    chain_reward_system: str = "euclidean"

    # Credit every drone on a simple base-target path, including alternate routes.
    allow_redundancy_reward: bool = False
    enable_chain_efficiency_reward: bool = False
    chain_efficiency_bonus: float = 0.5

    exploration_bonus: float = 0.25
    finder_bonus: float = 0.0
    target_found_bonus: float = 100.0
    success_bonus: float = 500.0

    collision_penalty: float = 0.5
    # Team cost, divided among active drones on every transition.
    time_penalty_per_step: float = 5.0
    # Per contributing drone, per metre of the base-target route already closed.
    gap_reduction_meter_bonus: float = 0.125
    no_movement_termination_penalty: float = -1000.0


# training
# Keep training action perturbations aligned with the default used by the
# standalone robust checkpoint evaluator.
DEFAULT_ACTION_NOISE_LEVEL = 0.011

@dataclass(frozen=True)
class TrainingConfig:
    total_timesteps: int = 250_000_000
    num_envs: int = 4000
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
    eval_parallel_envs: int = 4000
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

    # Periodic metrics may use the same ensemble as the always-robust final eval.
    training_robustness: bool = False
    eval_robustness_runs: int = 5
    eval_action_noise_max: float = DEFAULT_ACTION_NOISE_LEVEL

    early_exit: bool = False
    success_condition: str | None = None  # null follows env; coverage | discovery | delivery | chain_held
    early_exit_success_rate: float = 0.99
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
    eval_not_deliv_not_visual_splitt_in_two: bool = False


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

    suppress_xla_warnings: bool = True
    terminal_logging_frequency: int = 1
    terminal_log_warmup: bool = False
    terminal_log_init: bool = False


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
    random_buildings: RandomBuildingConfig = RandomBuildingConfig()

    @property
    def building_name(self) -> str:
        return self.map_names[0]

    @property
    def ideal_chain_margin_m(self) -> float:
        from swarmecho.env.environment import maximum_chain_distance
        return maximum_chain_distance(self.env) - self.building.max_base_to_top_corner_m

    @property
    def num_updates(self) -> int:
        return self.training.total_timesteps // (self.training.num_envs * self.training.num_steps)


def resolve_evaluation_level(level: Level) -> Level:
    """Return the level configuration whose building should be used for eval."""
    evaluation = level.evaluation
    if evaluation.success_condition is not None:
        level = replace(level, env=replace(level.env, success_condition=evaluation.success_condition))
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


def _strict_dataclass(cls, values: object, label: str):
    if not isinstance(values, dict):
        raise ValueError(f"{label} must be a mapping.")
    unknown = set(values) - set(cls.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown {label} fields: {', '.join(sorted(unknown))}.")
    return cls(**values)


def load_level(name_or_path: str | Path = "M00_no_maze_open_cuboid", overrides: list[str] | None = None) -> Level:
    """Load a strict level through the canonical config module."""
    from omegaconf import OmegaConf
    from swarmecho.env.buildings import load_building

    source = resolve_level_file(name_or_path, LEVEL_DIR)
    if not source.exists():
        raise FileNotFoundError(f"level not found: {name_or_path}")
    data = OmegaConf.to_container(OmegaConf.load(source), resolve=True)
    if overrides:
        data = OmegaConf.to_container(OmegaConf.merge(OmegaConf.create(data), OmegaConf.from_dotlist(overrides)), resolve=True)
    allowed = {"env", "reward", "training", "network", "evaluation", "logging", "random_buildings"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown config sections: {', '.join(sorted(unknown))}.")
    env_data = dict(data.get("env", {}))
    map_names = env_data.pop("map_names", None)
    generation = _strict_dataclass(RandomBuildingConfig, data.get("random_buildings", {}), "random_buildings")
    env = _strict_dataclass(EnvConfig, env_data, "env")
    training = _strict_dataclass(TrainingConfig, data.get("training", {}), "training")
    if generation.load_maps_from_bank and not generation.enabled:
        raise ValueError("random_buildings.load_maps_from_bank requires random_buildings.enabled=true.")
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
        if not isinstance(map_names, list) or len(map_names) != 1:
            raise ValueError("env.map_names must select exactly one map.")
        building = load_building(resolve_map_file(MAP_DIR / f"{Path(map_names[0]).stem}.yaml", MAP_DIR))
    level = Level(
        name=canonical_level_name(source.stem),
        map_names=[canonical_map_name(str(map_names[0]))],
        building=building,
        env=env,
        reward=_strict_dataclass(RewardConfig, data.get("reward", {}), "reward"),
        training=training,
        network=_strict_dataclass(NetworkConfig, data.get("network", {}), "network"),
        evaluation=_strict_dataclass(EvaluationConfig, data.get("evaluation", {}), "evaluation"),
        logging=_strict_dataclass(LoggingConfig, data.get("logging", {}), "logging"),
        random_buildings=generation,
    )
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
    if level.network.memory_comm_every_k_steps < 1:
        raise ValueError("network.memory_comm_every_k_steps must be positive.")
    if type(env.base_keeps_informing) is not bool:
        raise ValueError("env.base_keeps_informing must be a boolean.")
    if level.training.noise_level < 0.0:
        raise ValueError("training.noise_level must be non-negative.")
    if level.evaluation.eval_parallel_envs < 1:
        raise ValueError("evaluation.eval_parallel_envs must be positive.")
    ev = level.evaluation
    if type(ev.apply_trainings_min_geo_separation) is not bool:
        raise ValueError("evaluation.apply_trainings_min_geo_separation must be a boolean.")
    if ev.minimum_geodesic_separation is not None and type(ev.minimum_geodesic_separation) is not bool:
        raise ValueError("evaluation.minimum_geodesic_separation must be boolean or null.")
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
    speed_gate = level.evaluation.early_exit_min_success_length_reduction
    if speed_gate is not None and not 0.0 <= speed_gate <= 1.0:
        raise ValueError("evaluation.early_exit_min_success_length_reduction must be in [0, 1] or null.")
    if type(level.evaluation.training_robustness) is not bool:
        raise ValueError("evaluation.training_robustness must be a boolean.")
    if type(level.evaluation.eval_robustness_runs) is not int or level.evaluation.eval_robustness_runs < 2:
        raise ValueError("evaluation.eval_robustness_runs must be at least 2 (no 1/1 robust evaluations).")
    if type(level.evaluation.early_exit_hold_evals) is not int or level.evaluation.early_exit_hold_evals < 0:
        raise ValueError("evaluation.early_exit_hold_evals must be a non-negative integer.")
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
    if level.reward.chain_reward_system not in {"euclidean", "obstacle_geodesic"}:
        raise ValueError("reward.chain_reward_system must be euclidean or obstacle_geodesic.")
    for name in ("time_penalty_per_step", "gap_reduction_meter_bonus"):
        value = getattr(level.reward, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"reward.{name} must be finite and nonnegative.")
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
