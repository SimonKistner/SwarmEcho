"""Strict 3D level configuration and key=value CLI overrides."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import warnings

from omegaconf import OmegaConf

_SRC_ROOT = Path(__file__).resolve().parent.parent
MAP_DIR = _SRC_ROOT / "curriculum_config" / "maps"
LEVEL_DIR = _SRC_ROOT / "curriculum_config" / "levels"

# Keep training action perturbations aligned with the default used by the
# standalone robust checkpoint evaluator.
DEFAULT_ACTION_NOISE_LEVEL = 0.011

@dataclass(frozen=True)
class NetworkConfig:
    hidden_dim: int = 256
    num_layers: int = 3
    actor_num_layers: int = 3
    actor_memory: bool = True
    critic_memory: bool = True
    critic_type: str = "observation"  # observation (unchanged) | privileged (compact 3D state)
    memory_comm_enabled: bool = True
    memory_comm_every_k_steps: int = 5
    tarmac_sig_dim: int = 16
    tarmac_val_dim: int = 32
    tarmac_include_self: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    randomize_base: bool = False
    minimum_geodesic_separation: bool = False
    minimum_geodesic_separation_multiplier: float = 2.0
    spawn_pair_max_attempts: int = 1024
    total_timesteps: int = 250_000_000
    profile_timing: bool = False
    seed: int = 42
    num_envs: int = 4000
    num_steps: int = 100
    num_epochs: int = 4
    num_minibatches: int = 20
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    actor_clip_eps: float = 0.2
    # Raw value units, or normalized units with value_normalization=running.
    # None (YAML: null) disables value clipping; zero does NOT disable it.
    value_clip_eps: float | None = 0.2
    entropy_mode: str = "legacy"  # legacy or squashed (reparameterized tanh entropy)
    value_normalization: str = "none"  # none or running; checkpointed critic units
    diagnostics_every: int = 1  # PPO updates between before/after diagnostic passes
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    checkpoint_path: str | None = None
    checkpoint_step_offset: int | None = None
    ckpt_loading_mode: str = "branch"  # "resume" continues counters; "branch" carries progress; "init" loads weights only
    # Perturb sampled pre-tanh actions before stepping training environments.
    # Evaluation uses its separate eval_action_noise_max configuration.
    training_noise: bool = False
    noise_level: float = DEFAULT_ACTION_NOISE_LEVEL


@dataclass(frozen=True)
class EvaluationConfig:
    eval_freq: int = 20
    eval_offset: int = 1
    eval_min_train_success: float = 0.0
    eval_differes_from_training_map: bool = False
    eval_map: str | None = None
    eval_parallel_envs: int = 4000
    # Periodic metrics may use the same ensemble as the always-robust final eval.
    training_robustness: bool = False
    eval_robustness_runs: int = 5
    eval_action_noise_max: float = DEFAULT_ACTION_NOISE_LEVEL
    # Optional handcrafted [min_x,min_y,min_z,max_x,max_y,max_z] cuboids.
    eval_fixed_obstacle_bounds: tuple[tuple[float, float, float, float, float, float], ...] | None = None
    eval_broadcast_on_curriculum_early_stop: bool = False
    early_exit: bool = False
    early_exit_threshold: float = 0.99
    early_exit_hold_evals: int = 0
    eval_video: bool = True
    eval_video_freq: int = 20
    eval_video_offset: int = 1
    training_heatmap_creation: bool = False
    eval_not_deliv_not_visual_splitt_in_two: bool = False
    save_model: bool = True
    checkpoint_freq: int = 50
    checkpoint_offset: int = 0
    checkpoint_dir: str | None = None


@dataclass(frozen=True)
class LoggingConfig:
    run_name: str | None = "M00_no_maze_open_cuboid_3D"
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
    env: object
    reward: object
    training: TrainingConfig
    network: NetworkConfig
    evaluation: EvaluationConfig
    logging: LoggingConfig

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

    requested = Path(evaluation.eval_map)
    source = requested if requested.exists() else MAP_DIR / f"{requested.stem}.yaml"
    if not source.exists():
        raise FileNotFoundError(
            f"Evaluation map not found: {evaluation.eval_map!r}. "
            f"Expected a map name in {MAP_DIR} or an existing path."
        )
    map_name = source.stem
    if map_name == Path(level.building_name).stem:
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
        raise ValueError(f"Unknown {label} fields: {', '.join(sorted(unknown))}.")
    return cls(**values)


def load_level(name_or_path: str | Path = "M00_no_maze_open_cuboid_3D", overrides: list[str] | None = None) -> Level:
    """Load a strict 3D level through the canonical config module."""
    from swarmecho.env.environment import EnvConfig, RewardConfig
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
    level = Level(
        name=source.stem,
        map_names=[str(map_names[0])],
        building=load_building(MAP_DIR / f"{Path(map_names[0]).stem}.yaml"),
        env=_strict_dataclass(EnvConfig, env_data, "env"),
        reward=_strict_dataclass(RewardConfig, data.get("reward", {}), "reward"),
        training=_strict_dataclass(TrainingConfig, data.get("training", {}), "training"),
        network=_strict_dataclass(NetworkConfig, data.get("network", {}), "network"),
        evaluation=_strict_dataclass(EvaluationConfig, data.get("evaluation", {}), "evaluation"),
        logging=_strict_dataclass(LoggingConfig, data.get("logging", {}), "logging"),
    )
    if level.ideal_chain_margin_m < 0:
        raise ValueError(f"3D level {level.name!r} is geometrically unsolvable: ideal chain margin is {level.ideal_chain_margin_m:.3f} m.")
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
    if level.training.noise_level < 0.0:
        raise ValueError("training.noise_level must be non-negative.")
    if level.evaluation.eval_parallel_envs < 1:
        raise ValueError("evaluation.eval_parallel_envs must be positive.")
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
    for name in ("early_exit_threshold", "eval_min_train_success"):
        if not 0.0 <= getattr(level.evaluation, name) <= 1.0:
            raise ValueError(f"evaluation.{name} must be in [0, 1].")
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
        raise ValueError("3D reward.chain_reward_system must be euclidean or obstacle_geodesic.")
    if level.env.num_obstacles and level.env.obstacle_spawn_layer_max <= level.env.obstacle_spawn_layer_min:
        raise ValueError("The 3D obstacle spawn layer range must have positive height.")
    return level


def load_level_cli(arguments: list[str]) -> Level:
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
    return load_level(level_name, overrides)
