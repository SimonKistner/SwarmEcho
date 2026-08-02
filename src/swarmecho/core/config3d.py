"""Strict 3D configuration with the same domains as the maintained config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from swarmecho.env.baseline3d import (
    Baseline3DConfig,
    Baseline3DRewardConfig,
    maximum_chain_distance,
)
from swarmecho.env.buildings import BuildingArrays, load_building

_SRC_ROOT = Path(__file__).parents[1]
LEVEL_DIR = _SRC_ROOT / "curriculum_config/levels"
MAP_DIR = _SRC_ROOT / "curriculum_config/maps"


@dataclass(frozen=True)
class Network3DConfig:
    hidden_dim: int = 256
    num_layers: int = 3
    actor_num_layers: int = 3
    actor_memory: bool = True
    critic_memory: bool = True
    critic_type: str = "observation"
    memory_comm_enabled: bool = True
    memory_comm_every_k_steps: int = 1
    tarmac_sig_dim: int = 16
    tarmac_val_dim: int = 32
    tarmac_include_self: bool = False


@dataclass(frozen=True)
class Training3DConfig:
    total_timesteps: int = 16_384_000
    seed: int = 42
    num_envs: int = 256
    num_steps: int = 64
    num_epochs: int = 2
    num_minibatches: int = 8
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
    eval_freq: int = 50
    eval_offset: int = 0
    eval_min_train_success: float = 0.0
    eval_parallel_envs: int = 8
    early_exit: bool = False
    early_exit_threshold: float = 0.99
    eval_video: bool = True
    eval_video_freq: int = 50
    eval_video_offset: int = 0
    save_model: bool = True
    checkpoint_freq: int = 50
    checkpoint_offset: int = 0
    checkpoint_dir: str | None = None


@dataclass(frozen=True)
class Logging3DConfig:
    run_name: str | None = "M00_no_maze_open_cuboid_3D"
    use_timestamp_postfix: bool = False
    log_dir: str = "outputs"
    wandb_mode: str = "disabled"
    wandb_project: str = "SwarmEcho"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    suppress_xla_warnings: bool = True
    log_every: int = 1


@dataclass(frozen=True)
class Level3D:
    name: str
    map_names: list[str]
    building: BuildingArrays
    env: Baseline3DConfig
    reward: Baseline3DRewardConfig
    training: Training3DConfig
    network: Network3DConfig
    evaluation: Evaluation3DConfig
    logging: Logging3DConfig

    @property
    def building_name(self) -> str:
        return self.map_names[0]

    @property
    def ideal_chain_margin_m(self) -> float:
        return maximum_chain_distance(self.env) - self.building.max_base_to_top_corner_m

    @property
    def num_updates(self) -> int:
        return self.training.total_timesteps // (
            self.training.num_envs * self.training.num_steps
        )


def _strict_dataclass(cls, values: object, label: str):
    if not isinstance(values, dict):
        raise ValueError(f"{label} must be a mapping.")
    allowed = set(cls.__dataclass_fields__)
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {label} fields: {', '.join(sorted(unknown))}.")
    return cls(**values)


def load_level_3d(name_or_path: str | Path = "M00_no_maze_open_cuboid_3D") -> Level3D:
    """Load one strict 3D level from the standard level/map directories."""
    source = Path(name_or_path)
    if not source.exists():
        source = LEVEL_DIR / f"{source.stem}.yaml"
    if not source.exists():
        raise FileNotFoundError(f"3D level not found: {name_or_path}")
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("3D level root must be a mapping.")
    env_data = dict(data.get("env", {}))
    map_names = env_data.pop("map_names", None)
    if not isinstance(map_names, list) or len(map_names) != 1:
        raise ValueError("3D env.map_names must select exactly one map.")
    env = _strict_dataclass(Baseline3DConfig, env_data, "env")
    level = Level3D(
        name=source.stem,
        map_names=[str(map_names[0])],
        building=load_building(MAP_DIR / f"{Path(map_names[0]).stem}.yaml"),
        env=env,
        reward=_strict_dataclass(Baseline3DRewardConfig, data.get("reward", {}), "reward"),
        training=_strict_dataclass(Training3DConfig, data.get("training", {}), "training"),
        network=_strict_dataclass(Network3DConfig, data.get("network", {}), "network"),
        evaluation=_strict_dataclass(Evaluation3DConfig, data.get("evaluation", {}), "evaluation"),
        logging=_strict_dataclass(Logging3DConfig, data.get("logging", {}), "logging"),
    )
    if level.ideal_chain_margin_m < 0:
        raise ValueError(
            f"3D level {level.name!r} is geometrically unsolvable: ideal chain "
            f"margin is {level.ideal_chain_margin_m:.3f} m. Increase agents/radii "
            "or reduce the map dimensions."
        )
    if level.training.num_envs % level.training.num_minibatches:
        raise ValueError("Recurrent training requires num_envs divisible by num_minibatches.")
    if level.num_updates < 1:
        raise ValueError("training.total_timesteps must cover at least one rollout.")
    if level.logging.wandb_mode not in {"disabled", "offline", "online"}:
        raise ValueError("logging.wandb_mode must be disabled, offline, or online.")
    if level.reward.chain_reward_system != "euclidean":
        raise ValueError("3D reward.chain_reward_system currently supports only euclidean.")
    return level
