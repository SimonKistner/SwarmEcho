"""Strict configuration loader for the 3D-only migration runtime."""

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


LEVEL_3D_DIR = Path(__file__).parents[1] / "curriculum_config/levels_3d"
BUILDING_DIR = Path(__file__).parents[1] / "curriculum_config/buildings"


@dataclass(frozen=True)
class Level3D:
    name: str
    building_name: str
    building: BuildingArrays
    env: Baseline3DConfig
    reward: Baseline3DRewardConfig
    training: "Training3DConfig"

    @property
    def ideal_chain_margin_m(self) -> float:
        return maximum_chain_distance(self.env) - self.building.max_base_to_top_corner_m


def _strict_dataclass(cls, values: object, label: str):
    if not isinstance(values, dict):
        raise ValueError(f"{label} must be a mapping.")
    allowed = set(cls.__dataclass_fields__)
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {label} fields: {', '.join(sorted(unknown))}.")
    return cls(**values)


@dataclass(frozen=True)
class Training3DConfig:
    updates: int = 10
    num_envs: int = 32
    num_steps: int = 64
    num_epochs: int = 2
    num_minibatches: int = 2
    hidden_dim: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    seed: int = 42
    output_dir: str = "outputs/3d_baseline"


def load_level_3d(name_or_path: str | Path = "B00_3d_baseline") -> Level3D:
    """Load a 3D level without routing through the legacy 2D config loader."""
    source = Path(name_or_path)
    if not source.exists():
        source = LEVEL_3D_DIR / f"{source.stem}.yaml"
    if not source.exists():
        raise FileNotFoundError(f"3D level not found: {name_or_path}")
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("3D level root must be a mapping.")
    building_name = str(data.get("building", ""))
    if not building_name:
        raise ValueError("3D level must select a building.")
    building_path = BUILDING_DIR / f"{Path(building_name).stem}.yaml"
    level = Level3D(
        name=str(data.get("name", source.stem)),
        building_name=Path(building_name).stem,
        building=load_building(building_path),
        env=_strict_dataclass(Baseline3DConfig, data.get("env", {}), "env"),
        reward=_strict_dataclass(Baseline3DRewardConfig, data.get("reward", {}), "reward"),
        training=_strict_dataclass(Training3DConfig, data.get("training", {}), "training"),
    )
    if level.ideal_chain_margin_m < 0:
        raise ValueError(
            f"3D level {level.name!r} is geometrically unsolvable: ideal chain "
            f"margin is {level.ideal_chain_margin_m:.3f} m. Increase agents/radii "
            "or reduce the building dimensions."
        )
    if level.training.num_envs % level.training.num_minibatches:
        raise ValueError("3D recurrent training requires num_envs divisible by num_minibatches.")
    return level
