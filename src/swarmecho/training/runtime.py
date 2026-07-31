"""Shared construction of SwarmEcho environment, model, and checkpoint runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flax import nnx
from omegaconf import DictConfig

from swarmecho.core.config import compute_action_dim, compute_obs_dim
from swarmecho.env.observations import make_obs_fns
from swarmecho.env.physics import make_env_fns
from swarmecho.env.rewards import make_reward_fn
from swarmecho.models.mappo import MAPPOModel
from swarmecho.training.checkpoints import restore_model_checkpoint


@dataclass(frozen=True)
class EnvironmentRuntime:
    env_step: Any
    reset: Any
    compute_obs: Any
    compute_reward: Any
    width: float
    height: float
    occupancy_grid: Any


@dataclass(frozen=True)
class EvaluationRuntime:
    model: MAPPOModel
    environment: EnvironmentRuntime
    checkpoint_path: Path


def build_environment_runtime(cfg: DictConfig) -> EnvironmentRuntime:
    """Build the one maintained environment/observation/reward function set."""
    env_step, reset, _, (width, height, occupancy_grid) = make_env_fns(cfg)
    compute_obs, _ = make_obs_fns(cfg, width, height, occupancy_grid)
    compute_reward = make_reward_fn(cfg)
    return EnvironmentRuntime(
        env_step=env_step,
        reset=reset,
        compute_obs=compute_obs,
        compute_reward=compute_reward,
        width=float(width),
        height=float(height),
        occupancy_grid=occupancy_grid,
    )


def build_model(cfg: DictConfig, *, rng_seed: int) -> MAPPOModel:
    """Construct the configured MAPPO model through one shared factory."""
    return MAPPOModel(
        obs_dim=compute_obs_dim(cfg),
        act_dim=compute_action_dim(cfg),
        num_agents=int(cfg.env.num_agents),
        hidden_dim=int(cfg.network.hidden_dim),
        num_layers=int(cfg.network.num_layers),
        actor_num_layers=int(cfg.network.actor_num_layers),
        actor_memory=bool(cfg.network.get("actor_memory", False)),
        critic_memory=bool(cfg.network.get("critic_memory", False)),
        rngs=nnx.Rngs(int(rng_seed)),
        memory_comm_enabled=bool(cfg.network.get("memory_comm_enabled", False)),
        memory_comm_every_k_steps=int(cfg.network.get("memory_comm_every_k_steps", 5)),
        tarmac_sig_dim=int(cfg.network.get("tarmac_sig_dim", 64)),
        tarmac_val_dim=int(cfg.network.get("tarmac_val_dim", 128)),
        tarmac_include_self=bool(cfg.network.get("tarmac_include_self", True)),
    )


def build_evaluation_runtime(
    cfg: DictConfig,
    checkpoint_path: str | Path,
    *,
    rng_seed: int = 0,
) -> EvaluationRuntime:
    """Build evaluation dependencies and restore one checkpoint."""
    environment = build_environment_runtime(cfg)
    model = build_model(cfg, rng_seed=rng_seed)
    resolved_checkpoint = restore_model_checkpoint(model, checkpoint_path)
    return EvaluationRuntime(
        model=model,
        environment=environment,
        checkpoint_path=resolved_checkpoint,
    )
