"""Rescore the recorded B02c 180M/map 0001 trajectory with legacy gap reward.

Run from the repository root with the project's normal Python environment:
    uv run python tools/rescore_b02c_replay.py

The policy is never called. All recorded frames are copied unchanged; only the
gap component of each per-agent reward is exchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from swarmecho.core.config import EnvConfig, RewardConfig
from swarmecho.env.buildings import load_building
from swarmecho.env.environment import (
    EnvState, credited_chain_progress, vested_delivery_gap_credit,
)
from swarmecho.env.obstacles import building_roadmap
from swarmecho.visualize.replay import load_replay


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/solo_B02c_a4c4_15x15_rewardChange"
SOURCE = RUN / "artifacts/train/replays/eval_u000451_s00180M_random_eval_0001.json"
TARGET_CONFIG = ROOT / "src/swarmecho/curriculum_config/levels/B02c_random_buildings.yaml"
OUTPUT = (RUN / "artifacts/eval/replays"
          / "eval_u000451_s00180M_random_eval_0001_legacy_gap_fixed_trajectory")


def _credited_metres(arrays: dict[str, np.ndarray], env: EnvConfig,
                     reward: RewardConfig, building) -> np.ndarray:
    """Use the environment's own relay geometry and credit assignment."""
    geometry = building_roadmap(building, env, plan=True)

    def one(position, base, target, active, sees, conn_base, conn_target,
            known, connected, obstacle_min, obstacle_max):
        state = EnvState(
            pos=position, vel=jnp.zeros_like(position), base_pos=base,
            target_pos=target, active=active,
            coverage=jnp.zeros((1, 1, 1), dtype=jnp.bool_), step=jnp.int32(0),
            key=jnp.zeros((2,), dtype=jnp.uint32),
            directly_sees_target=sees, is_conn_base=conn_base,
            is_conn_target=conn_target, target_known=known,
            success=jnp.bool_(False), fully_connected=connected,
            chain_held_steps=jnp.int32(0), done=jnp.bool_(False),
            collided=jnp.zeros_like(active),
            coverage_credit=jnp.zeros_like(active, dtype=jnp.float32),
            obstacle_min=obstacle_min, obstacle_max=obstacle_max,
            shared_geometry=geometry,
        )
        return credited_chain_progress(
            state, env, chain_reward_system=reward.chain_reward_system,
            allow_redundancy_reward=reward.allow_redundancy_reward,
        )

    batched = jax.jit(jax.vmap(one))
    keys = ("position", "base_position", "target_position", "active",
            "directly_sees_target", "connected_to_base", "connected_to_target",
            "target_known", "fully_connected", "obstacle_min", "obstacle_max")
    count = len(arrays["position"])
    result = []
    for start in range(0, count, 32):
        indices = np.minimum(np.arange(start, start + 32), count - 1)
        result.append(np.asarray(batched(*(jnp.asarray(arrays[key][indices]) for key in keys))))
    return np.concatenate(result, axis=0)[:count]


def rescore() -> Path:
    source_manifest, arrays = load_replay(SOURCE)
    run_config = yaml.safe_load((RUN / "config.yaml").read_text(encoding="utf-8"))
    target_config = yaml.safe_load(TARGET_CONFIG.read_text(encoding="utf-8"))
    source_reward = RewardConfig(**run_config["reward"])
    target_reward = RewardConfig(**target_config["reward"])
    env = EnvConfig(**run_config["env"])
    changed = {name for name in source_reward.__dataclass_fields__
               if getattr(source_reward, name) != getattr(target_reward, name)}
    if changed - {"gap_reward_uses_change", "gap_change_meter_bonus",
                   "gap_reduction_meter_bonus", "delivery_gap_vesting_enabled",
                   "delivery_gap_vesting_steps"}:
        raise ValueError(f"Other reward settings changed; cannot swap only gap credit: {sorted(changed)}")
    if (not source_reward.gap_reward_uses_change
            or not source_reward.delivery_gap_vesting_enabled
            or target_reward.gap_reward_uses_change
            or target_reward.delivery_gap_vesting_enabled):
        raise ValueError("Expected a vested change source and a legacy gap target.")
    if not source_reward.target_found_requires_delivery:
        raise ValueError("This rescore expects delivery-gated gap reward.")
    if arrays["reward_terms"].shape != (source_manifest["frames"], source_manifest["agents"], 1):
        raise ValueError("Unexpected replay reward shape.")

    map_path = RUN / "random_eval_maps" / f"{source_manifest['map_name']}.yaml"
    building = load_building(map_path)
    snapshot = source_manifest["building_snapshot"]
    if (not np.allclose(building.solid_min_m, np.asarray(snapshot["solid_min_m"]), rtol=0, atol=1e-6)
            or not np.allclose(building.solid_max_m, np.asarray(snapshot["solid_max_m"]), rtol=0, atol=1e-6)):
        raise ValueError("Frozen evaluation map geometry differs from the replay snapshot.")

    credit = _credited_metres(arrays, env, source_reward, building)
    known = np.asarray(arrays["base_target_known"], dtype=bool).reshape(-1)
    done = np.asarray(arrays["done"], dtype=bool).reshape(-1)
    steps = np.asarray(arrays["step"], dtype=np.int32).reshape(-1)
    active = np.asarray(arrays["active"], dtype=bool)
    deliveries = np.flatnonzero(known & ~np.r_[False, known[:-1]])
    if len(deliveries) != 1:
        raise ValueError(f"Expected exactly one target delivery, found {len(deliveries)}.")
    delivered_at = int(deliveries[0])
    initial_credit = credit[delivered_at]
    elapsed = steps - steps[delivered_at]
    vested = np.array(vested_delivery_gap_credit(
        jnp.asarray(credit), jnp.asarray(initial_credit),
        jnp.asarray(elapsed[:, None]), source_reward.delivery_gap_vesting_steps,
        finish=jnp.asarray(done[:, None]),
    ), copy=True)
    vested[~known] = 0.0
    previous_vested = np.concatenate((np.zeros_like(vested[:1]), vested[:-1]))
    source_gap = source_reward.gap_change_meter_bonus * (vested - previous_vested)
    source_gap[~known] = 0.0
    target_gap = target_reward.gap_reduction_meter_bonus * credit
    target_gap[~known] = 0.0
    source_gap *= active
    target_gap *= active

    revised = np.asarray(arrays["reward_terms"], dtype=np.float32).copy()
    revised[:, :, 0] += (target_gap - source_gap).astype(np.float32)
    arrays["reward_terms"] = revised

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    data_path = OUTPUT.with_suffix(".npz")
    manifest_path = OUTPUT.with_suffix(".json")
    if data_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Comparison replay already exists: {manifest_path}")
    temporary_data = data_path.with_suffix(".npz.tmp")
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    with temporary_data.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary_data.replace(data_path)
    manifest = dict(source_manifest)
    manifest.update(
        data_file=data_path.name,
        fields=sorted(arrays),
        artifact_scope="eval",
        eval_name="legacy gap fixed trajectory",
        reward_rescore={
            "method": "recorded_state_gap_component_swap",
            "source_replay": str(SOURCE),
            "source_gap_mode": "vested_change",
            "target_gap_mode": "legacy_recurring",
            "source_gap_change_meter_bonus": source_reward.gap_change_meter_bonus,
            "target_gap_reduction_meter_bonus": target_reward.gap_reduction_meter_bonus,
            "source_gap_total_by_agent": source_gap.sum(axis=0).tolist(),
            "target_gap_total_by_agent": target_gap.sum(axis=0).tolist(),
            "nearest_saved_checkpoint": str(RUN / "checkpoints/ckpt_000450"),
            "note": "Uses exact source frames; no checkpoint policy or actions were rerun.",
        },
    )
    for key in ("random_eval_group", "random_eval_maps"):
        manifest.pop(key, None)
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    return manifest_path


if __name__ == "__main__":
    print(rescore())
