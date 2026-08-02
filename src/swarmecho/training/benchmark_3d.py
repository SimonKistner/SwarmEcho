"""CPU/CUDA benchmark and verification harness for the minimum 3D environment."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from swarmecho.env.baseline3d import Baseline3DConfig, make_baseline_3d_fns
from swarmecho.env.buildings import load_building


DEFAULT_BUILDING = Path(__file__).parents[1] / "curriculum_config/buildings/B00_baseline_cuboid.yaml"


def run_benchmark(num_envs: int, steps: int, radar_bins: int) -> dict:
    building = load_building(DEFAULT_BUILDING)
    cfg = Baseline3DConfig(radar_bins=radar_bins)
    reset, step, observations, _ = make_baseline_3d_fns(building, cfg)
    reset_batch = jax.jit(jax.vmap(reset))
    step_batch = jax.jit(jax.vmap(step))
    obs_batch = jax.jit(jax.vmap(observations))
    keys = jax.random.split(jax.random.PRNGKey(0), num_envs)
    actions = jnp.zeros((num_envs, cfg.num_agents, 3), dtype=jnp.float32)

    start = time.perf_counter()
    states = reset_batch(keys)
    observations_batch = obs_batch(states)
    jax.block_until_ready(observations_batch)
    compile_seconds = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(steps):
        states = step_batch(states, actions)
    jax.block_until_ready(states.pos)
    run_seconds = time.perf_counter() - start
    transitions = num_envs * steps
    memory_stats = jax.devices()[0].memory_stats() or {}
    return {
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "jax_version": jax.__version__,
        "num_envs": num_envs,
        "steps": steps,
        "radar_bins": radar_bins,
        "compile_seconds": compile_seconds,
        "run_seconds": run_seconds,
        "environment_steps_per_second": transitions / run_seconds,
        "device_memory_bytes_in_use": memory_stats.get("bytes_in_use"),
        "device_memory_peak_bytes_in_use": memory_stats.get("peak_bytes_in_use"),
        "coverage_shape": list(states.coverage.shape),
        "observation_shape": list(observations_batch.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", default="256")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--radar-bins", default="8")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    environment_counts = [int(value) for value in args.num_envs.split(",")]
    radar_counts = [int(value) for value in args.radar_bins.split(",")]
    reports = [
        run_benchmark(num_envs, args.steps, radar_bins)
        for num_envs in environment_counts
        for radar_bins in radar_counts
    ]
    report = reports[0] if len(reports) == 1 else {"benchmarks": reports}
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
