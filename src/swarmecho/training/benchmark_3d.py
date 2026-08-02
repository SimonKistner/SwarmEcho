"""CPU/CUDA benchmark and verification harness for the minimum 3D environment."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax

from swarmecho.env.baseline3d import (
    Baseline3DConfig,
    make_baseline_3d_fns,
    maximum_chain_distance,
)
from swarmecho.env.buildings import load_building, make_cuboid_building


DEFAULT_BUILDING = Path(__file__).parents[1] / "curriculum_config/maps/M00_no_maze_open_cuboid.yaml"


def run_benchmark(
    num_envs: int,
    steps: int,
    radar_bins: int,
    grid: tuple[int, int, int] = (4, 4, 4),
) -> dict:
    building = (
        load_building(DEFAULT_BUILDING)
        if grid == (4, 4, 4)
        else make_cuboid_building(grid)
    )
    cfg = Baseline3DConfig(radar_bins=radar_bins)
    reset, step, observations, _ = make_baseline_3d_fns(building, cfg)
    keys = jax.random.split(jax.random.PRNGKey(0), num_envs)
    actions = jax.random.uniform(
        jax.random.PRNGKey(1),
        (num_envs, steps, cfg.num_agents, 3),
        minval=-1.0,
        maxval=1.0,
    )

    def rollout(key, action_sequence):
        initial = reset(key)

        def transition(state, action):
            next_state = step(state, action)
            return next_state, observations(next_state)

        return jax.lax.scan(transition, initial, action_sequence)

    rollout_batch = jax.jit(jax.vmap(rollout))

    start = time.perf_counter()
    states, observations_batch = rollout_batch(keys, actions)
    jax.block_until_ready(observations_batch)
    compile_seconds = time.perf_counter() - start

    start = time.perf_counter()
    states, observations_batch = rollout_batch(keys, actions)
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
        "grid": list(grid),
        "max_base_to_top_corner_m": building.max_base_to_top_corner_m,
        "ideal_chain_reach_m": maximum_chain_distance(cfg),
        "ideal_chain_margin_m": maximum_chain_distance(cfg)
        - building.max_base_to_top_corner_m,
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
    parser.add_argument("--grid", default="4x4x4")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    environment_counts = [int(value) for value in args.num_envs.split(",")]
    radar_counts = [int(value) for value in args.radar_bins.split(",")]
    grids = [tuple(int(axis) for axis in value.split("x")) for value in args.grid.split(",")]
    if any(len(grid) != 3 for grid in grids):
        parser.error("--grid entries must use XxYxZ, for example 12x12x8")
    reports = [
        run_benchmark(num_envs, args.steps, radar_bins, grid)
        for num_envs in environment_counts
        for radar_bins in radar_counts
        for grid in grids
    ]
    report = reports[0] if len(reports) == 1 else {"benchmarks": reports}
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
