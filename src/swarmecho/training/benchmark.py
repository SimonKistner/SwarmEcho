"""CPU/CUDA benchmark and verification harness for the minimum environment."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax
from swarmecho.core.config import MAP_DIR

from swarmecho.env.environment import (
    EnvConfig,
    make_env_fns,
    maximum_chain_distance,
)
from swarmecho.env.buildings import load_building, make_cuboid_building


DEFAULT_BUILDING = MAP_DIR / "M00_no_maze_open_cuboid.yaml"


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
    cfg = EnvConfig(radar_bins=radar_bins)
    reset, step, observations, _ = make_env_fns(building, cfg)
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
    options = {"num_envs": "256", "steps": "100", "radar_bins": "8", "grid": "4x4x4"}
    output: Path | None = None
    for argument in sys.argv[1:]:
        if "=" not in argument:
            raise ValueError("Benchmark options use key=value syntax.")
        key, value = argument.split("=", 1)
        if key == "output":
            output = Path(value)
        elif key in options:
            options[key] = value
        else:
            raise ValueError(f"Unknown benchmark option {key!r}.")
    environment_counts = [int(value) for value in options["num_envs"].split(",")]
    radar_counts = [int(value) for value in options["radar_bins"].split(",")]
    grids = [tuple(int(axis) for axis in value.split("x")) for value in options["grid"].split(",")]
    if any(len(grid) != 3 for grid in grids):
        raise ValueError("grid entries must use XxYxZ, for example 12x12x8")
    steps = int(options["steps"])
    reports = [
        run_benchmark(num_envs, steps, radar_bins, grid)
        for num_envs in environment_counts
        for radar_bins in radar_counts
        for grid in grids
    ]
    report = reports[0] if len(reports) == 1 else {"benchmarks": reports}
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if output:
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
