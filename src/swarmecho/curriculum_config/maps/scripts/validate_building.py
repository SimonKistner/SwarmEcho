"""Validate a builder map with the same contracts consumed by training."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from swarmecho.env.buildings import load_building


def validate_building_file(path: str | Path, *, runtime_smoke: bool = False) -> dict[str, object]:
    """Load a map and optionally construct/reset the CPU runtime."""
    source = Path(path)
    building = load_building(source)
    report: dict[str, object] = {
        "path": str(source),
        "world_size_m": building.world_size_m.tolist(),
        "interior_cells": int(building.interior_cells.sum()),
        "target_candidate_cells": int(
            (building.interior_cells & ~building.target_exclusion).sum()
        ),
        "authored_solids": int(building.solid_min_m.shape[0]),
        "runtime_smoke": False,
    }
    if runtime_smoke:
        import jax

        from swarmecho.env.environment import EnvConfig, make_env_fns

        cfg = EnvConfig(
            num_agents=2,
            comm_radius_base=0.1,
            comm_radius=0.1,
            visual_radius=0.1,
            target_spawn_buffer=0.0,
            coverage_voxel_size=building.cell_size_m,
            max_steps=2,
        )
        reset, step, observations, metrics = make_env_fns(building, cfg)
        state = reset(jax.random.PRNGKey(0))
        observation = observations(state)
        next_state = step(state, np.zeros((cfg.num_agents, 3), dtype=np.float32))
        runtime_metrics = metrics(next_state)
        report.update(
            {
                "runtime_smoke": True,
                "observation_shape": list(observation.shape),
                "target_position_m": np.asarray(state.target_pos).tolist(),
                "coverage_fraction": float(runtime_metrics["coverage_fraction"]),
            }
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a SwarmEcho v1 building map before creating a level."
    )
    parser.add_argument("map", help="Path to the building YAML produced by the editor")
    parser.add_argument(
        "--runtime-smoke",
        action="store_true",
        help="also construct and reset the CPU JAX environment (no training or CUDA)",
    )
    args = parser.parse_args()
    report = validate_building_file(args.map, runtime_smoke=args.runtime_smoke)
    print("VALID BUILDING")
    for key, value in report.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
