"""
generate_b_curriculum.py
========================
Generates B-series curriculum level YAMLs and their corresponding square
map YAMLs for the SwarmEcho relay-chain environment.

Each B-level represents a progressively harder relay-chain task:
  - The world is a perfect square
  - The base spawns exactly at the map centre (no random)
  - Drones spawn stacked at the base with a spawn_delay stagger
  - The target spawns uniformly inside a ring around the base:
    - min radius: comm_radius_base + (comm_radius * (num_agents - 1)) - 10
    - max radius: comm_radius_base + (comm_radius * (num_agents - 1)) - num_agents
    
Map dimensions: side = 2 * (max_target_radius + 10) metres

Level naming: B01_agents_in_square_50m_comm_50m_base
Map   naming: square_B01_50m_comm_50m_base

The comm radii are encoded in the file names so changing them always
produces a new, distinguishable file (the old one is not overwritten unless
--overwrite is passed).

Usage
-----
    # Generate B01 through B05 with default radii (comm=50, base_comm=50)
    python generate_b_curriculum.py

    # Custom radii and agent range
    python generate_b_curriculum.py --max-agents 8 --comm-radius 60 --comm-radius-base 60

    # Dry-run (print what would be created, write nothing)
    python generate_b_curriculum.py --dry-run

    # Force overwrite existing files
    python generate_b_curriculum.py --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_ROOT   = _SCRIPT_DIR.parent.parent          # .../src
_PROJECT_ROOT = _SRC_ROOT.parent                 # .../SwarmEcho
sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.config import load_config

_LEVELS_DIR = _SRC_ROOT / "curriculum_config" / "levels"
_MAPS_DIR   = _SRC_ROOT / "curriculum_config" / "maps"


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------

def generate_level(
    n_agents: int,
    comm_r: float = 50.0,
    comm_r_base: float = 50.0,
    max_steps: int | None = None,
    num_envs: int = 4096,
    num_steps: int = 64,
    num_epochs: int = 10,
    num_minibatches: int = 16,
    total_timesteps: int | None = None,
    spawn_delay: int = 3,
    success_threshold: float = 0.95,
    dry_run: bool = False,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """
    Generate a single B-series level YAML and its square map YAML.

    Parameters
    ----------
    n_agents : int
        Number of relay drones for this level.
    comm_r : float
        Drone-to-drone communication radius (metres).
    comm_r_base : float
        Base-station communication radius for first-hop (metres).
    max_steps : int, optional
        Episode length.  Defaults to 400 + 100 * n_agents.
    spawn_delay : int
        Steps between successive drone activations (written into level YAML).
        Defaults to 3 (B-series default; global config default stays 0).
    success_threshold : float
        Success rate (0-1) written into level YAML as curriculum.success_threshold.
        The curriculum runner reads this to trigger early level advancement.
        Use 0.0 to omit the key entirely (timesteps-only transition).
    dry_run : bool
        If True, print what would be created but write nothing.
    overwrite : bool
        If False (default), skip generation if the level file already exists.

    Returns
    -------
    (level_path, map_path) - the paths that were (or would be) written.
    """
    # ---- Derived geometry -------------------------------------------------
    min_radius = max(0.0, comm_r_base + (comm_r * (n_agents - 1)) - 10.0)
    max_radius = comm_r_base + (comm_r * (n_agents - 1)) - float(n_agents)
    
    side   = 2.0 * (max_radius + 10.0)
    center = side / 2.0

    if max_steps is None:
        # Scales as: 1000, 3000, 6000, 10000, 15000...
        max_steps = 100 * n_agents * (n_agents + 1)
    if total_timesteps is None:
        total_timesteps = 500_000_000

    # ---- Naming: encode comm radii so name is unique per param set --------
    comm_r_int      = int(comm_r)
    comm_r_base_int = int(comm_r_base)
    level_id        = f"B{n_agents:02d}"
    radius_suffix   = f"{comm_r_int}m_comm_{comm_r_base_int}m_base"
    map_name        = f"square_{level_id}_{radius_suffix}"
    level_stem      = f"{level_id}_agents_in_square_{radius_suffix}"

    map_path   = _MAPS_DIR   / f"{map_name}.yaml"
    level_path = _LEVELS_DIR / f"{level_stem}.yaml"

    # ---- Check existing ---------------------------------------------------
    if not overwrite and level_path.exists():
        print(f"  [gen] {level_path.name} already exists — skipping.")
        return level_path, map_path

    # ---- Map YAML ---------------------------------------------------------
    # Spawn zones are near-center as a readable hint / fallback.
    # Actual spawn logic is controlled by use_random_*_spawn flags in the level.
    half_cell = 0.5
    map_data = {
        "name": map_name,
        "width":  round(side, 2),
        "height": round(side, 2),
        "spawn_zones": {
            "base": [
                round(center - half_cell, 2), round(center - half_cell, 2),
                round(center + half_cell, 2), round(center + half_cell, 2),
            ],
            "target": [0.0, 0.0, round(side, 2), round(side, 2)],
            "drone": [
                round(center - half_cell, 2), round(center - half_cell, 2),
                round(center + half_cell, 2), round(center + half_cell, 2),
            ],
        },
        "rooms":    [],
        "hallways": [],
        "walls":    [],
    }

    # ---- Level YAML -------------------------------------------------------
    level_data: dict = {
        "env": {
            "map_names":                   [map_name],
            "num_agents":                  n_agents,
            "max_steps":                   max_steps,
            "use_random_base_spawn":       False,
            "use_random_drone_spawn":      False,
            "target_spawn_method":         "ring",
            "target_spawn_radius":         round(max_radius, 2),
            "target_spawn_radius_min":     round(min_radius, 2),
        },
        "training": {
            "total_timesteps": total_timesteps,
            "num_envs":        num_envs,
            "num_steps":       num_steps,
            "num_epochs":      num_epochs,
            "num_minibatches": num_minibatches,
        },
        "logging": {
            "video_freq": 50,
        },
    }

    # success_threshold is an explicit B-series override (not in the global default)
    if success_threshold > 0.0:
        level_data["curriculum"] = {"success_threshold": success_threshold}

    # ---- Write / dry-run --------------------------------------------------
    if dry_run:
        print(f"  [dry-run] MAP  : {map_path.name}")
        print(f"            radius: {min_radius:.1f}m -> {max_radius:.1f}m, side={side:.1f}m")
        print(f"  [dry-run] LEVEL: {level_path.name}")
        print(f"            n_agents={n_agents}, spawn_delay={spawn_delay}, "
              f"max_steps={max_steps}, total_ts={total_timesteps:,}, "
              f"success_threshold={success_threshold}")
    else:
        _MAPS_DIR.mkdir(parents=True, exist_ok=True)
        _LEVELS_DIR.mkdir(parents=True, exist_ok=True)

        with open(map_path, "w", encoding="utf-8") as f:
            yaml.dump(map_data, f, default_flow_style=False, sort_keys=False)

        with open(level_path, "w", encoding="utf-8") as f:
            f.write(f"# Level B{n_agents:02d}\n#\n")
            # Write env part
            env_str = yaml.dump({"env": level_data["env"]}, default_flow_style=False, sort_keys=False)
            env_str = env_str.replace(f"  map_names:\n  - {map_name}", f"  map_names: [\"{map_name}\"]")
            f.write(env_str)
            f.write("\n# 128, 256, 512, 1024, 2048, 4096, 8192, 16384\n\n")
            
            # Write training part
            train_str = yaml.dump({"training": level_data["training"]}, default_flow_style=False, sort_keys=False)
            train_str = train_str.replace("total_timesteps: 500000000", "total_timesteps: 500_000_000")
            f.write(train_str)
            f.write("\n")
            
            if "curriculum" in level_data:
                yaml.dump({"curriculum": level_data["curriculum"]}, f, default_flow_style=False, sort_keys=False)
                f.write("\n")
                
            yaml.dump({"logging": level_data["logging"]}, f, default_flow_style=False, sort_keys=False)

        print(f"  [gen] MAP   written: {map_path.name}  (side={side:.0f}m, radius: {min_radius:.1f}->{max_radius:.1f}m)")
        print(f"  [gen] LEVEL written: {level_path.name}  (n={n_agents}, max_steps={max_steps})")

    return level_path, map_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    # Load config.py defaults without parsing CLI overrides (to avoid conflicts with script args)
    cfg = load_config(cli_overrides=False)
    
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--max-nr-agents",    type=int,   default=5,
                   help="Maximum agent count to generate up to (default 5)")
    p.add_argument("--comm-radius",      type=float, default=cfg.env.comm_radius,
                   help=f"Drone comm radius in metres (default {cfg.env.comm_radius})")
    p.add_argument("--comm-radius-base", type=float, default=cfg.env.comm_radius_base,
                   help=f"Base-station comm radius in metres (default {cfg.env.comm_radius_base})")
    p.add_argument("--spawn-delay",      type=int,   default=3,
                   help="Steps between drone activations (default 3)")
    p.add_argument("--success-threshold", type=float, default=0.95,
                   help="Success rate for level advancement (0=disabled, default 0.95)")
    p.add_argument("--num-envs",         type=int,   default=4096,
                   help="Parallel environments (default 4096)")
    p.add_argument("--dry-run",          action="store_true",
                   help="Print what would be created, write nothing")
    p.add_argument("--overwrite",        action="store_true",
                   help="Overwrite existing level/map files")
    return p.parse_args()


def main():
    args = _parse_args()

    print("\n" + "-"*60)
    print("  SwarmEcho B-Curriculum Generator")
    print(f"  agents : B02 -> B{args.max_nr_agents:02d}")
    print(f"  comm_r ={args.comm_radius}m  comm_r_base={args.comm_radius_base}m")
    print(f"  success_threshold={args.success_threshold}  spawn_delay={args.spawn_delay}")
    print(f"  dry_run={args.dry_run}  overwrite={args.overwrite}")
    print("-"*60)

    for n in range(2, args.max_nr_agents + 1):
        generate_level(
            n_agents=n,
            comm_r=args.comm_radius,
            comm_r_base=args.comm_radius_base,
            spawn_delay=args.spawn_delay,
            success_threshold=args.success_threshold,
            num_envs=args.num_envs,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )

    processed_count = args.max_nr_agents - 1 if args.max_nr_agents >= 2 else 0
    print(f"\n  Done -- {processed_count} levels processed.")


if __name__ == "__main__":
    main()
