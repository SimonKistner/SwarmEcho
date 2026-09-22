"""Export training-generated buildings and the existing raster extreme routes.

Run: uv run python -m swarmecho.analysis.generate_random_maps
"""
import argparse
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from swarmecho.core.config import load_level
from swarmecho.env.random_buildings import generate_maps
from swarmecho.analysis.roadmap_scan import scan
from swarmecho.training.artifacts import write_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", default="B02_random_buildings")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, help="Override training.seed")
    parser.add_argument("--spacing", type=float, help="Raster spacing in metres, default cell size")
    parser.add_argument("--output-dir", default="outputs/testresults")
    parser.add_argument("--maps-only", action="store_true", help="Skip the roadmap raster scan")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Level override, repeatable")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    overrides = args.set + ([] if args.seed is None else [f"training.seed={args.seed}"])
    level = load_level(args.level, overrides)
    if not level.random_buildings.enabled:
        parser.error("The level must enable random_buildings")
    root = Path(args.output_dir) / f"random_buildings_{datetime.now():%Y%m%d_%H%M%S_%f}_seed{level.training.seed}"
    maps_dir = root / "maps"
    _, manifest = generate_maps(level.random_buildings, level.training.seed, args.count,
        directory=maps_dir, log=print, drone_clearance=level.env.drone_radius + level.env.obstacle_planning_clearance_m)
    write_manifest(root / "generation.json", manifest)
    # Persist effective level settings so scans use precisely the training cfg.
    from dataclasses import asdict
    import yaml
    effective = {key: asdict(getattr(level, key)) for key in
                 ("env", "reward", "training", "network", "evaluation", "logging", "random_buildings")}
    effective["env"]["map_names"] = []
    settings_path = root / "generation_level.yaml"
    settings_path.write_text(yaml.safe_dump(effective, sort_keys=False), encoding="utf-8")
    if not args.maps_only:
        for record in manifest["maps"]:
            path = (maps_dir / f"{record['id']}.yaml").resolve()
            scan(SimpleNamespace(level=str(settings_path), map=str(path), spacing=args.spacing,
                                 paths=1, corner_bonus_m=None, output_dir=str(root)))
    print(f"Inspectable maps and routes: {root.resolve()}")
    print(f"Maze builder: uv run swarmecho-maze-builder --map-dir {maps_dir.resolve()}")


if __name__ == "__main__":
    main()
