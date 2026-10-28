"""Compatibility entry point; historical implementation lives under legacy/."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geo_experiments.legacy.run import main

if __name__ == "__main__":
    main(default_config="own_cell.yaml")
