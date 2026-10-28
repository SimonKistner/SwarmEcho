"""Compatibility entry point; historical implementation lives under legacy/."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geo_experiments.legacy.run import main

from geo_experiments.execution import execute_target as run_target_experiment

if __name__ == "__main__":
    main()
