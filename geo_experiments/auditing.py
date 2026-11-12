"""Refresh saved target-location diagnostics (compatibility CLI)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geo_experiments.shared.diagnostics.auditing import main

if __name__ == "__main__":
    main()
