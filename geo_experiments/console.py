"""Use the training console formatter without importing the JAX runtime."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_source = Path(__file__).resolve().parents[1] / "src/swarmecho/core/terminal.py"
_spec = spec_from_file_location("_geo_training_terminal", _source)
_terminal = module_from_spec(_spec)
_spec.loader.exec_module(_terminal)
terminal_print = _terminal.terminal_print
