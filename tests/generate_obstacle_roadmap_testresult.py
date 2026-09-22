"""Compatibility entry point; shared implementation lives in swarmecho.analysis."""
from swarmecho.analysis.roadmap_paths import *
from swarmecho.analysis.roadmap_paths import _shortest, _visible
from swarmecho.analysis.roadmap_paths import generate as _generate


def load_level(*args, **kwargs):
    from swarmecho.core.config import load_level as implementation
    return implementation(*args, **kwargs)


def generate_obstacles(*args, **kwargs):
    from swarmecho.env.obstacles import generate_obstacles as implementation
    return implementation(*args, **kwargs)


def generate(*args, **kwargs):
    # Retain the legacy module's injectable dependencies for existing callers.
    kwargs.setdefault("level_loader", load_level)
    kwargs.setdefault("obstacle_generator", generate_obstacles)
    return _generate(*args, **kwargs)

if __name__ == "__main__":
    main()
