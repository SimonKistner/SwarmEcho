"""
Render a short static preview of the eight-agent memory T-maze map.

This is intentionally not a test assertion. It produces a five-frame fast
renderer video so the map layout, spawn points, targets, and base can be
inspected before designing the actual memory benchmark.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.config import load_config, validate_config
from env.physics import make_env_fns
from visualize.renderer import render_video


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    level_path = root / "src" / "curriculum_config" / "levels" / "MEM_T8_memory.yaml"
    cfg = load_config(level_path, cli_overrides=False)
    validate_config(cfg)

    _env_step, reset, _update_coverage, _meta = make_env_fns(cfg)
    state = reset(jax.random.PRNGKey(0))

    frames = [state] * 5
    stacked = jax.tree.map(lambda *xs: np.array(jnp.stack(xs)), *frames)
    traj = SimpleNamespace(**{f.name: getattr(stacked, f.name) for f in dataclasses.fields(stacked)})

    out_path = root / "outputs" / "memory_preview" / "memory_t_maze_8_static.mp4"
    rewards = np.zeros((5, int(cfg.env.num_agents)), dtype=np.float32)
    video_path = render_video(
        traj,
        cfg,
        filename=out_path,
        fps=2,
        renderer="fast",
        rewards=rewards,
        extra_metrics=None,
    )
    print(video_path)


if __name__ == "__main__":
    main()
