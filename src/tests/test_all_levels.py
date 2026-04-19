"""
scripts/test_all_levels.py
==========================
One-click visual verification for all curriculum stages.
Renders a short 100-step sequence for each map to verify that
the new 'Local Spotlight' raycasting and coverage logic 
are consistent across stages.
"""

import sys
import os
from pathlib import Path
import datetime

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

from core.config import load_config
from env.physics import make_env_fns
from visualize.renderer import render_video

def test_levels():
    t_start = datetime.datetime.now()
    levels = ["00", "01", "02"]
    out_dir = Path("outputs/tests/levels")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 Starting Multi-Level Visual Verification...")
    print(f"Results will be saved to: {out_dir}\n")

    for level_id in levels:
        print(f"--- Processing Level {level_id} ---")
        
        # 1. Load config for this level
        cfg = load_config(cli_overrides=False)
        OmegaConf.set_readonly(cfg, False)
        
        level_cfg_path = Path("src/curriculum_config/levels") / f"{level_id}.yaml"
        if level_cfg_path.exists():
            level_overrides = OmegaConf.load(level_cfg_path)
            cfg = OmegaConf.merge(cfg, level_overrides)
        
        cfg.visualize.renderer = "fast"
        cfg.logging.suppress_xla_warnings = True
        
        # Ensure some movement
        cfg.env.spawn_delay = 0
        
        env_step, reset, _ = make_env_fns(cfg)
        key = jax.random.PRNGKey(42)
        state = reset(key)
        
        trajectory = []
        
        # 2. Run random-ish simulation
        for s in range(100):
            # Random exploration-like force
            key, subkey = jax.random.split(key)
            actions = jax.random.uniform(subkey, (cfg.env.num_agents, 2), minval=-20.0, maxval=20.0)
            state = env_step(state, actions)
            trajectory.append(state)
        
        # 3. Render
        traj_stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *trajectory)
        fname = f"level_{level_id}_smoke_test"
        vpath = render_video(traj_stacked, cfg, filename=str(out_dir / fname))
        print(f"  ✓ Video: {vpath}\n")

    print(f"✅ All levels verified in {(datetime.datetime.now() - t_start).total_seconds():.2f}s.")

if __name__ == "__main__":
    test_levels()


