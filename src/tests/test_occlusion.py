"""
scripts/test_occlusion.py
==========================
Comprehensive visual test suite for SwarmEcho physics & visuals.
Verifies:
1. Universal Occlusion (coverage and comms blocked by walls)
2. Ghost Entity Suppression (no links to disabled bases/targets)
3. Both Renderer Engines (CV2 Fast & Matplotlib Premium)
"""

import sys
import dataclasses
from pathlib import Path
import datetime

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

from core.config import load_config
from env.physics import make_env_fns
from visualize.renderer import render_video

def test_visual_occlusion():
    levels = ["00", "01", "02"]
    renderers = ["fast", "slow"]
    out_dir = Path("outputs/tests/occlusion")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 Starting Multi-Level, Multi-Renderer Stress Test...")
    print(f"Force: [20.0, 10.0] (Top-Right)")

    for level_id in levels:
        print(f"\n--- Checking Level {level_id} ---")
        
        # 1. Load config for this level
        cfg = load_config(cli_overrides=False)
        OmegaConf.set_readonly(cfg, False)
        
        level_cfg_path = Path("src/curriculum_config/levels") / f"{level_id}.yaml"
        if level_cfg_path.exists():
            level_overrides = OmegaConf.load(level_cfg_path)
            cfg = OmegaConf.merge(cfg, level_overrides)
        
        cfg.env.spawn_delay = 0
        cfg.logging.suppress_xla_warnings = True

        env_step, reset, _, _ = make_env_fns(cfg)
        key = jax.random.PRNGKey(123)
        state = reset(key)
        
        # Special Maze Placement
        if level_id == "02":
            new_pos = np.array(state.pos)
            new_pos[0] = [40.0, 40.0]
            new_pos[1] = [110.0, 40.0]
            state = dataclasses.replace(state, pos=jnp.array(new_pos))
        
        # 2. High-Performance Rollout
        def scan_step(carry, _):
            actions = jnp.tile(jnp.array([20.0, 10.0]), (cfg.env.num_agents, 1))
            new_state = env_step(carry, actions)
            return new_state, new_state
            
        rollout_fn = jax.jit(lambda s: jax.lax.scan(scan_step, s, None, length=150))
        print(f"  Simulating 150 steps (GPU)...")
        _, traj_stacked = rollout_fn(state)
            
        # 3. Render with both engines
        for r_type in renderers:
            print(f"  Rendering [{r_type}]...")
            cfg.visualize.renderer = r_type
            fname = f"stress_L{level_id}_{r_type}.mp4"
            vpath = render_video(traj_stacked, cfg, filename=str(out_dir / fname))
            print(f"    ✓ {vpath}")

    print(f"\n✅ All tests complete.")

if __name__ == "__main__":
    test_visual_occlusion()


