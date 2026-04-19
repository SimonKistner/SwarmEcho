"""
scripts/debug_maze_fast.py
==========================
Isolates Level 02 (Maze) to debug Drone 4 wall-phasing.
Renders with the Fast (CV2) engine.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

from core.config import load_config
from env.physics import make_env_fns
from visualize.renderer_cv2 import render_video_cv2

def debug_maze():
    level_id = "02"
    out_dir = Path("outputs/tests/debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🧩 Debugging Level {level_id} [Fast Renderer]...")

    # Load level 02 config
    cfg = load_config(cli_overrides=False)
    OmegaConf.set_readonly(cfg, False)
    
    level_cfg_path = Path("src/curriculum_config/levels") / f"{level_id}.yaml"
    if level_cfg_path.exists():
        level_overrides = OmegaConf.load(level_cfg_path)
        cfg = OmegaConf.merge(cfg, level_overrides)
    
    cfg.env.spawn_delay = 0
    cfg.visualize.renderer = "fast"
    cfg.logging.suppress_xla_warnings = True

    env_step, reset, _ = make_env_fns(cfg)
    key = jax.random.PRNGKey(42)
    state = reset(key)
    
    # Run 150 step rollout (Fast JAX loop)
    def scan_step(carry, _):
        # Apply constant force to trigger Drone 4 movement
        actions = jnp.tile(jnp.array([20.0, 10.0]), (cfg.env.num_agents, 1))
        new_state = env_step(carry, actions)
        return new_state, new_state
        
    rollout_fn = jax.jit(lambda s: jax.lax.scan(scan_step, s, None, length=150))
    print(f"  Simulating...")
    _, traj_stacked = rollout_fn(state)
        
    print(f"  Rendering [fast]...")
    fname = f"debug_L02_phasing.mp4"
    vpath = render_video_cv2(traj_stacked, cfg, filename=str(out_dir / fname))
    print(f"  ✓ {vpath}\n")

if __name__ == "__main__":
    debug_maze()


