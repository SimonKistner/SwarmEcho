import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
import datetime
from pathlib import Path
from omegaconf import OmegaConf

# Add src directory to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.config import load_config, MAP_DIR
from env.physics import make_env_fns
from env.observations import make_obs_fns
from visualize.renderer_cv2 import render_video_cv2
from visualize.renderer_mpl import render_video

def prep_map(map_name, cfg_base):
    print(f"\n{'='*60}")
    print(f" PREPARING MAP: {map_name}")
    print(f"{'='*60}")
    
    # Update config for this map
    cfg = OmegaConf.to_container(cfg_base, resolve=True)
    cfg['env']['map_names'] = [map_name]
    # Small box dimensions as fallback if no map sizing logic, 
    # but physics.py handles it from map_def.
    cfg = OmegaConf.create(cfg)
    
    # 1. Initialize environment
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Initializing Env...")
    env_step, reset, _ = make_env_fns(cfg)
    compute_obs, _ = make_obs_fns(cfg)
    
    key = jax.random.PRNGKey(42)
    state = reset(key)
    
    # Run short trajectory
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Generating 20-step trajectory...")
    states = [state]
    for _ in range(20):
        # random actions
        key, subkey = jax.random.split(key)
        actions = jax.random.uniform(subkey, (cfg.env.num_agents, 2), minval=-1.0, maxval=1.0)
        state = env_step(state, actions)
        states.append(state)
    
    # Stack states
    trajectory = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
    
    # 2. Render CV2
    out_dir = Path("outputs/map_checks")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cv2_path = out_dir / f"{map_name}_cv2.mp4"
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Rendering CV2...")
    render_video_cv2(trajectory, cfg, cv2_path, fps=10)
    
    # 3. Render MPL
    mpl_path = out_dir / f"{map_name}_mpl.mp4"
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Rendering Matplotlib...")
    render_video(trajectory, cfg, mpl_path, fps=10)

def main():
    cfg = load_config(cli_overrides=False)
    
    map_dir = MAP_DIR
    map_files = list(map_dir.glob("*.yaml"))
    map_names = [f.stem for f in map_files]
    
    print(f"Found {len(map_names)} maps: {map_names}")
    
    for name in map_names:
        try:
            prep_map(name, cfg)
        except Exception as e:
            print(f"FAILED TO PREP MAP {name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()


