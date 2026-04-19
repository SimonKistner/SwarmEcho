"""
scripts/benchmark_renderers.py
==============================
Performance limit-test: Fast (CV2) vs Premium (Matplotlib).
Renders 1,000 frames and compares throughput (FPS).
"""

import sys
import time
from pathlib import Path
import dataclasses

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from core.config import load_config
from env.physics import make_env_fns
from visualize.renderer_cv2 import render_video_cv2
from visualize.renderer_mpl import render_video as render_video_mpl

def run_benchmark():
    level_id = "02" # High complexity Maze
    out_dir = Path("outputs/tests/benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 Benchmarking Renderers [1,000 Frames | Level {level_id}]")
    
    # 1. Setup Environment
    cfg = load_config(cli_overrides=False)
    OmegaConf.set_readonly(cfg, False)
    
    level_cfg_path = Path("src/curriculum_config/levels") / f"{level_id}.yaml"
    if level_cfg_path.exists():
        level_overrides = OmegaConf.load(level_cfg_path)
        cfg = OmegaConf.merge(cfg, level_overrides)
    
    cfg.env.spawn_delay = 0
    cfg.logging.suppress_xla_warnings = True
    
    env_step, reset, _ = make_env_fns(cfg)
    key = jax.random.PRNGKey(42)
    state = reset(key)
    
    # 2. 1,000 Step Simulation (JAX Scan)
    def scan_step(carry, _):
        # Move drones in a circle or something dynamic
        actions = jnp.tile(jnp.array([20.0, 5.0]), (cfg.env.num_agents, 1))
        new_state = env_step(carry, actions)
        return new_state, new_state
        
    rollout_fn = jax.jit(lambda s: jax.lax.scan(scan_step, s, None, length=1000))
    print(f"  Simulating 1,000 steps on GPU...")
    start_sim = time.perf_counter()
    _, traj_stacked = rollout_fn(state)
    jax.block_until_ready(traj_stacked) # Ensure simulation is done
    sim_time = time.perf_counter() - start_sim
    print(f"  ✓ Simulation Complete ({sim_time:.2f}s | {1000/sim_time:.1f} FPS)")

    results = []

    # 3. Benchmark Fast Renderer (CV2)
    print(f"\n  [1/2] Benchmarking FAST Renderer (OpenCV + Parallel)...")
    t0 = time.perf_counter()
    cv2_path = out_dir / "benchmark_fast_1000.mp4"
    render_video_cv2(traj_stacked, cfg, filename=str(cv2_path))
    t1 = time.perf_counter()
    cv2_time = t1 - t0
    cv2_fps = 1000 / cv2_time
    print(f"    ✓ CV2 Done: {cv2_time:.2f}s ({cv2_fps:.1f} FPS)")
    results.append(("Fast (CV2)", cv2_time, cv2_fps))

    # 4. Benchmark Premium Renderer (Matplotlib)
    print(f"\n  [2/2] Benchmarking PREMIUM Renderer (Matplotlib)...")
    # Note: MPL is currently single-threaded. This is the real test.
    t0 = time.perf_counter()
    mpl_path = out_dir / "benchmark_premium_1000.mp4"
    render_video_mpl(traj_stacked, cfg, filename=str(mpl_path))
    t1 = time.perf_counter()
    mpl_time = t1 - t0
    mpl_fps = 1000 / mpl_time
    print(f"    ✓ MPL Done: {mpl_time:.2f}s ({mpl_fps:.1f} FPS)")
    results.append(("Premium (MPL)", mpl_time, mpl_fps))

    # 5. Final Report
    print(f"\n" + "="*50)
    print(f"{'RENDERER':<20} | {'TIME (s)':<10} | {'FPS':<10}")
    print("-" * 50)
    for name, t, fps in results:
        print(f"{name:<20} | {t:<10.2f} | {fps:<10.1f}")
    
    speedup = mpl_time / cv2_time
    print("="*50)
    print(f"🚀 FAST (CV2) is {speedup:.1f}x faster than PREMIUM (MPL)!")
    print(f"Videos saved to: {out_dir}\n")

if __name__ == "__main__":
    run_benchmark()


