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
    
    # Parse levels from sys.argv if present (e.g. levels=B02,B03,B04,B05,B05b)
    levels = ["00", "01", "02"]
    filtered_args = []
    for arg in sys.argv[1:]:
        if arg.startswith("levels="):
            val = arg.split("levels=", 1)[1]
            levels = [lvl.strip() for lvl in val.split(",")]
        else:
            filtered_args.append(arg)

    out_dir = Path("outputs/tests/levels")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 Starting Multi-Level Visual Verification...")
    print(f"Results will be saved to: {out_dir}\n")
    print(f"Levels to test: {', '.join(levels)}")
    print("-" * 50)

    for level_id in levels:
        print(f"--- Processing Level {level_id} ---")
        
        # 1. Load config for this level
        cfg = load_config(cli_overrides=False)
        OmegaConf.set_readonly(cfg, False)
        
        # Resolve level-specific YAML dynamically (glob)
        level_dir = Path("src/curriculum_config/levels")
        matches = (
            list(level_dir.glob(f"{level_id}_*.yaml"))
            + list(level_dir.glob(f"{level_id}_*.yml"))
            # Fallback: any file starting with the prefix (A-series)
            + list(level_dir.glob(f"{level_id}*.yaml"))
            + list(level_dir.glob(f"{level_id}*.yml"))
        )
        seen = set()
        matches = [m for m in matches if not (m in seen or seen.add(m))]
        
        if matches:
            level_cfg_path = matches[0]
            print(f"  Found level config: {level_cfg_path.name}")
            level_overrides = OmegaConf.load(level_cfg_path)
            cfg = OmegaConf.merge(cfg, level_overrides)
        else:
            print(f"  ⚠️ Warning: No level config found matching '{level_id}', using base defaults")
        
        cfg.visualize.renderer = "fast"
        cfg.logging.suppress_xla_warnings = True
        
        # Ensure some movement
        cfg.env.spawn_delay = 0
        
        env_step, reset, _, _ = make_env_fns(cfg)
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
        
        # 3. Render Both Backends
        traj_stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *trajectory)
        
        # --- Generate Beautiful Mock Data to Show Subplots & Metrics ---
        import numpy as np
        T_len = len(trajectory)
        N_agents = cfg.env.num_agents
        
        # 1. Generate individual agent rewards (shape: T_len, N_agents)
        mock_rewards = np.zeros((T_len, N_agents))
        for t in range(T_len):
            # Base exploration reward + random noise
            mock_rewards[t] = 0.015 + np.random.normal(0.0, 0.03, size=N_agents)
            # Simulated collision penalties early in the episode
            if t < 25:
                mock_rewards[t, :min(2, N_agents)] -= 0.5
            # Simulated target discovery at step 75
            if t == 75:
                mock_rewards[t, 0] += 125.0  # finder bonus
                mock_rewards[t] += 250.0 / N_agents  # global success bonus
        
        # 2. Generate smooth extra metrics (each shape: T_len)
        mock_r_explor = np.cumsum(0.015 + np.random.uniform(0.0, 0.01, size=T_len))
        mock_r_gap = np.clip(5.0 - np.linspace(0.0, 4.5, T_len) + np.random.normal(0.0, 0.1, size=T_len), 0.0, 5.0)
        mock_chain_pct = np.clip(np.linspace(0.0, 100.0, T_len) + np.random.normal(0.0, 4.0, size=T_len), 0.0, 100.0)
        if T_len > 75:
            mock_chain_pct[75:] = 100.0
        mock_r_total = mock_r_explor - mock_r_gap
        
        mock_extra_metrics = {
            "r_explor": mock_r_explor,
            "r_gap": mock_r_gap,
            "chain_pct": mock_chain_pct,
            "r_total": mock_r_total,
        }

        # Fast (CV2)
        cfg.visualize.renderer = "fast"
        fname_fast = f"level_{level_id}_smoke_test_fast.mp4"
        vpath_fast = render_video(
            traj_stacked, cfg, filename=str(out_dir / fname_fast),
            rewards=mock_rewards, extra_metrics=mock_extra_metrics
        )
        print(f"  ✓ Fast Video (CV2)      : {vpath_fast}")
        
        # Premium (Matplotlib)
        cfg.visualize.renderer = "slow"
        fname_slow = f"level_{level_id}_smoke_test_premium.mp4"
        vpath_slow = render_video(
            traj_stacked, cfg, filename=str(out_dir / fname_slow),
            rewards=mock_rewards, extra_metrics=mock_extra_metrics
        )
        print(f"  ✓ Premium Video (MPL)   : {vpath_slow}\n")

    print(f"✅ All levels verified in {(datetime.datetime.now() - t_start).total_seconds():.2f}s.")

if __name__ == "__main__":
    test_levels()


