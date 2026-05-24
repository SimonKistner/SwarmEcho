"""
test_square_maps.py
===================
Validates the B-series curriculum maps and level configurations.

Usage:
    python src/tests/test_square_maps.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

from core.config import load_config, validate_config
from env.physics import make_env_fns
from env.maps import MapDefinition


def run_tests():
    print("============================================================")
    print("  SwarmEcho B-Series & Spawn Logic Tests")
    print("============================================================")

    # 1. Test A-series backward compatibility
    print("\n--- Testing A-Series Load ---")
    try:
        cfg_a = load_config(cli_overrides=True, overrides=["level=A00"])
        validate_config(cfg_a)
        print(f"  [OK] A00 loaded (agents={cfg_a.env.num_agents}, comm_r_base={cfg_a.env.comm_radius_base})")
    except Exception as e:
        print(f"  [FAIL] A00 failed to load: {e}")

    # 2. Test B-series loading and dynamic properties
    print("\n--- Testing B-Series Config & Dynamic Radius ---")
    try:
        cfg_b = load_config(cli_overrides=True, overrides=["level=B01"])
        validate_config(cfg_b)
        print(f"  [OK] B01 loaded (agents={cfg_b.env.num_agents})")
        print(f"       target_spawn_radius={cfg_b.env.target_spawn_radius} (from YAML before dynamic computation)")
        
        # Verify dynamic radius is computed properly inside make_env_fns
        _, reset_fn, _, _ = make_env_fns(cfg_b)
        
        # Compute expected radius: (50 - 5) + (50 - 5) * 1 = 45 + 45 = 90
        # For N=1, comm_r=50, comm_r_base=50.
        expected_r = (cfg_b.env.comm_radius_base - 5.0) + (cfg_b.env.comm_radius - 5.0) * cfg_b.env.num_agents
        
        key = jax.random.PRNGKey(42)
        state = reset_fn(key)
        state.pos.block_until_ready()
        
        print("  [OK] Physics reset executed successfully.")
        
        # Base should be at center: side=2*(90+10)=200, center=100
        expected_side = 2.0 * (expected_r + 10.0)
        expected_center = expected_side / 2.0
        
        bx, by = float(state.base_pos[0]), float(state.base_pos[1])
        if abs(bx - expected_center) < 1e-4 and abs(by - expected_center) < 1e-4:
            print(f"  [OK] Base spawned at center ({bx:.1f}, {by:.1f})")
        else:
            print(f"  [FAIL] Base spawned at ({bx:.1f}, {by:.1f}), expected ({expected_center}, {expected_center})")
            
        # Drones should be stacked at base
        d0x, d0y = float(state.pos[0, 0]), float(state.pos[0, 1])
        if abs(d0x - bx) < 1e-4 and abs(d0y - by) < 1e-4:
            print(f"  [OK] Drone spawned exactly at base position.")
        else:
            print(f"  [FAIL] Drone spawned at ({d0x:.1f}, {d0y:.1f}), expected ({bx:.1f}, {by:.1f})")
            
        # Target should be within target_spawn_radius
        tx, ty = float(state.target_pos[0]), float(state.target_pos[1])
        dist_to_base = ((tx - bx)**2 + (ty - by)**2)**0.5
        if dist_to_base <= expected_r + 1e-4:
            print(f"  [OK] Target spawned within radius (dist={dist_to_base:.1f}, max={expected_r:.1f})")
        else:
            print(f"  [FAIL] Target distance {dist_to_base:.1f} > max {expected_r:.1f}")

    except Exception as e:
        print(f"  [FAIL] B01 tests failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n============================================================")
    print("  Tests Complete.")
    print("============================================================")

if __name__ == "__main__":
    run_tests()
