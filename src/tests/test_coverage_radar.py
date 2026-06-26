
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from env.state import EnvState
from env.observations import make_obs_fns

def test_coverage_radar():
    # 1. Setup a dummy config
    cfg = OmegaConf.create({
        "env": {
            "num_agents": 1,
            "num_targets": 0,
            "num_bases": 0,
            "box_width": 100.0,
            "box_height": 100.0,
            "grid_cell_size": 1.0,
            "max_grid_width": 100,
            "max_grid_height": 100,
            "radar_bins": 4,
            "visual_radius": 5.0,
            "comm_radius": 10.0,
            "max_speed": 10.0,
            "map_names": []
        },
        "network": {
            "radar_bins": 4
        }
    })

    occ = jnp.zeros((100, 100), dtype=jnp.bool_)
    obs_fn, obs_dim = make_obs_fns(cfg, 100.0, 100.0, occ)
    
    # 2. Create a state where the LEFT half of the world (x < 50) is covered
    # Grid size is 100x100
    coverage = jnp.zeros((100, 100), dtype=jnp.bool_)
    coverage = coverage.at[:50, :].set(True)
    
    # Place drone at (50, 50) - exactly on the boundary
    state = EnvState(
        pos           = jnp.array([[50.0, 50.0]], dtype=jnp.float32),
        vel           = jnp.zeros((1, 2), dtype=jnp.float32),
        base_pos      = jnp.array([0.0, 0.0], dtype=jnp.float32),
        target_pos    = jnp.array([100.0, 100.0], dtype=jnp.float32),
        active        = jnp.array([True], dtype=jnp.bool_),
        collides      = jnp.array([False], dtype=jnp.bool_),
        target_known  = jnp.array([False], dtype=jnp.bool_),
        coverage_grid = coverage,
        step          = jnp.array(0, dtype=jnp.int32),
        key           = jnp.array([0, 0], dtype=jnp.uint32),
        last_cov_delta= jnp.zeros(1, dtype=jnp.int32),
        box_width     = jnp.array(100.0, dtype=jnp.float32),
        box_height    = jnp.array(100.0, dtype=jnp.float32),
        base_target_known = jnp.bool_(False),
        chain_held_steps = jnp.int32(0),
        is_conn_base      = jnp.array([False], dtype=jnp.bool_),
        is_conn_target    = jnp.array([False], dtype=jnp.bool_),
    )
    
    obs = obs_fn(state) # (1, obs_dim)
    
    # The Local Coverage block is 16 bits (circular layout starting from North clockwise)
    # sampling_radius = 10.0
    # Drone at (50, 50)
    # Samples with x < 50 are covered (1.0).
    # Since boundary is at x = 50, samples with cos(angle) < 0 are covered.
    
    # Self-block is 5 dims (vel(2), conn_b, conn_t, known)
    # Local coverage starts at index 5.
    cov_bits = obs[0, 5:21]
    
    print(f"Drone at (50, 50), Left half covered.")
    print(f"Observed Coverage Bits: {cov_bits}")
    
    # Expected:
    # index 0 to 8: cos(angle) >= 0 -> False (0.0)
    # index 9 to 15: cos(angle) < 0 -> True (1.0)
    
    expected = jnp.array([0.0] * 9 + [1.0] * 7)
    
    if jnp.allclose(cov_bits, expected):
        print("✅ SUCCESS: Coverage radar correctly detects the boundary.")
    else:
        print("❌ FAILURE: Coverage radar mismatch.")
        print(f"Expected: {expected}")
        print(f"Actual:   {cov_bits}")

if __name__ == "__main__":
    test_coverage_radar()
