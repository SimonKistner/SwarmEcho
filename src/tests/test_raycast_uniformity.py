import sys
from pathlib import Path

# Add the 'src' directory to sys.path (matching other test scripts)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
from env.raycast import dda_raycast
from visualize.renderer_mpl import _dda_raycast_np

def run_test():
    print("Setting up test occupancy grid...")
    # Create a 100x100 grid
    grid_shape = (100, 100)
    occ_grid = np.zeros(grid_shape, dtype=bool)
    
    # Place a vertical wall segment at x = 50, from y = 40 to y = 60
    occ_grid[50, 40:61] = True
    
    # Convert grid to JAX array
    occ_grid_jax = jnp.array(occ_grid)
    
    # Define the explicit edge cases from the reported mismatches
    edge_cases = [
        ((25.0, 45.0), (75.0, 55.0), "Primary Diagonal Path"),
        ((3.59810843, 58.40600696), (93.14256366, 57.39646943), "Horizontal dominant crossing (Bug B)"),
        ((73.27512231, 45.31735821), (23.01127265, 45.33907258), "Horizontal dominant crossing reverse (Bug B)"),
        ((51.70974247, 43.18188621), (32.0607656,  43.59040416), "Horizontal crossing close to wall"),
        ((67.35782269, 60.36636645), (36.73018536, 22.35347032), "Diagonal clear path just below wall (Bug A)"),
        ((13.04622649, 88.08858987), (50.30222718, 45.03627472), "Diagonal crossing ending at wall border"),
    ]
    
    print("\nRunning explicit edge cases...")
    for idx, (p1, p2, name) in enumerate(edge_cases, 1):
        print(f"Edge case {idx} [{name}]: {p1} -> {p2}")
        np_res = _dda_raycast_np(p1, p2, occ_grid)
        jax_res = bool(dda_raycast(jnp.array(p1), jnp.array(p2), occ_grid_jax))
        print(f"  - NumPy clear: {np_res} | JAX clear: {jax_res}")
        if np_res != jax_res:
            print(f"❌ MISMATCH DETECTED ON EDGE CASE: {name}!")
            raise AssertionError(f"Mismatch on edge case: {name} ({p1} -> {p2})")
    print("✅ All explicit edge cases match perfectly!")

    # Run a batch of 500 random tests to verify uniformity across all angles
    print("\nRunning a batch of 500 random paths to verify uniformity...")
    mismatches = 0
    np.random.seed(42)
    
    from tqdm import tqdm
    for _ in tqdm(range(500), desc="Verifying raycasts"):
        # Pick random points in the grid
        sp = np.random.uniform(1.0, 99.0, 2)
        ep = np.random.uniform(1.0, 99.0, 2)
        
        res_np = _dda_raycast_np(sp, ep, occ_grid)
        res_jax = bool(dda_raycast(jnp.array(sp), jnp.array(ep), occ_grid_jax))
        
        if res_np != res_jax:
            mismatches += 1
            if mismatches <= 5:
                print(f"\nMismatch: start={sp}, end={ep} -> NumPy clear={res_np}, JAX clear={res_jax}")

    print(f"\nFound {mismatches} mismatches out of 500 random paths.")
    
    if mismatches > 0:
        raise AssertionError(f"JAX dda_raycast and NumPy _dda_raycast_np did not produce identical results on {mismatches} random paths!")
    else:
        print("\n✅ SUCCESS: Raycast results are 100% identical on all 500 random paths!")

if __name__ == "__main__":
    try:
        run_test()
    except AssertionError as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)
    sys.exit(0)
