import jax
import jax.numpy as jnp
import numpy as np

# Padding radius used across the project to handle LoS near map edges
PAD_RADIUS = 60.0

# ---------------------------------------------------------------------------
# 1. DDA Raycast (Pure JAX version for while_loop fallback)
# ---------------------------------------------------------------------------

def dda_raycast(p1: jax.Array, p2: jax.Array, occupancy_grid: jax.Array) -> jax.Array:
    """
    Check for line-of-sight between p1 and p2 on a 2D occupancy grid using DDA.
    Both p1 and p2 must be in grid coordinates (0 to G-1).
    Returns True if the path is CLEAR, False if a wall (True in grid) is hit.
    """
    G_x, G_y = occupancy_grid.shape
    x0, y0 = p1
    x1, y1 = p2
    dx, dy = x1 - x0, y1 - y0
    
    step_x, step_y = jnp.sign(dx), jnp.sign(dy)
    eps = 1e-8
    t_delta_x = jnp.abs(1.0 / (dx + eps))
    t_delta_y = jnp.abs(1.0 / (dy + eps))
    
    dist_to_boundary_x = jnp.where(step_x > 0, jnp.floor(x0 + 1) - x0, x0 - jnp.ceil(x0 - 1))
    dist_to_boundary_y = jnp.where(step_y > 0, jnp.floor(y0 + 1) - y0, y0 - jnp.ceil(y0 - 1))
    
    t_max_x = dist_to_boundary_x * t_delta_x
    t_max_y = dist_to_boundary_y * t_delta_y
    max_steps = G_x + G_y
    
    def cond_fn(val):
        ix, iy, t_max_x_val, t_max_y_val, steps, hit = val
        in_bounds = (ix >= 0) & (ix < G_x) & (iy >= 0) & (iy < G_y)
        not_reached = (steps < max_steps) & (jnp.maximum(t_max_x_val, t_max_y_val) < 1.0 + eps)
        return in_bounds & (~hit) & not_reached

    def body_fn(val):
        ix, iy, t_max_x_val, t_max_y_val, steps, hit = val
        # Step in the direction of the smaller t_max
        is_x_step = t_max_x_val < t_max_y_val
        new_ix = jnp.where(is_x_step, ix + step_x, ix)
        new_iy = jnp.where(is_x_step, iy, iy + step_y)
        new_t_max_x = jnp.where(is_x_step, t_max_x_val + t_delta_x, t_max_x_val)
        new_t_max_y = jnp.where(is_x_step, t_max_x_val, t_max_y_val + t_delta_y)
        
        n_ix, n_iy = new_ix.astype(jnp.int32), new_iy.astype(jnp.int32)
        oob = (n_ix < 0) | (n_ix >= G_x) | (n_iy < 0) | (n_iy >= G_y)
        new_hit = jnp.where(oob, False, occupancy_grid[n_ix, n_iy])
        return (new_ix, new_iy, new_t_max_x, new_t_max_y, steps + 1, new_hit)

    init_ix, init_iy = jnp.floor(x0).astype(jnp.int32), jnp.floor(y0).astype(jnp.int32)
    initial_hit = occupancy_grid[init_ix, init_iy]
    val_init = (init_ix.astype(jnp.float32), init_iy.astype(jnp.float32), 
                t_max_x, t_max_y, jnp.array(0, dtype=jnp.int32), initial_hit)
    
    final_val = jax.lax.while_loop(cond_fn, body_fn, val_init)
    return ~final_val[5]

def dda_dist(p1: jax.Array, p2: jax.Array, occupancy_grid: jax.Array) -> jax.Array:
    """
    Find the fractional distance (0 to 1) along the ray from p1 to p2 until 
    an occupancy grid cell is hit.
    Both p1 and p2 must be in grid coordinates.
    Returns:
        t_hit: jax.Array (float32, 0.0 to 1.0). 1.0 if no hit.
    """
    G_x, G_y = occupancy_grid.shape
    x0, y0 = p1
    x1, y1 = p2
    dx, dy = x1 - x0, y1 - y0
    
    step_x, step_y = jnp.sign(dx), jnp.sign(dy)
    eps = 1e-8
    t_delta_x = jnp.abs(1.0 / (dx + eps))
    t_delta_y = jnp.abs(1.0 / (dy + eps))
    
    dist_to_boundary_x = jnp.where(step_x > 0, jnp.floor(x0 + 1) - x0, x0 - jnp.ceil(x0 - 1))
    dist_to_boundary_y = jnp.where(step_y > 0, jnp.floor(y0 + 1) - y0, y0 - jnp.ceil(y0 - 1))
    
    t_max_x = dist_to_boundary_x * t_delta_x
    t_max_y = dist_to_boundary_y * t_delta_y
    max_steps = G_x + G_y
    
    def cond_fn(val):
        ix, iy, t_max_x_val, t_max_y_val, steps, hit, captured_t = val
        in_bounds = (ix >= 0) & (ix < G_x) & (iy >= 0) & (iy < G_y)
        not_reached = (steps < max_steps) & (jnp.minimum(t_max_x_val, t_max_y_val) < 1.0 - eps)
        return in_bounds & (~hit) & not_reached

    def body_fn(val):
        ix, iy, t_max_x_val, t_max_y_val, steps, hit, captured_t = val
        is_x_step = t_max_x_val < t_max_y_val
        
        # Calculate the t-value for THIS step before moving
        current_t = jnp.minimum(t_max_x_val, t_max_y_val)
        
        new_ix = jnp.where(is_x_step, ix + step_x, ix)
        new_iy = jnp.where(is_x_step, iy, iy + step_y)
        new_t_max_x = jnp.where(is_x_step, t_max_x_val + t_delta_x, t_max_x_val)
        new_t_max_y = jnp.where(is_x_step, t_max_y_val, t_max_y_val + t_delta_y)
        
        n_ix, n_iy = new_ix.astype(jnp.int32), new_iy.astype(jnp.int32)
        oob = (n_ix < 0) | (n_ix >= G_x) | (n_iy < 0) | (n_iy >= G_y)
        new_hit = jnp.where(oob, False, occupancy_grid[n_ix, n_iy])
        
        # If we hit, we capture the 't' value from just before the step into the wall
        captured_t = jnp.where(new_hit, current_t, 1.0)
        
        return (new_ix, new_iy, new_t_max_x, new_t_max_y, steps + 1, new_hit, captured_t)

    init_ix, init_iy = jnp.floor(x0).astype(jnp.int32), jnp.floor(y0).astype(jnp.int32)
    # Initial hit check
    initial_hit = occupancy_grid[init_ix, init_iy]
    val_init = (init_ix.astype(jnp.float32), init_iy.astype(jnp.float32), 
                t_max_x, t_max_y, jnp.array(0, dtype=jnp.int32), initial_hit, 
                jnp.where(initial_hit, 0.0, 1.0))
    
    final_val = jax.lax.while_loop(cond_fn, body_fn, val_init)
    return jnp.clip(final_val[6], 0.0, 1.0)


# ---------------------------------------------------------------------------
# 2. Local Ray Stencil (Bresenham-style precomputed paths)
# ---------------------------------------------------------------------------

def get_ray_stencil(radius: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Precomputes the paths for all rays in a (2*radius + 1)^2 box.
    Returns:
        ray_coords : (NumRays, MaxLen, 2) integer offsets from (0,0)
        ray_lengths: (NumRays,) total length of each ray
    """
    R = radius
    size = 2 * R + 1
    # Target points: all cells in the local box
    dx, dy = np.meshgrid(np.arange(-R, R+1), np.arange(-R, R+1), indexing='ij')
    targets = np.stack([dx.ravel(), dy.ravel()], axis=-1)
    
    NumRays = targets.shape[0]
    # Bounded by maximum sampling density to prevent any broadcast errors
    MaxLen = int(np.ceil(R * 2.0)) + 5
    
    stencil_coords = np.zeros((NumRays, MaxLen, 2), dtype=np.int32)
    stencil_lengths = np.zeros(NumRays, dtype=np.int32)
    
    # Bresenham-like line tracing for each target
    for i in range(NumRays):
        t_x, t_y = targets[i]
        
        # Ray from (0.5, 0.5) to (t_x + 0.5, t_y + 0.5)
        # Using a simple stepping approach for pre-computing
        norm = np.sqrt(t_x**2 + t_y**2)
        if norm == 0:
            stencil_coords[i, 0] = [0, 0]
            stencil_lengths[i] = 1
            continue
            
        # Number of samples depends on distance to ensure no holes
        n_samples = int(np.ceil(np.max(np.abs([t_x, t_y])) * 1.5)) + 1
        n_samples = max(n_samples, 2)
        
        # Trace path
        path = []
        for s in range(n_samples):
            f = s / (n_samples - 1)
            px, py = int(np.floor(f * t_x)), int(np.floor(f * t_y))
            if not path or (px != path[-1][0] or py != path[-1][1]):
                path.append((px, py))
        
        # Pad and Store
        L = len(path)
        stencil_coords[i, :L] = np.array(path)
        # Pad with final coordinate (so cummax "holds" the wall value if hit)
        stencil_coords[i, L:] = path[-1]
        stencil_lengths[i] = L
        
    return stencil_coords, stencil_lengths


# ---------------------------------------------------------------------------
# 3. High-Speed Stencil Visibility
# ---------------------------------------------------------------------------

def compute_local_visibility(
    local_patch: jax.Array,   # (2R+1, 2R+1) bool
    stencil:     jax.Array,   # (NumRays, MaxLen, 2) int32
) -> jax.Array:
    """
    Use the Stencil + Cummax trick to compute visibility for a local patch.
    local_patch: occupancy grid around the drone. (0,0) is center (R, R).
    """
    R = (local_patch.shape[0] - 1) // 2
    # Convert stencil offsets to patch indices
    # Stencil is offset from drone center (R, R)
    patch_idx = stencil + R   # (NumRays, MaxLen, 2)
    
    # Gather wall values along all rays
    # walls shape: (NumRays, MaxLen)
    walls = local_patch[patch_idx[..., 0], patch_idx[..., 1]]
    
    # Propagate shadow: everything AFTER a 1 becomes a 1
    shadows = jnp.maximum.accumulate(walls, axis=-1)
    
    # The result for each target is the shadow value at the END of its ray
    # Wait, we need the visibility mask for the WHOLE patch.
    # Note: Stencil index (i, L_i-1) is the target cell for Ray i.
    # We can use jax.lax.gather or just index with scatter/select.
    
    # Each row in the stencil represents a target cell.
    # We can just take the final value of the shadow for that ray.
    # However, if a ray is padded with its target coord, cummax at the end 
    # tells us if the target cell itself is obscured by something BEFORE it.
    
    # Correct Visibility Visibility:
    # A cell is visible if the cumulative max is 0 AT THE POINT WE REACH THE CELL.
    # To handle the cell itself being a wall (but still "visible" as a wall), 
    # we can use the penultimate value or just shift.
    
    # Simpler: A target is visible if NO wall was hit BEFORE it.
    # Shift shadows right by 1
    visible_paths = jnp.logical_not(jnp.concatenate([jnp.zeros((shadows.shape[0], 1), dtype=bool), shadows[:, :-1]], axis=-1))
    
    # For each ray, we only care about the result for the specific target it represents.
    # Stencil row i corresponds to targets.ravel()[i]
    num_rays = shadows.shape[0]
    # We need to map back to 2D
    size = 2 * R + 1
    vis_1d = jnp.zeros(num_rays, dtype=bool)
    
    # This is slightly tricky: multiple entries in shadows[...] correspond to different 
    # depths along a single long ray. But we want a MASK for the whole grid.
    
    # NEW APPROACH:
    # 1. Gather walls: walls_along_rays (NumRays, MaxLen)
    # 2. Shadows: cummax(...)
    # 3. Mask: !shadows
    # 4. Scatter back to (size, size)
    # Note: Many rays share segments. Scatter-Min (or Max) handles overlaps correctly.
    
    vis_local = jnp.ones((size, size), dtype=bool)
    # flattened target pixels (NumRays, 1) to (NumRays, MaxLen)
    flat_patch_idx = patch_idx[..., 0] * size + patch_idx[..., 1]
    
    # We scatter 'visible_paths' into a flat (size*size) array
    # If multiple rays hit the same cell (common near center), use the most optimistic (Visible)
    # 0 = Hidden, 1 = Visible. We want max (any ray that sees it makes it visible)
    vis_flat = jnp.zeros(size * size, dtype=jnp.int32)
    vis_flat = vis_flat.at[flat_patch_idx.ravel()].max(visible_paths.ravel().astype(jnp.int32))
    
    return vis_flat.reshape(size, size).astype(bool)


