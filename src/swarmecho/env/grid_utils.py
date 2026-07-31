"""Small helpers for interpreting environment occupancy grids."""

import jax
import jax.numpy as jnp


def has_inner_obstacles(grid: jax.Array) -> bool:
    """Return whether ``grid`` contains obstacles away from the outer border.

    Open square maps commonly include the world boundary as wall cells.  Those
    boundary cells should not force line-of-sight raycasts for communication
    between in-bounds agents, because any valid in-world communication segment
    cannot be blocked by an outer border unless an endpoint has already left the
    playable area.  This helper therefore ignores the one-cell perimeter and
    reports only true interior blockers.

    Very small grids cannot have a meaningful one-cell interior, so they fall
    back to checking the whole grid.
    """
    if grid.shape[0] <= 2 or grid.shape[1] <= 2:
        return bool(jnp.any(grid))
    return bool(jnp.any(grid[1:-1, 1:-1]))

