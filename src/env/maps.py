from __future__ import annotations

import dataclasses
import datetime
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from PIL import Image
from env.raycast import PAD_RADIUS


@dataclasses.dataclass
class MapDefinition:
    """
    Static map definition loaded from YAML/PNG.
    """
    name:          str
    width:         float
    height:        float
    
    # Spawn Zones (Absolute meter coords)
    # [x_min, y_min, x_max, y_max]
    base_spawn_zone:   list[float]
    target_spawn_zone: list[float]
    drone_spawn_zone:  list[float]
    
    # Physical walls (Occupancy grid)
    # shape (H_cells, W_cells), True where wall exists
    occupancy_grid: np.ndarray | None = None
    
    # Padded occupancy grid (for safe local slicing in JAX)
    # shape (W+2R, H+2R), True where wall exists
    padded_occupancy_grid: jax.Array | None = None

    # High-level Blueprint Primitives (for intuitive editing)
    # Rooms: {x, y, w, h, door_side: 'N'|'S'|'E'|'W'|'none'}
    # Hallways: {x1, y1, x2, y2, width}
    rooms:    list[dict] = dataclasses.field(default_factory=list)
    hallways: list[dict] = dataclasses.field(default_factory=list)
    walls:    list[list[float]] = dataclasses.field(default_factory=list) # Raw segments
    valid_base_coords:   jax.Array | None = None
    valid_target_coords: jax.Array | None = None
    valid_drone_coords:  jax.Array | None = None
    
    @classmethod
    def load(cls, path: str | Path, cell_size: float | None = None, padding_radius: float = 60.0) -> MapDefinition:
        path = Path(path)
        with open(path.with_suffix(".yaml"), "r") as f:
            data = yaml.safe_load(f)
            
        m = cls(
            name              = data["name"],
            width             = float(data["width"]),
            height            = float(data["height"]),
            base_spawn_zone   = data["spawn_zones"]["base"],
            target_spawn_zone = data["spawn_zones"]["target"],
            drone_spawn_zone  = data["spawn_zones"]["drone"],
            rooms             = data.get("rooms", []),
            hallways          = data.get("hallways", []),
            walls             = data.get("walls", []),
        )
        
        # Grid dimensions at 1m resolution (or custom cell_size)
        res = cell_size if cell_size is not None else 1.0
        m.rasterize(res)
        
        # Pre-calculate safe spawn indices in meters
        w_idx, h_idx = np.where(~m.occupancy_grid)
        valid_indices = np.stack([w_idx, h_idx], axis=-1)
        sx, sy = m.occupancy_grid.shape[0] / m.width, m.occupancy_grid.shape[1] / m.height
        coords_m = valid_indices.astype(np.float32) / np.array([sx, sy])
        
        def _get_valid_coords(zone):
            x_min, y_min, x_max, y_max = zone
            mask = (coords_m[:, 0] >= x_min) & (coords_m[:, 0] <= x_max) & \
                   (coords_m[:, 1] >= y_min) & (coords_m[:, 1] <= y_max)
            valid = coords_m[mask]
            if len(valid) == 0:
                # Fallback to center if no valid cells found
                return jnp.array([[(x_min + x_max)/2, (y_min + y_max)/2]], dtype=jnp.float32)
            return jnp.array(valid, dtype=jnp.float32)
            
        m.valid_base_coords = _get_valid_coords(m.base_spawn_zone)
        m.valid_target_coords = _get_valid_coords(m.target_spawn_zone)
        m.valid_drone_coords = _get_valid_coords(m.drone_spawn_zone)

        # 1. Create JAX Occupational Grid
        occ_jax = jnp.array(m.occupancy_grid, dtype=jnp.bool_)
        
        # 2. Create Padded Grid (True/Walls outside map)
        R_cells = int(padding_radius / res) + 2
        padded = jnp.ones((occ_jax.shape[0] + 2*R_cells, occ_jax.shape[1] + 2*R_cells), dtype=jnp.bool_)
        m.padded_occupancy_grid = padded.at[R_cells:-R_cells, R_cells:-R_cells].set(occ_jax)
            
        return m

    def rasterize(self, resolution: float = 1.0):
        """
        Convert logical wall edges into a high-res occupancy grid (PNG equivalent).
        resolution: meters per pixel in the resulting occupancy grid.
        """
        # Create a grid matching the width/height at the given resolution
        W_px = int(self.width / resolution)
        H_px = int(self.height / resolution)
        grid = np.zeros((W_px, H_px), dtype=bool)
        
        if not self.walls:
            self.occupancy_grid = grid
            return

        # Scale factors: world meters to pixels
        sx = W_px / self.width
        sy = H_px / self.height
        
        from PIL import Image, ImageDraw
        img = Image.new("L", (W_px, H_px), 255) # white
        draw = ImageDraw.Draw(img)
        
        # 1. Compile Primitives into effective walls
        all_segments = list(self.walls) # Start with custom walls
        
        # Add Room boundaries
        dw = 10.0 # Door width in meters
        for r in self.rooms:
            x, y, w, h = r['x'], r['y'], r['w'], r['h']
            ds = r.get('door_side', 'none')
            
            # Helper to draw wall with optional gap
            def draw_wall(p1, p2, side):
                if ds == side:
                    # Draw as two segments with a gap in the middle
                    if side in ['N', 'S']:
                        all_segments.append([p1[0], p1[1], x + w/2 - dw/2, p1[1]])
                        all_segments.append([x + w/2 + dw/2, p1[1], p2[0], p2[1]])
                    else:
                        all_segments.append([p1[0], p1[1], p1[0], y + h/2 - dw/2])
                        all_segments.append([p1[0], y + h/2 + dw/2, p2[0], p2[1]])
                else:
                    all_segments.append([p1[0], p1[1], p2[0], p2[1]])

            draw_wall([x, y + h], [x + w, y + h], 'N') # Top
            draw_wall([x, y], [x + w, y], 'S')         # Bottom
            draw_wall([x, y], [x, y + h], 'W')         # Left
            draw_wall([x + w, y], [x + w, y + h], 'E') # Right
            
        # Add Hallway boundaries
        for h in self.hallways:
            x1, y1, x2, y2, width = h['x1'], h['y1'], h['x2'], h['y2'], h['width']
            # Simplistic perpendicular offset for hallway walls
            dx, dy = x2 - x1, y2 - y1
            dist = (dx**2 + dy**2)**0.5
            if dist > 0:
                ux, uy = -dy/dist * (width/2), dx/dist * (width/2)
                all_segments.append([x1 + ux, y1 + uy, x2 + ux, y2 + uy])
                all_segments.append([x1 - ux, y1 - uy, x2 - ux, y2 - uy])

        # 2. Draw all compiled segments
        for seg in all_segments:
            if len(seg) == 4:
                x1, y1, x2, y2 = seg
                # Snap to pixels/meters (width=1 ensures 1m thick walls)
                draw.line([x1*sx, (self.height - y1)*sy, x2*sx, (self.height - y2)*sy], fill=0, width=1)
        
        # In current SwarmEcho world (physics.py), occupancy_grid[ix, iy] 
        # is accessed where ix is world-x, iy is world-y index.
        # PIL image rows are Y-pixels. We flip Y so index 0 is bottom (y=0 in meters).
        # We transpose so first axis is X.
        grid_flipped = np.array(img.transpose(Image.FLIP_TOP_BOTTOM)) # (H, W), y=0 at index 0
        self.occupancy_grid = (grid_flipped < 128).T # (W, H)

    def sample_base(self, key: jax.Array) -> jax.Array:
        if self.valid_base_coords is None:
            return jnp.array([(self.base_spawn_zone[0] + self.base_spawn_zone[2])/2, 
                              (self.base_spawn_zone[1] + self.base_spawn_zone[3])/2], dtype=jnp.float32)
        idx = jax.random.randint(key, shape=(), minval=0, maxval=len(self.valid_base_coords))
        return self.valid_base_coords[idx]

    def sample_target(self, key: jax.Array) -> jax.Array:
        if self.valid_target_coords is None:
            return jnp.array([(self.target_spawn_zone[0] + self.target_spawn_zone[2])/2, 
                              (self.target_spawn_zone[1] + self.target_spawn_zone[3])/2], dtype=jnp.float32)
        idx = jax.random.randint(key, shape=(), minval=0, maxval=len(self.valid_target_coords))
        return self.valid_target_coords[idx]

    def sample_drones(self, key: jax.Array, N: int) -> jax.Array:
        if self.valid_drone_coords is None:
            return jnp.zeros((N, 2), dtype=jnp.float32)
        keys = jax.random.split(key, N)
        def _sample_one(k):
            idx = jax.random.randint(k, shape=(), minval=0, maxval=len(self.valid_drone_coords))
            return self.valid_drone_coords[idx]
        return jax.vmap(_sample_one)(keys)


