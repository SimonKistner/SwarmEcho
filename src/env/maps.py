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

try:
    from scipy.ndimage import binary_dilation
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


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

    # Coarse logical maze grid used by discrete route features. This grid is
    # intentionally separate from 1 m physics/coverage grids: finder-path
    # tracking, discrete chain rewards, path rendering, and preview grid lines
    # should all use these map-construction cells.
    maze_cell_cols: int | None = None
    maze_cell_rows: int | None = None

    # Optional per-zone wall-clearance distances (metres). Candidates within
    # this distance of any wall are excluded from the valid coord pool.
    target_wall_clearance: float = 0.0
    base_wall_clearance:   float = 0.0
    drone_wall_clearance:  float = 0.0

    # Optional list of axis-aligned bounding boxes to *exclude* from the
    # target spawn pool (e.g., to keep target out of the base room).
    # Each entry: [x_min, y_min, x_max, y_max] in world metres.
    target_exclude_zones: list[list[float]] = dataclasses.field(default_factory=list)

    # Physical walls (Occupancy grid)
    # shape (H_cells, W_cells), True where wall exists
    occupancy_grid: np.ndarray | None = None

    # Communication occupancy grid: normal walls block communication, mesh walls do not.
    # Movement, coverage, and visual line-of-sight still use occupancy_grid.
    communication_occupancy_grid: np.ndarray | None = None

    # Padded occupancy grid (for safe local slicing in JAX)
    # shape (W+2R, H+2R), True where wall exists
    padded_occupancy_grid: jax.Array | None = None

    # High-level Blueprint Primitives (for intuitive editing)
    # Rooms: {x, y, w, h, door_side: 'N'|'S'|'E'|'W'|'none'}
    # Hallways: {x1, y1, x2, y2, width}
    rooms:    list[dict] = dataclasses.field(default_factory=list)
    hallways: list[dict] = dataclasses.field(default_factory=list)
    walls:    list[list[float]] = dataclasses.field(default_factory=list) # Raw segments
    # Mesh walls block movement/coverage/visual LoS like walls, but communication can pass through them.
    mesh_walls: list[list[float]] = dataclasses.field(default_factory=list)
    drone_spawn_points:       jax.Array | None = None
    # MEM_T8-only diagnostic scaffolding: fixed per-agent target/anti-target slots.
    # Normal SwarmEcho task maps use one target sampled from target_spawn_zone.
    target_spawn_points:      jax.Array | None = None
    anti_target_spawn_points: jax.Array | None = None
    valid_base_coords:   jax.Array | None = None
    valid_target_coords: jax.Array | None = None
    valid_drone_coords:  jax.Array | None = None

    @classmethod
    def load(cls, path: str | Path, cell_size: float | None = None, padding_radius: float = 60.0, extra_walls: list[list[float]] | None = None) -> MapDefinition:
        path = Path(path)
        with open(path.with_suffix(".yaml"), "r") as f:
            data = yaml.safe_load(f)

        m = cls(
            name              = data["name"],
            width             = float(data["width"]),
            height            = float(data["height"]),
            maze_cell_cols    = int(data["maze_cell_grid"]["cols"]) if data.get("maze_cell_grid") else None,
            maze_cell_rows    = int(data["maze_cell_grid"]["rows"]) if data.get("maze_cell_grid") else None,
            base_spawn_zone   = data["spawn_zones"]["base"],
            target_spawn_zone = data["spawn_zones"]["target"],
            drone_spawn_zone  = data["spawn_zones"]["drone"],
            rooms             = data.get("rooms", []),
            hallways          = data.get("hallways", []),
            walls             = list(data.get("walls", [])) + list(extra_walls or []),
            mesh_walls        = data.get("mesh_walls", data.get("mesh", [])),
            target_wall_clearance = float(data.get("target_wall_clearance", 0.0)),
            base_wall_clearance   = float(data.get("base_wall_clearance",   0.0)),
            drone_wall_clearance  = float(data.get("drone_wall_clearance",  0.0)),
            target_exclude_zones  = data.get("target_exclude_zones", []),
        )
        if data.get("spawn_points", {}).get("drone") is not None:
            m.drone_spawn_points = jnp.array(data["spawn_points"]["drone"], dtype=jnp.float32)
        # MEM_T8-only: when env.num_targets > 1, physics interprets these as
        # fixed per-agent target slots for the memory diagnostic.
        if data.get("spawn_points", {}).get("target") is not None:
            m.target_spawn_points = jnp.array(data["spawn_points"]["target"], dtype=jnp.float32)
        # MEM_T8-only: paired wrong-branch decoys for the memory diagnostic.
        if data.get("spawn_points", {}).get("anti_target") is not None:
            m.anti_target_spawn_points = jnp.array(data["spawn_points"]["anti_target"], dtype=jnp.float32)

        # Grid dimensions at 1m resolution (or custom cell_size)
        res = cell_size if cell_size is not None else 1.0
        m.rasterize(res)

        # Pre-calculate safe spawn indices in meters
        # Build an eroded occupancy grid for each clearance distance:
        # a cell is "too close to a wall" if any wall cell lies within
        # `clearance` metres (= cells at 1 m/cell resolution).
        occ = m.occupancy_grid  # shape (W_px, H_px), True=wall
        sx_scale = occ.shape[0] / m.width
        sy_scale = occ.shape[1] / m.height

        def _build_clearance_mask(clearance_m: float) -> np.ndarray:
            """Return bool mask (W_px, H_px): True where safe (not near wall)."""
            if clearance_m <= 0.0:
                return ~occ
            if not _HAVE_SCIPY:
                raise ImportError(
                    f"scipy is required to apply wall_clearance={clearance_m}m. "
                    "Install it with: pip install scipy"
                )
            # Number of cells to erode (at 1 m resolution)
            r_cells = max(1, int(np.ceil(clearance_m * ((sx_scale + sy_scale) / 2))))
            # Dilate the wall map by r_cells; then safe = NOT dilated
            dilated = binary_dilation(occ, iterations=r_cells)
            return ~dilated

        safe_target = _build_clearance_mask(m.target_wall_clearance)
        safe_base   = _build_clearance_mask(m.base_wall_clearance)
        safe_drone  = _build_clearance_mask(m.drone_wall_clearance)

        def _get_valid_coords(zone, safe_mask, exclude_zones=None):
            x_min, y_min, x_max, y_max = zone
            # Convert zone to pixel ranges
            ix_lo = int(np.floor(x_min * sx_scale))
            ix_hi = int(np.ceil(x_max  * sx_scale))
            iy_lo = int(np.floor(y_min * sy_scale))
            iy_hi = int(np.ceil(y_max  * sy_scale))
            # Clamp to grid bounds
            ix_lo = max(0, ix_lo); ix_hi = min(safe_mask.shape[0], ix_hi)
            iy_lo = max(0, iy_lo); iy_hi = min(safe_mask.shape[1], iy_hi)

            # Slice and find safe cells
            local_safe = safe_mask[ix_lo:ix_hi, iy_lo:iy_hi]
            w_idx, h_idx = np.where(local_safe)
            # Back to world metres (cell centre = cell_index + 0.5 at 1 m/cell)
            coords_m = np.stack([
                (w_idx + ix_lo + 0.5) / sx_scale,
                (h_idx + iy_lo + 0.5) / sy_scale,
            ], axis=-1).astype(np.float32)

            # Apply optional exclusion zones (e.g., base spawn room)
            if exclude_zones:
                keep = np.ones(len(coords_m), dtype=bool)
                for ez in exclude_zones:
                    ex0, ey0, ex1, ey1 = ez
                    in_zone = (
                        (coords_m[:, 0] >= ex0) & (coords_m[:, 0] <= ex1) &
                        (coords_m[:, 1] >= ey0) & (coords_m[:, 1] <= ey1)
                    )
                    keep &= ~in_zone
                coords_m = coords_m[keep]

            if len(coords_m) == 0:
                # Fallback to zone centre
                return jnp.array([[(x_min + x_max) / 2, (y_min + y_max) / 2]], dtype=jnp.float32)
            return jnp.array(coords_m, dtype=jnp.float32)

        m.valid_base_coords   = _get_valid_coords(m.base_spawn_zone,   safe_base)
        m.valid_target_coords = _get_valid_coords(m.target_spawn_zone, safe_target,
                                                   exclude_zones=m.target_exclude_zones)
        m.valid_drone_coords  = _get_valid_coords(m.drone_spawn_zone,  safe_drone)

        # 1. Create JAX Occupational Grid
        occ_jax = jnp.array(m.occupancy_grid, dtype=jnp.bool_)

        # 2. Create Padded Grid (True/Walls outside map)
        R_cells = int(padding_radius / res) + 2
        padded = jnp.ones((occ_jax.shape[0] + 2*R_cells, occ_jax.shape[1] + 2*R_cells), dtype=jnp.bool_)
        m.padded_occupancy_grid = padded.at[R_cells:-R_cells, R_cells:-R_cells].set(occ_jax)

        return m

    def rasterize(self, resolution: float = 1.0):
        """
        Convert logical wall edges into high-res occupancy grids.

        Normal walls block movement, coverage/visual line-of-sight, and
        communication. Mesh walls block movement and coverage/visual line-of-sight
        but are omitted from ``communication_occupancy_grid`` so drones can share
        memory through them.
        """
        W_px = int(self.width / resolution)
        H_px = int(self.height / resolution)

        # Scale factors: world meters to pixels
        sx = W_px / self.width
        sy = H_px / self.height

        # 1. Compile primitives into effective normal wall segments.
        normal_segments = list(self.walls)

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
                        normal_segments.append([p1[0], p1[1], x + w/2 - dw/2, p1[1]])
                        normal_segments.append([x + w/2 + dw/2, p1[1], p2[0], p2[1]])
                    else:
                        normal_segments.append([p1[0], p1[1], p1[0], y + h/2 - dw/2])
                        normal_segments.append([p1[0], y + h/2 + dw/2, p2[0], p2[1]])
                else:
                    normal_segments.append([p1[0], p1[1], p2[0], p2[1]])

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
                normal_segments.append([x1 + ux, y1 + uy, x2 + ux, y2 + uy])
                normal_segments.append([x1 - ux, y1 - uy, x2 - ux, y2 - uy])

        def _draw_segments(segments: list[list[float]]) -> np.ndarray:
            from PIL import Image, ImageDraw

            img = Image.new("L", (W_px, H_px), 255) # white
            draw = ImageDraw.Draw(img)
            for seg in segments:
                if len(seg) == 4:
                    x1, y1, x2, y2 = seg
                    # Clamp coordinates to [0, W_px - 1] and [0, H_px - 1] to prevent clipping at boundaries
                    cx1 = min(max(x1 * sx, 0.0), W_px - 1)
                    cy1 = min(max((self.height - y1) * sy, 0.0), H_px - 1)
                    cx2 = min(max(x2 * sx, 0.0), W_px - 1)
                    cy2 = min(max((self.height - y2) * sy, 0.0), H_px - 1)
                    draw.line([cx1, cy1, cx2, cy2], fill=0, width=1)

            # In current SwarmEcho world (physics.py), occupancy_grid[ix, iy]
            # is accessed where ix is world-x, iy is world-y index.
            # PIL image rows are Y-pixels. We flip Y so index 0 is bottom (y=0 in meters).
            # We transpose so first axis is X.
            grid_flipped = np.array(img.transpose(Image.FLIP_TOP_BOTTOM)) # (H, W), y=0 at index 0
            return (grid_flipped < 128).T # (W, H)

        physical_segments = normal_segments + list(self.mesh_walls)
        if physical_segments:
            self.occupancy_grid = _draw_segments(physical_segments)
        else:
            self.occupancy_grid = np.zeros((W_px, H_px), dtype=bool)

        if normal_segments:
            self.communication_occupancy_grid = _draw_segments(normal_segments)
        else:
            self.communication_occupancy_grid = np.zeros((W_px, H_px), dtype=bool)

    def sample_base(self, key: jax.Array) -> jax.Array:
        if self.valid_base_coords is None:
            return jnp.array([(self.base_spawn_zone[0] + self.base_spawn_zone[2])/2,
                              (self.base_spawn_zone[1] + self.base_spawn_zone[3])/2], dtype=jnp.float32)
        idx = jax.random.randint(key, shape=(), minval=0, maxval=len(self.valid_base_coords))
        return self.valid_base_coords[idx]

    def sample_target(self, key: jax.Array) -> jax.Array:
        if self.target_spawn_points is not None:
            return self.target_spawn_points[0]
        if self.valid_target_coords is None:
            return jnp.array([(self.target_spawn_zone[0] + self.target_spawn_zone[2])/2,
                              (self.target_spawn_zone[1] + self.target_spawn_zone[3])/2], dtype=jnp.float32)
        idx = jax.random.randint(key, shape=(), minval=0, maxval=len(self.valid_target_coords))
        return self.valid_target_coords[idx]

    def sample_drones(self, key: jax.Array, N: int) -> jax.Array:
        if self.drone_spawn_points is not None and len(self.drone_spawn_points) >= N:
            return self.drone_spawn_points[:N]
        if self.valid_drone_coords is None:
            return jnp.zeros((N, 2), dtype=jnp.float32)
        keys = jax.random.split(key, N)
        def _sample_one(k):
            idx = jax.random.randint(k, shape=(), minval=0, maxval=len(self.valid_drone_coords))
            return self.valid_drone_coords[idx]
        return jax.vmap(_sample_one)(keys)
