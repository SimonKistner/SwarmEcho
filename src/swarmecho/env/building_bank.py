"""Immutable, padded map bank indexed by a persistent environment map ID."""
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np

from swarmecho.env.buildings import make_cuboid_building
from swarmecho.env.obstacles import building_roadmap


@jax.tree_util.register_pytree_node_class
@dataclass(eq=False)
class BuildingBank:
    envelope: object
    arrays: dict

    def tree_flatten(self):
        return (), self

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        return auxiliary

    def get(self, key, map_id):
        return self.arrays[key][map_id]


def balanced_map_ids(num_envs: int, num_maps: int) -> np.ndarray:
    """Assign persistent lane IDs with map counts differing by at most one."""
    if num_envs < 1 or num_maps < 1 or num_maps > num_envs:
        raise ValueError("A map bank needs between one and num_envs maps.")
    return np.arange(num_envs, dtype=np.int32) % num_maps


def prepare_bank(buildings, cfg, *, plan, corner_bonus_m=0., log=lambda message: None):
    from swarmecho.env.buildings import target_spawn_boxes
    first = buildings[0]
    maps = []
    for i, building in enumerate(buildings):
        if any(getattr(building, key) != getattr(first, key) for key in
               ("cell_size_m", "wall_thickness_m", "tile_thickness_m")):
            raise ValueError("A training map bank must use the same cell size and wall/tile thicknesses.")
        graph = building_roadmap(building, cfg, plan=plan, corner_bonus_m=corner_bonus_m)
        if plan and not len(graph.vertices):
            raise ValueError(f"Training map {i} has no planning nodes (roadmap_approach={cfg.roadmap_approach}); inspect its geometry and roadmap settings.")
        if plan and np.any(graph.distances[0] >= 1e6):
            raise ValueError(f"Training map {i} has a disconnected planning graph ({len(graph.vertices)} nodes); inspect its geometry before training.")
        base = target_spawn_boxes(building, cfg, include_excluded=True)
        record = dict(solid_min=graph.solid_min, solid_max=graph.solid_max,
                      base_position=building.base_position_m, world_size=building.world_size_m)
        if cfg.success_condition != "coverage":
            record.update(vertices=graph.vertices, distances=graph.distances,
                          corner_distances=graph.corner_distances)
        boxes_to_store = [("base", base)]
        if cfg.success_condition != "coverage":
            boxes_to_store.append(("target", target_spawn_boxes(building, cfg)))
        for prefix, boxes in boxes_to_store:
            for suffix, value in zip(("lower", "upper", "volumes"), boxes):
                record[f"{prefix}_{suffix}"] = np.asarray(value)
        maps.append(record)
        if i == 0 or (i + 1) % 100 == 0 or i + 1 == len(buildings):
            log(f"Compiled map bank {i + 1}/{len(buildings)}; {len(graph.vertices)} roadmap nodes")
    return prepare_bank_from_records(maps, first, cfg, interiors=[b.interior_cells for b in buildings], log=log)


def prepare_bank_from_records(maps, first, cfg, *, interiors=None, log=lambda message: None):
    """Pad already compiled per-map records for the JAX training batch."""
    if not maps:
        raise ValueError("A training map bank must contain at least one map.")
    maps = [dict(record, world_size=record.get("world_size", first.world_size_m)) for record in maps]
    dimensions = (first.interior_cells.shape if interiors is None else
                  tuple(max(interior.shape[axis] for interior in interiors) for axis in range(3)))
    envelope = make_cuboid_building(dimensions, cell_size_m=first.cell_size_m,
                                    tile_thickness_m=first.tile_thickness_m, wall_thickness_m=first.wall_thickness_m)
    from swarmecho.env.environment import coverage_grid_geometry

    voxel_size, shape = coverage_grid_geometry(envelope, cfg)
    indices = np.stack(np.meshgrid(*[np.arange(size) for size in shape], indexing="ij"), axis=-1)
    centres = (indices.astype(np.float32) + 0.5) * voxel_size
    cells = np.floor(centres / first.cell_size_m).astype(int)
    masks = np.empty((len(maps), *shape), dtype=np.bool_)
    for i, record in enumerate(maps):
        interior = first.interior_cells if interiors is None else interiors[i]
        valid = np.all(cells < np.asarray(interior.shape), axis=-1)
        local_cells = np.minimum(cells, np.asarray(interior.shape) - 1)
        eligible = valid & interior[local_cells[..., 0], local_cells[..., 1], local_cells[..., 2]]
        if len(record["solid_min"]):
            inside = np.any(np.all(
                (centres[..., None, :] >= record["solid_min"])
                & (centres[..., None, :] <= record["solid_max"]), axis=-1), axis=-1)
            eligible &= ~inside
        masks[i] = eligible
    arrays = {"coverage_eligible": masks}
    for key in maps[0]:
        shape = tuple(max(m[key].shape[d] for m in maps) for d in range(maps[0][key].ndim))
        fill = 1e6 if key in {"solid_min", "solid_max", "distances", "corner_distances", "vertices"} else 0.
        value = np.full((len(maps),) + shape, fill, dtype=np.float32)
        for i, record in enumerate(maps):
            value[(i,) + tuple(slice(0, n) for n in record[key].shape)] = record[key]
        arrays[key] = value
    total = sum(a.nbytes for a in arrays.values())
    log(f"Map bank immutable arrays: {total / 2**20:.1f} MiB; solids {arrays['solid_min'].shape}"
        + (f", roadmaps {arrays['distances'].shape}" if "distances" in arrays else ", targetless"))
    # Transfer once. These are closed-over device arrays, never rollout leaves.
    device_arrays = {key: jnp.asarray(value) for key, value in arrays.items()}
    return BuildingBank(envelope, device_arrays)
