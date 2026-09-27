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


def prepare_bank(buildings, cfg, *, plan, corner_bonus_m=0., log=lambda message: None):
    from swarmecho.env.buildings import target_spawn_boxes
    first = buildings[0]
    maps = []
    for i, building in enumerate(buildings):
        if (building.interior_cells.shape != first.interior_cells.shape
                or not np.array_equal(building.world_size_m, first.world_size_m)):
            raise ValueError("A training map bank must use the same grid dimensions and cell size.")
        graph = building_roadmap(building, cfg, plan=plan, corner_bonus_m=corner_bonus_m)
        if plan and (not len(graph.vertices) or np.any(graph.distances[0] >= 1e6)):
            raise ValueError(f"Generated map {i} has a disconnected planning graph; inspect its seed before training.")
        base = target_spawn_boxes(building, cfg, include_excluded=True)
        record = dict(solid_min=graph.solid_min, solid_max=graph.solid_max,
                      base_position=building.base_position_m)
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
    envelope = make_cuboid_building(first.interior_cells.shape, cell_size_m=first.cell_size_m,
                                    tile_thickness_m=first.tile_thickness_m, wall_thickness_m=first.wall_thickness_m)
    from swarmecho.env.environment import coverage_grid_geometry

    voxel_size, shape = coverage_grid_geometry(first, cfg)
    indices = np.stack(np.meshgrid(*[np.arange(size) for size in shape], indexing="ij"), axis=-1)
    centres = (indices.astype(np.float32) + 0.5) * voxel_size
    cells = np.minimum(np.floor(centres / first.cell_size_m).astype(int),
                       np.asarray(first.interior_cells.shape) - 1)
    masks = np.empty((len(maps), *shape), dtype=np.bool_)
    for i, record in enumerate(maps):
        interior = first.interior_cells if interiors is None else interiors[i]
        eligible = interior[cells[..., 0], cells[..., 1], cells[..., 2]]
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
