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
    envelope = make_cuboid_building(first.interior_cells.shape, cell_size_m=first.cell_size_m,
                                  tile_thickness_m=first.tile_thickness_m, wall_thickness_m=first.wall_thickness_m)
    maps = []
    for i, building in enumerate(buildings):
        if (building.interior_cells.shape != first.interior_cells.shape
                or not np.array_equal(building.world_size_m, first.world_size_m)):
            raise ValueError("A training map bank must use the same grid dimensions and cell size.")
        graph = building_roadmap(building, cfg, plan=plan, corner_bonus_m=corner_bonus_m)
        if plan and (not len(graph.vertices) or np.any(graph.distances[0] >= 1e6)):
            raise ValueError(f"Generated map {i} has a disconnected planning graph; inspect its seed before training.")
        target = target_spawn_boxes(building, cfg)
        base = target_spawn_boxes(building, cfg, include_excluded=True)
        record = dict(solid_min=graph.solid_min, solid_max=graph.solid_max, vertices=graph.vertices,
                      distances=graph.distances, corner_distances=graph.corner_distances,
                      base_position=building.base_position_m)
        for prefix, boxes in (("target", target), ("base", base)):
            for suffix, value in zip(("lower", "upper", "volumes"), boxes):
                record[f"{prefix}_{suffix}"] = np.asarray(value)
        maps.append(record)
        if i == 0 or (i + 1) % 100 == 0 or i + 1 == len(buildings):
            log(f"Compiled map bank {i + 1}/{len(buildings)}; {len(graph.vertices)} roadmap nodes")
    arrays = {}
    for key in maps[0]:
        shape = tuple(max(m[key].shape[d] for m in maps) for d in range(maps[0][key].ndim))
        fill = 1e6 if key in {"solid_min", "solid_max", "distances", "corner_distances", "vertices"} else 0.
        value = np.full((len(maps),) + shape, fill, dtype=np.float32)
        for i, record in enumerate(maps):
            value[(i,) + tuple(slice(0, n) for n in record[key].shape)] = record[key]
        arrays[key] = value
    total = sum(a.nbytes for a in arrays.values())
    log(f"Map bank immutable arrays: {total / 2**20:.1f} MiB; solids {arrays['solid_min'].shape}, roadmaps {arrays['distances'].shape}")
    # Transfer once. These are closed-over device arrays, never rollout leaves.
    return BuildingBank(envelope, {key: jnp.asarray(value) for key, value in arrays.items()})
