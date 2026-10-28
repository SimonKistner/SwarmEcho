"""Immutable, padded map bank indexed by a persistent environment map ID."""
from dataclasses import dataclass, fields
import hashlib
import jax
import jax.numpy as jnp
import numpy as np

from swarmecho.env.buildings import make_cuboid_building
from swarmecho.env.obstacles import building_roadmap


_RELAY_BANK_CACHE = {}


def _relay_bank_key(records, cfg, extras=()):
    digest = hashlib.sha256()
    settings = (cfg.drone_radius, cfg.obstacle_planning_clearance_m, cfg.roadmap_approach,
                cfg.roadmap_node_density, cfg.roadmap_merge_wall_end_nodes,
                cfg.target_wall_buffer_fraction, cfg.coverage_voxel_size, cfg.success_condition, extras)
    digest.update(repr(settings).encode())
    for record in records:
        values = record.items() if isinstance(record, dict) else ((f.name, getattr(record, f.name)) for f in fields(record))
        for name, value in values:
            digest.update(name.encode())
            if isinstance(value, np.ndarray):
                digest.update(str((value.shape, value.dtype.str)).encode())
                digest.update(value.tobytes())
            else:
                digest.update(repr(value).encode())
    return digest.hexdigest()


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


def prepare_bank(buildings, cfg, *, plan, corner_bonus_m=0., dynamic_max_geo_distance=False,
                 relay_options=None, randomize_base=False, relay_cache_dir=None, log=lambda message: None):
    from swarmecho.env.buildings import target_spawn_boxes
    first = buildings[0]
    cache_key = None
    if dynamic_max_geo_distance:
        extras = (plan, corner_bonus_m)
        if relay_options is not None and relay_options.mode == "global_candidates":
            from swarmecho.env.relay_candidates import settings
            extras += (settings(relay_options), cfg.comm_radius, cfg.comm_radius_base,
                       cfg.visual_radius, randomize_base)
        cache_key = _relay_bank_key(buildings, cfg, extras)
        if cache_key in _RELAY_BANK_CACHE:
            log("Reusing prepared relay map bank for the next agent stage.")
            return _RELAY_BANK_CACHE[cache_key]
    maps = []
    for i, building in enumerate(buildings):
        if any(getattr(building, key) != getattr(first, key) for key in
               ("cell_size_m", "wall_thickness_m", "tile_thickness_m")):
            raise ValueError("A training map bank must use the same cell size and wall/tile thicknesses.")
        graph = building_roadmap(building, cfg, plan=plan or dynamic_max_geo_distance, corner_bonus_m=corner_bonus_m)
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
        if dynamic_max_geo_distance:
            from swarmecho.env.relay import prepare_record
            record = prepare_record(record, cfg, building, options=relay_options,
                                    randomize_base=randomize_base, cache_dir=relay_cache_dir)
            if "gc_points" in record and (i == 0 or (i + 1) % 100 == 0 or i + 1 == len(buildings)):
                log(f"Global relay map {i + 1}/{len(buildings)}: {int(record['gc_real_size'])} candidates, "
                    f"{int(record['gc_random_size'])} random + {int(record['gc_anchor_size'])} anchors")
        maps.append(record)
        if i == 0 or (i + 1) % 100 == 0 or i + 1 == len(buildings):
            log(f"Compiled map bank {i + 1}/{len(buildings)}; {len(graph.vertices)} roadmap nodes")
    result = prepare_bank_from_records(maps, first, cfg, interiors=[b.interior_cells for b in buildings], log=log)
    if cache_key is not None:
        _RELAY_BANK_CACHE[cache_key] = result
    return result


def prepare_bank_from_records(maps, first, cfg, *, interiors=None, dynamic_max_geo_distance=False,
                              relay_options=None, randomize_base=False, relay_cache_dir=None, log=lambda message: None):
    """Pad already compiled per-map records for the JAX training batch."""
    if not maps:
        raise ValueError("A training map bank must contain at least one map.")
    maps = [dict(record, world_size=record.get("world_size", first.world_size_m)) for record in maps]
    cache_key = None
    if dynamic_max_geo_distance:
        extras = (first.cell_size_m, first.wall_thickness_m,
            first.tile_thickness_m, first.interior_cells.tobytes() if interiors is None
            else tuple(np.asarray(interior).tobytes() for interior in interiors))
        global_mode = relay_options is not None and relay_options.mode == "global_candidates"
        if global_mode:
            from swarmecho.env.relay_candidates import settings
            extras += (settings(relay_options), cfg.comm_radius, cfg.comm_radius_base,
                       cfg.visual_radius, randomize_base)
        cache_key = _relay_bank_key(maps, cfg, extras)
        if cache_key in _RELAY_BANK_CACHE:
            log("Reusing prepared relay records for the next agent stage.")
            return _RELAY_BANK_CACHE[cache_key]
        from swarmecho.env.relay import prepare_record
        if global_mode:
            prepared = []
            for i, record in enumerate(maps):
                prepared.append(prepare_record(record, cfg, first, options=relay_options,
                    randomize_base=randomize_base, cache_dir=relay_cache_dir))
                if i == 0 or (i + 1) % 100 == 0 or i + 1 == len(maps):
                    log(f"Prepared global relay map {i + 1}/{len(maps)}; "
                        f"{int(prepared[-1]['gc_real_size'])} candidates "
                        f"({int(prepared[-1]['gc_random_size'])} random + {int(prepared[-1]['gc_anchor_size'])} anchors)")
            maps = prepared
        else:
            maps = [record if "relay_predecessors" in record else prepare_record(record, cfg) for record in maps]
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
        fill = (1e6 if key in {"solid_min", "solid_max", "distances", "corner_distances", "vertices", "relay_distances"}
                else -1 if key == "relay_predecessors" else 0.)
        dtype = np.int32 if key in {"relay_predecessors", "relay_nodes", "relay_edges", "relay_edge_count", "relay_opening_count"} else np.float32
        if key.startswith("gc_"):
            dtype = maps[0][key].dtype
            fill = (1e6 if key in {"gc_points", "gc_hops"}
                    else -2 if key == "gc_parents" else 0)
        value = np.full((len(maps),) + shape, fill, dtype=dtype)
        for i, record in enumerate(maps):
            value[(i,) + tuple(slice(0, n) for n in record[key].shape)] = record[key]
        arrays[key] = value
    total = sum(a.nbytes for a in arrays.values())
    if "gc_points" in arrays:
        log(f"Global relay bank: candidates {arrays['gc_points'].shape}; "
            f"{sum(a.nbytes for k, a in arrays.items() if k.startswith('gc_')) / 2**20:.1f} MiB "
            f"({'randomized-base graphs' if 'gc_radio' in arrays else 'fixed-base hop labels'})")
    log(f"Map bank immutable arrays: {total / 2**20:.1f} MiB; solids {arrays['solid_min'].shape}"
        + (f", roadmaps {arrays['distances'].shape}" if "distances" in arrays else ", targetless"))
    # Transfer once. These are closed-over device arrays, never rollout leaves.
    device_arrays = {key: jnp.asarray(value) for key, value in arrays.items()}
    result = BuildingBank(envelope, device_arrays)
    if cache_key is not None:
        _RELAY_BANK_CACHE[cache_key] = result
    return result
