"""Geometric wall-sensor checks against analytic distances and compiled windows.

Run in the existing WSL environment:
    JAX_PLATFORMS=cpu uv run pytest tests/test_wall_sensing.py -v
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from swarmecho.core.config import EnvConfig, load_level, validate_radar_config
from swarmecho.env.buildings import building_snapshot, compile_building, make_cuboid_building
from swarmecho.env.environment import (
    make_env_fns,
    observation_dim,
    spherical_directions,
    wall_sensing_directions,
)


def _document():
    data = building_snapshot(make_cuboid_building((4, 4, 4)))
    data.update(name="wall_sensing_test", width=20.0, height=20.0, depth=20.0,
                base_position_m=[2.5, 2.5, 0.125])
    return data


def _window_building(axis, sign):
    """A 5 m cell wall with the production 2 x 2 m window and a wall behind it."""
    data = _document()
    key = "x" if axis == 0 else "y"
    geometry = data["geometry"]
    # A complete internal sheet prevents a ray from escaping around the wall.
    geometry[f"{key}_walls"].extend(
        [2, along, z] if axis == 0 else [along, 2, z]
        for along in range(4) for z in range(4)
    )
    window = [2, 1, 1] if axis == 0 else [1, 2, 1]
    geometry[f"{key}_windows"] = [window]
    rear = window.copy()
    rear[axis] = 3 if sign > 0 else 1
    geometry[f"{key}_walls"].append(rear)
    return compile_building(data)


def _closed_room():
    """A single 5 m interior cell, enclosed by six real wall/tile prisms."""
    data = _document()
    data.update(interior_cells=[[1, 1, 1]], target_exclusion_cells=[],
                base_position_m=[7.5, 7.5, 5.125])
    data["geometry"]["x_walls"].extend([[1, 1, 1], [2, 1, 1]])
    data["geometry"]["y_walls"].extend([[1, 1, 1], [1, 2, 1]])
    data["geometry"]["tiles"].extend([[1, 1, 1], [1, 1, 2]])
    return compile_building(data)


def _config(count=26, **overrides):
    return replace(EnvConfig(), num_agents=1, spawn_delay=0,
                   wall_sensing_layout="xyz", wall_sensing_rays=count,
                   wall_sensing_ray_max_distance=8.0, drone_radar_bins=16,
                   radar_mode="band_sep_counting", distance_bands=3, **overrides)


def _sensor(building, cfg):
    reset, step, observations, _ = make_env_fns(building, cfg, plan_geodesic=False)
    state = reset(jax.random.PRNGKey(0), target_pos=jnp.asarray([7.5, 7.5, 7.5]))
    return state, step, observations


def _wall_signals(observation, cfg):
    # Fixtures disable all optional self/probe features; six self values precede walls.
    values = np.asarray(observation)
    assert values.shape == (cfg.num_agents, observation_dim(cfg))
    assert np.isfinite(values).all()
    return values[:, 6:6 + cfg.wall_sensing_rays]


@pytest.mark.parametrize("count,plane_count,corner_count", [(14, 0, 8), (18, 12, 0), (26, 12, 8)])
def test_xyz_distribution_has_unit_unique_axes_and_cubic_symmetry(count, plane_count, corner_count):
    directions = wall_sensing_directions(count, "xyz")
    assert directions.shape == (count, 3)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-7)
    points = {tuple(row) for row in directions}
    assert len(points) == count
    for axis in np.eye(3, dtype=np.float32):
        assert tuple(axis) in points
        assert tuple(-axis) in points
    nonzero = np.count_nonzero(directions, axis=1)
    assert np.count_nonzero(nonzero == 1) == 6
    assert np.count_nonzero(nonzero == 2) == plane_count
    assert np.count_nonzero(nonzero == 3) == corner_count
    # Mirror reflections and swapping coordinate axes must preserve the set.
    for row in directions:
        for axis in range(3):
            mirrored = row.copy()
            mirrored[axis] *= -1
            assert tuple(mirrored) in points
        assert tuple(row[[1, 0, 2]]) in points
        assert tuple(row[[0, 2, 1]]) in points


@pytest.mark.parametrize("count", [8, 16, 25, 32])
def test_fibonacci_retains_legacy_direction_values_and_order(count):
    np.testing.assert_array_equal(wall_sensing_directions(count, "fibonacci"),
                                  spherical_directions(count))


@pytest.mark.parametrize("count", [14, 18, 26])
def test_every_xyz_ray_reports_analytic_first_hit_in_a_closed_room(count):
    cfg = _config(count)
    state, _, observations = _sensor(_closed_room(), cfg)
    state = state._replace(pos=jnp.asarray([[7.5, 7.5, 7.5]]))
    # Each physical inner face is 2.375 m from the cell centre. Unit axis,
    # plane-diagonal and corner-diagonal rays travel d, d*sqrt(2), d*sqrt(3).
    nonzero = np.count_nonzero(wall_sensing_directions(count, "xyz"), axis=1)
    distances = 2.375 * np.sqrt(nonzero)
    expected = 1.0 - distances / 8.0
    eager = _wall_signals(observations(state), cfg)[0]
    compiled = _wall_signals(jax.jit(observations)(state), cfg)[0]
    # First-hit distances use eight bisections: normalized error <= 1/256.
    np.testing.assert_allclose(eager, expected, rtol=0, atol=1 / 256 + 1e-6)
    np.testing.assert_allclose(compiled, expected, rtol=0, atol=1 / 256 + 1e-6)
    np.testing.assert_allclose(compiled, eager, rtol=0, atol=1e-6)


@pytest.mark.parametrize("count", [14, 18, 26])
@pytest.mark.parametrize("axis,sign", [(0, 1), (0, -1), (1, 1), (1, -1)],
                         ids=["+X", "-X", "+Y", "-Y"])
def test_axis_ray_sees_through_compiled_window_and_reports_frame_and_rear_wall(count, axis, sign):
    cfg = _config(count)
    state, _, observations = _sensor(_window_building(axis, sign), cfg)
    compiled = jax.jit(observations)
    direction = np.eye(3, dtype=np.float32)[axis] * sign
    ray = np.flatnonzero(np.all(wall_sensing_directions(count, "xyz") == direction, axis=1)).item()
    near_face = 10.0 - sign * 0.125
    # The window spans 6.5..8.5 m in both transverse coordinates.
    for distance, offset, expected in (
        (2.0, 0.0, 1.0 - 7.0 / 8.0),   # Through window: rear wall is the first hit.
        (2.0, 1.25, 1.0 - 2.0 / 8.0),  # Shift sideways: nearer window frame wins.
        (6.0, 0.0, 0.0),                # Window at 6 m; rear wall beyond range.
        (6.0, 1.25, 1.0 - 6.0 / 8.0),  # Same range, but a solid part of the wall.
        (8.5, 1.25, 0.0),               # Solid wall beyond the configured limit.
    ):
        position = np.full(3, 7.5, dtype=np.float32)
        position[axis] = near_face - sign * distance
        position[1 - axis] += offset
        query = state._replace(pos=jnp.asarray(position[None]))
        for observe in (observations, compiled):
            signal = _wall_signals(observe(query), cfg)[0, ray]
            assert signal == pytest.approx(expected, abs=1 / 256 + 1e-6), (
                f"Wrong first hit at {distance} m, transverse offset {offset} m"
            )


@pytest.mark.parametrize("wall_range,visual_range,expected", [
    (4.0, 1.0, 0.0), (4.0, 20.0, 0.0),
    (8.0, 1.0, 0.25), (8.0, 20.0, 0.25),
])
def test_wall_range_controls_reach_and_normalization_independently(wall_range, visual_range, expected):
    cfg = replace(_config(), wall_sensing_ray_max_distance=wall_range, visual_radius=visual_range)
    state, _, observations = _sensor(_window_building(0, 1), cfg)
    state = state._replace(pos=jnp.asarray([[3.875, 8.75, 7.5]]))
    ray = np.flatnonzero(np.all(wall_sensing_directions(26, "xyz") == [1, 0, 0], axis=1)).item()
    signal = _wall_signals(jax.jit(observations)(state), cfg)[0, ray]
    # Wall frame is 6 m away: outside a 4 m wall range, inside an 8 m range.
    assert signal == pytest.approx(expected, abs=1 / 256 + 1e-6)


def test_target_visibility_still_uses_visual_range_while_wall_readings_stay_equal():
    readings = []
    for visual_range, sees_target in ((1.0, False), (4.0, True)):
        cfg = replace(_config(), visual_radius=visual_range)
        state, step, observations = _sensor(_window_building(0, 1), cfg)
        state = state._replace(pos=jnp.asarray([[3.875, 7.5, 7.5]]),
                               target_pos=jnp.asarray([3.875, 10.5, 7.5]))
        next_state = jax.jit(step)(state, jnp.zeros((1, 3), dtype=jnp.float32))
        assert bool(next_state.directly_sees_target[0]) == sees_target
        readings.append(_wall_signals(jax.jit(observations)(next_state), cfg))
    np.testing.assert_array_equal(*readings)


def test_xyz_detects_a_centred_window_that_the_legacy_fibonacci_ray_misses():
    building = _window_building(0, 1)
    readings = {}
    for layout, count in (("xyz", 26), ("fibonacci", 16)):
        cfg = replace(_config(), wall_sensing_layout=layout, wall_sensing_rays=count)
        state, _, observations = _sensor(building, cfg)
        state = state._replace(pos=jnp.asarray([[3.875, 7.5, 7.5]]))
        directions = wall_sensing_directions(count, layout)
        ray = np.argmax(directions[:, 0])
        readings[layout] = _wall_signals(jax.jit(observations)(state), cfg)[0, ray]
    assert readings["xyz"] == 0.0
    assert readings["fibonacci"] > 0.15  # Its angled ray hits the frame about 6.4 m away.


@pytest.mark.parametrize("name", ["C01_linear_hallway", "C02_corner_hallway", "C03_space_station",
                                 "C05a_cellar", "C05b_cellar"])
def test_c_levels_select_xyz_26_and_independent_8_metre_wall_range(name):
    cfg = load_level(name).env
    assert cfg.wall_sensing_layout == "xyz"
    assert cfg.wall_sensing_rays == 26
    assert cfg.wall_sensing_ray_max_distance == 8.0
    assert cfg.visual_radius == 4.0
    assert cfg.drone_radar_bins == 16


@pytest.mark.parametrize("count", [8, 16, 32])
def test_xyz_rejects_unsupported_ray_counts(count):
    with pytest.raises(ValueError, match="14, 18, or 26"):
        validate_radar_config(_config(count))
    with pytest.raises(ValueError, match="14, 18, or 26"):
        wall_sensing_directions(count, "xyz")


@pytest.mark.parametrize("distance", [0.0, -1.0, float("inf"), float("nan")])
def test_wall_range_must_be_positive_and_finite(distance):
    with pytest.raises(ValueError, match="wall_sensing_ray_max_distance"):
        validate_radar_config(replace(_config(), wall_sensing_ray_max_distance=distance))
