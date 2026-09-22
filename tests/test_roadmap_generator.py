import importlib.util
from pathlib import Path

import numpy as np


spec = importlib.util.spec_from_file_location(
    "roadmap_generator", Path(__file__).with_name("generate_obstacle_roadmap_testresult.py")
)
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def test_ranked_paths_are_unique_loopless_and_sorted():
    adjacency = [[] for _ in range(5)]
    for a, b, cost in [(0, 2, 1), (2, 1, 1), (0, 3, 1), (3, 1, 2),
                       (0, 4, 2), (4, 1, 2), (2, 3, .5)]:
        adjacency[a].append((b, cost))
        adjacency[b].append((a, cost))
    routes = generator.ranked_paths(adjacency, 10)
    weights = [dict(edges) for edges in adjacency]
    lengths = [sum(weights[a][b] for a, b in zip(p, p[1:])) for p in routes]
    assert lengths == sorted(lengths)
    assert lengths[0] == 2
    assert len({tuple(p) for p in routes}) == len(routes)
    assert all(len(set(p)) == len(p) for p in routes)
    assert generator.ranked_paths([[], []], 5) == []


def test_visibility_blocks_crossings_but_allows_parallel_clear_paths():
    visible = generator._visible(
        np.array([0., 0., 0.]), np.array([[3., 0., 0.], [0., 3., 0.]]),
        np.array([[1., -.5, -.5]]), np.array([[2., .5, .5]]),
    )
    assert visible.tolist() == [False, True]


def test_obstacles_default_off_even_for_an_obstacle_level(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from swarmecho.env.environment import EnvConfig

    building = SimpleNamespace(
        world_size_m=np.array([10., 10., 10.]), base_position_m=np.array([1., 1., 1.]),
        cell_size_m=5., interior_cells=np.ones((2, 2, 2), dtype=bool),
        wall_thickness_m=.25, tile_thickness_m=.25,
        solid_min_m=np.zeros((0, 3)), solid_max_m=np.zeros((0, 3)),
    )
    level = SimpleNamespace(env=EnvConfig(num_obstacles=3), building=building, building_name="test")
    monkeypatch.setattr(generator, "load_level", lambda _: level)
    def unexpected(*args, **kwargs):
        raise AssertionError("Random obstacles must be opt-in")
    monkeypatch.setattr(generator, "generate_obstacles", unexpected)
    output = generator.generate(tmp_path / "test.roadmap.json", level_name="test", target=[8., 8., 8.])
    data = generator.json.loads(output.read_text())
    assert not data["manifest"]["generated_obstacles"]
    assert len(data["layouts"]) == 1
    assert data["layouts"][0]["obstacle_min"] == []
    assert len(data["layouts"][0]["paths"]) == 1
