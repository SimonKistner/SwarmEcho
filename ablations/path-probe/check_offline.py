"""Opt-in checks for offline probe collection, fitting, and reporting."""

from types import SimpleNamespace
from typing import NamedTuple
import copy
import time

from flax import nnx
from omegaconf import OmegaConf
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geo_experiments import suite
from geo_experiments.registry import get_probe, job_probe, PROBES
from geo_experiments.shared import artifacts, collector, fitting, path_geometry as geometry, runner
from geo_experiments.shared.definitions import Definition
from geo_experiments.shared.inputs import input_request, project_event_inputs, resolve_inputs
from geo_experiments.shared.path_collection import (
    candidate_buffer_bytes,
    input_rows,
    is_allocation_failure,
)
from geo_experiments.shared.rollout import make_collector
from geo_experiments.shared.sampling import hierarchical_weights, mask_scores, retained_candidates
from geo_experiments.shared.targets import build_target, encode_output


# Small deterministic rollout fixture

class State(NamedTuple):
    pos: object
    active: object
    target_known: object
    directly_sees_target: object
    base_pos: object
    target_pos: object
    solid_min: object
    solid_max: object
    step: object
    done: object
    success: object
    base_target_known: object
    target_required_agents: object
    spawn_pair_attempts: object
    adaptive: object = None


class Actor(nnx.Module):
    vote_mode = None

    def __call_team__(self, obs, hidden, signature, value, **kwargs):
        updated = jnp.broadcast_to(obs[:, :1] + 1., hidden.shape)
        return updated, signature + 1., value + 1., jnp.zeros((2, 3)), None


class Model(nnx.Module):
    hidden_dim = 3
    tarmac_sig_dim = 1
    tarmac_val_dim = 1
    memory_comm_every_k_steps = 2

    def __init__(self):
        self.actor = Actor()

    def initial_actor_hidden(self, shape, *, num_agents):
        return jnp.zeros((*shape, num_agents, 3))

    def initial_actor_signature(self, shape, *, num_agents):
        return jnp.zeros((*shape, num_agents, 1))

    initial_actor_value = initial_actor_signature


def environment():
    def reset(key):
        return State(jnp.array([[2.5, 2.5, 2.5], [3., 2.5, 2.5]]), jnp.array([True, True]),
            jnp.array([False, False]), jnp.array([False, False]), jnp.array([2.5, 2.5, 2.5]),
            jnp.array([7.5, 2.5, 2.5]), jnp.empty((0, 3)), jnp.empty((0, 3)),
            jnp.int32(0), jnp.bool_(False), jnp.bool_(False), jnp.bool_(False), jnp.int32(1), jnp.int32(1))

    def step(state, action):
        t = state.step + 1
        return state._replace(step=t, done=t >= 6, active=jnp.array([t < 4, True]),
            target_known=jnp.array([t >= 1, t >= 3]), directly_sees_target=jnp.array([t >= 1, False]),
            base_target_known=t >= 3)

    def observations(state):
        return jnp.full((2, 1), state.step, dtype=jnp.float32)

    return reset, step, observations, None


def level():
    return SimpleNamespace(env=SimpleNamespace(max_steps=6, num_agents=2, comm_radius=10., comm_radius_base=10., base_keeps_informing=False),
        building_name="Deterministic smoke fixture", map_names=["fixture"],
        building=SimpleNamespace(solid_min_m=np.empty((0, 3)), solid_max_m=np.empty((0, 3)),
            interior_cells=np.array([[0,0,0],[1,0,0]]), cell_size_m=5., vertices=np.empty((0,3))))


# Path geometry and readout

@pytest.fixture
def path_config():
    config = get_probe("path-predictor").settings()
    config["run"] = dict(name="path_test", timezone="Europe/Berlin", output_root="geo_experiments/outputs")
    config["policy"]["checkpoint"] = "outputs/example/checkpoints/ckpt_1"
    config["probe"].update(epochs=3, patience=3, updates_per_epoch=2, hidden_sizes=[8], seeds=[0], batch_size=8)
    return config


def graph_for(building):
    from scipy.sparse.csgraph import shortest_path
    vertices = building.vertices
    edges = np.array([geometry.visible(v, vertices, building.solid_min_m, building.solid_max_m) for v in vertices])
    np.fill_diagonal(edges, False)
    distances = np.where(edges, np.linalg.norm(vertices[:, None] - vertices, axis=-1), np.inf)
    np.fill_diagonal(distances, 0)
    return SimpleNamespace(vertices=vertices, visible_edges=edges, clearance=1e-4,
                           distances=shortest_path(distances, directed=False))


def building(rotation=0):
    matrix = geometry.ROTATIONS[rotation]
    translation = np.array([[0, 0, 0], [15, 0, 0], [15, 15, 0], [0, 15, 0]])[rotation]
    turn = lambda p: np.asarray(p) @ matrix.T + translation
    lo, hi = turn([[5., 4., 0.]]), turn([[10., 11., 15.]])
    item = SimpleNamespace(solid_min_m=np.minimum(lo, hi), solid_max_m=np.maximum(lo, hi),
        interior_cells=np.array([(x, y, z) for z in range(3) for y in range(3) for x in range(3)]), cell_size_m=5.,
        vertices=turn([[4.5, 3.5, 2.5], [10.5, 3.5, 2.5], [4.5, 11.5, 2.5], [10.5, 11.5, 2.5]]))
    return item, turn


def test_reference_route_is_rotation_invariant_and_reward_length_aligned(monkeypatch):
    import swarmecho.env.roadmap_cpu as cpu
    monkeypatch.setattr(cpu, "building_roadmap", lambda b, cfg, **kw: graph_for(b))
    masks, lengths = [], []
    for r in range(4):
        b, turn = building(r)
        route = geometry.ReferencePlanner(b, None).route(turn([2.5, 7.5, 2.5]), turn([12.5, 7.5, 2.5]))
        masks.append(route["mask"])
        lengths.append(route["length"])
        np.testing.assert_allclose(route["points"][[0, -1]], turn([[2.5, 7.5, 2.5], [12.5, 7.5, 2.5]]), atol=1e-5)
        assert route["length"] > 10  # Wall requires a real detour.
    for mask in masks[1:]:
        np.testing.assert_array_equal(mask, masks[0])
    np.testing.assert_allclose(lengths, lengths[0])


@pytest.mark.parametrize("disconnected", [False, True])
def test_missing_reward_reference_is_distinct_from_a_reconstruction_bug(monkeypatch, disconnected):
    import swarmecho.env.roadmap_cpu as cpu
    b, _ = building()
    # Both endpoints are in free space, but this sparse graph either has no
    # target attachment or uses the reward's 1e6 disconnected-distance sentinel.
    b.vertices = np.array([[2.5, 7.5, 2.5], [12.5, 7.5, 2.5]]) if disconnected else np.array([[2.5, 7.5, 2.5]])
    graph = graph_for(b)
    if disconnected:
        graph.distances = np.where(np.isfinite(graph.distances), graph.distances, 1e6)
    monkeypatch.setattr(cpu, "building_roadmap", lambda *args, **kwargs: graph)
    with pytest.raises(geometry.NoReferenceRoute) as caught:
        geometry.ReferencePlanner(b, None).route([2.5, 7.5, 2.5], [12.5, 7.5, 2.5])
    assert caught.value.base_attachments == 1
    assert caught.value.target_attachments == int(disconnected)


def test_base_frame_handles_upstairs_and_signed_offsets():
    path = np.array([[0, 0, 0], [-10, 0, 0], [-10, 0, -10]])
    mask = geometry.rasterize(path, 5.)
    included = set(map(tuple, geometry.OFFSETS[mask]))
    assert {(0, 0, 0), (-1, 0, 0), (-2, 0, 0), (-2, 0, -1), (-2, 0, -2)} == included


def test_real_building_boolean_interior_grid_is_converted_to_coordinates(monkeypatch):
    import swarmecho.env.roadmap_cpu as cpu
    b, turn = building()
    b.interior_cells = np.ones((3,3,3), dtype=bool)
    monkeypatch.setattr(cpu, "building_roadmap", lambda b, cfg, **kw: graph_for(b))
    planner = geometry.ReferencePlanner(b, None)
    assert planner.cells.shape == (27,3)
    assert planner.route([2.5,7.5,2.5],[12.5,7.5,2.5])["mask"].any()


def test_grid_face_base_choice_is_rotation_invariant(monkeypatch):
    import swarmecho.env.roadmap_cpu as cpu
    monkeypatch.setattr(cpu, "building_roadmap", lambda b, cfg, **kw: graph_for(b))
    masks = []
    for r in range(4):
        b, turn = building(r)
        masks.append(geometry.ReferencePlanner(b, None).route(turn([2.5,5,2.5]),turn([12.5,5,2.5]))["mask"])
    for mask in masks[1:]:
        np.testing.assert_array_equal(mask,masks[0])


def test_rasterization_traverses_segment_interiors_not_only_waypoints():
    mask = geometry.rasterize([[0, 0, 0], [10, 0, 0]], 5.)
    assert set(map(tuple, geometry.OFFSETS[mask])) == {(0, 0, 0), (1, 0, 0), (2, 0, 0)}
    diagonal = geometry.rasterize([[0, 0, 0], [5, 5, 0]], 5.)
    assert set(map(tuple, geometry.OFFSETS[diagonal])) == {(0, 0, 0), (1, 1, 0)}
    with pytest.raises(ValueError, match="supported"):
        geometry.rasterize([[0, 0, 0], [20, 0, 0]], 5.)


def test_even_candidate_selection_includes_endpoints_and_never_duplicates():
    np.testing.assert_array_equal(retained_candidates(np.arange(10) * 10, 4), [0, 3, 6, 9])
    np.testing.assert_array_equal(retained_candidates([2, 12, -1, -1], 20), [0, 1])
    np.testing.assert_array_equal(retained_candidates([2, 12, 22], 1), [0])


def test_weights_are_episode_then_agent_balanced():
    episodes = np.array([0, 0, 0, 1, 1, 1, 1])
    agents = np.array([0, 0, 1, 0, 0, 0, 0])
    weights = hierarchical_weights(episodes, agents)
    assert weights.sum() == pytest.approx(1)
    assert weights[episodes == 0].sum() == pytest.approx(.5)
    assert weights[(episodes == 0) & (agents == 0)].sum() == pytest.approx(.25)
    assert weights[(episodes == 0) & (agents == 1)].sum() == pytest.approx(.25)


def test_three_stage_sampler_matches_episode_agent_weights():
    from geo_experiments.shared.sampling import HierarchicalSampler
    episodes=np.array([0,0,0,1,1,1,1])
    agents=np.array([0,0,1,0,0,0,0])
    sampled=HierarchicalSampler(episodes,agents,np.random.default_rng(1)).draw(100000)
    observed=np.bincount(sampled,minlength=7)/len(sampled)
    np.testing.assert_allclose(observed,hierarchical_weights(episodes,agents),atol=.004)


def test_similarity_penalizes_missing_and_extra_cells_without_background_credit():
    truth = np.zeros((4, 125), bool)
    truth[:, :2] = True
    prediction = truth.astype(float)
    prediction[1, 2] = 1  # One extra: 4/5 similarity.
    prediction[2, 1] = 0  # One missing: 2/3 similarity.
    prediction[3] = 0
    exact, similarity = mask_scores(prediction, truth)
    np.testing.assert_array_equal(exact, [1, 0, 0, 0])
    np.testing.assert_allclose(similarity, [1, .8, 2/3, 0])


def test_first_and_repeated_grus_survive_missing_base_and_keep_agent_times(path_config):
    path_config["definition"]["sampling"]["repeated_snapshot_cap_per_agent"] = 2
    batch = dict(finder_id=np.array([0]), finder_step=np.array([3]), base_step=np.array([-1]),
        informed_first=np.array([[2, 4]]), finder_gru=np.ones((1, 3)), finder_message=np.ones((1, 2)),
        base_message=np.zeros((1, 2)), candidate_gru=np.arange(1*3*2*3).reshape(1, 3, 2, 3),
        candidate_steps=np.array([[[2, 4], [12, 14], [22, -1]]]))
    rows = input_rows(batch, 0, path_config, 7)
    assert not rows["base"]
    assert len(rows["all_knowing_after_info"]) == 2
    assert [(r["agent"], r["step"]) for r in rows["all_knowing_till_end"]] == [(0, 2), (0, 22), (1, 4), (1, 14)]
    assert all(r["episode"] == 7 for group in rows.values() for r in group)


def test_candidate_memory_estimate_is_independent_of_retained_cap(path_config):
    model = SimpleNamespace(hidden_dim=256)
    level = SimpleNamespace(env=SimpleNamespace(max_steps=700, num_agents=7))
    first = candidate_buffer_bytes(model, level, path_config, 4096)
    assert first == 4096 * 71 * 7 * (256 * 4 + 4)
    assert "capture_memory_mib" not in path_config["dataset"]
    assert path_config["dataset"]["eval_batch_size"] == 4096
    path_config["definition"]["sampling"]["repeated_snapshot_cap_per_agent"] = 1
    assert candidate_buffer_bytes(model, level, path_config, 4096) == first


@pytest.mark.parametrize("error,expected", [
    (RuntimeError("RESOURCE_EXHAUSTED: Out of memory while trying to allocate 1GiB"), True),
    (RuntimeError("RESOURCE_EXHAUSTED: Failed to allocate request"), True),
    (RuntimeError("CUDA_ERROR_OUT_OF_MEMORY"), True),
    (MemoryError(), True),
    (RuntimeError("RESOURCE_EXHAUSTED: Thread quota exceeded"), False),
    (RuntimeError("CUDA_ERROR_ILLEGAL_ADDRESS"), False),
    (RuntimeError("Incompatible shapes"), False),
])
def test_only_allocation_failures_allow_retry(error, expected):
    assert is_allocation_failure(error) == expected


@pytest.mark.parametrize("field,value", [("every_steps", 0), ("every_steps", True)])
def test_invalid_capture_options_are_rejected(path_config, field, value):
    raw = copy.deepcopy(path_config["definition"])
    raw["inputs"]["all_knowing_till_end"][field] = value
    from geo_experiments.shared.definitions import Definition
    with pytest.raises(ValueError, match="positive integer"):
        Definition(raw).validate(runnable=True)


def test_path_requests_resolve_through_the_shared_input_capability_registry(path_config):
    from geo_experiments.shared.inputs import input_request
    request=input_request(path_config)
    assert request.has_agent_events and request.needs_finder and request.needs_base
    assert dict(request.fields)["all_knowing_till_end"] == "candidate_gru"


def synthetic_arrays():
    from geo_experiments.shared.artifacts import add_group
    arrays = {}
    labels = np.zeros((6, 125), bool)
    labels[:, 62] = True
    labels[::2, 63] = True
    add_group(arrays, "labels", {"path": labels})
    add_group(arrays, "episodes", dict(episode_id=np.arange(6), map_id=np.zeros(6, int),
        required_agents=np.array([1, 2, 1, 2, 1, 2]), rotation=np.zeros(6, int), anchor=np.tile([2.5, 2.5, 2.5], (6, 1)),
        base_position=np.tile([2.5, 2.5, 2.5], (6, 1)), target_position=np.tile([7.5, 2.5, 2.5], (6, 1))))
    add_group(arrays, "geometry", {"map_names": np.array(["synthetic"]), "cell_size_m": np.array([5.]), "cell_offsets": geometry.OFFSETS})
    add_group(arrays, "paths", {"offsets": np.arange(7)*2, "points": np.tile([[2.5,2.5,2.5],[7.5,2.5,2.5]], (6,1))})
    add_group(arrays, "maps/0", {"solid_min": np.array([[0,0,0]]), "solid_max": np.array([[.2,15,15]]), "cells": np.array([[0,0,0],[1,0,0]])})
    add_group(arrays, "splits", {"train": np.array([0,1]), "validation": np.array([2,3]), "test": np.array([4,5])})
    for source in get_probe("path-predictor").settings()["definition"]["inputs"]:
        add_group(arrays, f"inputs/{source}", dict(features=np.eye(2)[np.arange(6)%2].astype(np.float32), episode=np.arange(6),
            agent=np.zeros(6,int), step=np.arange(6)+2, informed_step=np.ones(6,int)))
    return arrays


def test_real_binary_training_saves_probabilities_and_no_test_episode_is_trained(path_config):
    from geo_experiments.shared.mask_training import train_mask
    from geo_experiments.shared.artifacts import group
    arrays = synthetic_arrays()
    metadata, fitted = train_mask("finder_gru", 0, group(arrays, "inputs/finder_gru"), arrays["labels/path"], group(arrays, "splits"), path_config)
    assert metadata["status"] == "complete"
    assert fitted["probabilities"].shape == (2, 125)
    assert np.all((fitted["probabilities"] >= 0) & (fitted["probabilities"] <= 1))
    np.testing.assert_array_equal(fitted["row_indices"], [4,5])


def test_report_aggregation_balances_agents_and_uses_incorrect_only_histogram():
    from geo_experiments.shared.reports.path_mask import aggregate
    records = {("finder_gru", 0): dict(episodes=np.array([0,0,1]), agents=np.array([0,0,0]),
        exact=np.array([1.,1.,0.]), similarity=np.array([1.,1.,.5]))}
    result = aggregate(records, "finder_gru", [0,1])
    assert result["exact"] == pytest.approx(50)
    assert result["similarity"] == pytest.approx(75)
    assert result["hist"].sum() == pytest.approx(100)
    assert result["hist"][5] == pytest.approx(100)


def test_report_histogram_shows_similarity_instead_of_its_complement():
    from geo_experiments.shared.reports.path_mask import aggregate
    records = {("finder_gru", 0): dict(episodes=np.array([0,1]), agents=np.array([0,0]),
        exact=np.array([1.,0.]), similarity=np.array([1.,.85]))}
    summary = aggregate(records, "finder_gru", [0,1])
    assert summary["hist"][8] == pytest.approx(100)
    assert summary["hist"][1] == 0


def test_report_thresholds_recompute_masks_from_probabilities_without_refitting():
    from geo_experiments.shared.reports.path_mask import prediction_scores
    labels = np.zeros((1,125), dtype=bool)
    labels[0, :2] = True
    probabilities = np.zeros((1,125), dtype=float)
    probabilities[0, :3] = [.95,.80,.75]
    # At 75%, the third cell is an extra. At 80%, the exact path is selected.
    # At 90%, one true cell is missed. All 122 empty cells earn no credit.
    expected = {50:(0,.8), 75:(0,.8), 80:(1,1), 90:(0,2/3), 95:(0,2/3)}
    for threshold, (exact, similarity) in expected.items():
        scored = prediction_scores(probabilities, labels, np.array([0]), threshold/100)
        assert scored[0][0] == exact
        assert scored[1][0] == pytest.approx(similarity)


def test_report_building_uses_inspector_snapshot_and_verified_historical_map(tmp_path):
    import json
    import yaml
    from geo_experiments.shared.reports.building import building_for_report
    from geo_experiments.shared.common import fingerprint
    from swarmecho.visualize.inspector import building_geometry
    arrays = synthetic_arrays()
    snapshot = {"format":"swarmecho-map/v1", "building_cell_grid":{"cols":2,"rows":1,"layers":1},
        "cell_size_m":5, "wall_thickness_m":.25, "tile_thickness_m":.5,
        "interior_cells":[[0,0,0],[1,0,0]],
        "geometry":{"x_walls":[[0,0,0],[1,0,0],[2,0,0]],"y_walls":[],"tiles":[[0,0,0],[1,0,0],[0,0,1],[1,0,1]]}}
    arrays["maps/0/building_snapshot"] = np.array(json.dumps(snapshot))
    manifest = {"config":{"policy":{"checkpoint":str(tmp_path/"checkpoints"/"frozen")}}}
    expected = building_geometry({"building_snapshot":snapshot})
    assert building_for_report(arrays,0,manifest) == expected
    del arrays["maps/0/building_snapshot"]
    pool = tmp_path / "pool"
    (pool/"maps").mkdir(parents=True)
    path = pool / "maps" / "synthetic.yaml"
    path.write_text(yaml.safe_dump(snapshot),encoding="utf-8")
    identities = [{"name":"synthetic","sha256":fingerprint(path)}]
    manifest["provenance"] = {"identity":{"maps":identities}}
    (tmp_path/"probe_maps").mkdir()
    (tmp_path/"probe_maps"/"selection.json").write_text(json.dumps({"pool":str(pool),"maps":identities}),encoding="utf-8")
    assert building_for_report(arrays,0,manifest) == expected
    path.write_text("changed map",encoding="utf-8")
    assert building_for_report(arrays,0,manifest)["source"] == "saved collision geometry"


def test_report_defaults_to_80_percent_on_shared_episodes_but_keeps_full_populations(path_config, tmp_path, monkeypatch):
    from html.parser import HTMLParser
    from geo_experiments.shared import artifacts, dashboard
    from geo_experiments.shared.reports import path_mask
    arrays = synthetic_arrays()
    for source in path_config["definition"]["inputs"]:
        rows = np.array([4] if source == "base" else [4,5])
        artifacts.add_group(arrays,f"probes/path/{source}/seed_0",dict(row_indices=rows,
            probabilities=arrays["labels/path"][rows].astype(np.float32)*.8+.1, threshold=np.array(.5),
            history=np.array([[1,.12,0.,.11,0.],[2,.10,.5,.098,.5]]),best_epoch=np.array(2)))
    monkeypatch.setattr(path_mask,"build_dashboard",lambda:None)
    monkeypatch.setattr(path_mask,"report_controls",lambda folder:"")
    monkeypatch.setattr(dashboard,"report_id",lambda folder:"fixture")
    class Scores(HTMLParser):
        def __init__(self): super().__init__();self.cells=[]
        def handle_starttag(self,tag,attrs):
            attrs=dict(attrs)
            if tag=="td" and attrs.get("data-compare-row","").endswith(" / Overall"):
                self.cells.append(attrs)
    report=path_mask.generate_report(tmp_path,dict(config=path_config,collection={}),arrays)
    page=report.read_text(encoding="utf-8")
    parsed=Scores();parsed.feed(page)
    assert len(parsed.cells)==5*2*5*2  # Inputs, scores, thresholds, populations.
    assert {cell["data-evaluation-threshold"] for cell in parsed.cells}=={"50","75","80","90","95"}
    visible=[cell for cell in parsed.cells if "hidden" not in cell]
    assert len(visible)==10
    assert all(cell["data-evaluation-threshold"]=="80" and cell["data-evaluation-population"]=="common" for cell in visible)
    assert all("1 episodes" in cell["title"] for cell in visible)
    full=[cell for cell in parsed.cells if cell["data-evaluation-threshold"]=="80" and cell["data-evaluation-population"]=="available"]
    assert all(("1 episodes" if "Base stored message" in cell["data-compare-row"] else "2 episodes") in cell["title"] for cell in full)
    assert 'value="80" selected' in page and 'class="loss-grid"' in page
    assert 'BCE loss · lower is better' in page and '>Epoch</text>' in page
    assert 'Compare inputs on the same' not in page


def test_shared_jax_collector_captures_post_information_states_and_stops_decommissioned_agents():
    import jax
    from geo_experiments.shared.rollout import make_collector
    collect = make_collector(level(), agent_capture={"every_steps": 2}, environment=environment())
    batch = jax.device_get(collect(Model(), jax.random.split(jax.random.PRNGKey(0), 1)))
    np.testing.assert_array_equal(batch["informed_first"][0], [1,3])
    np.testing.assert_array_equal(batch["candidate_steps"][0, :2], [[1,3],[3,5]])
    assert np.all(batch["candidate_steps"][0, 2:] == -1)
    np.testing.assert_array_equal(batch["candidate_gru"][0, 0, 0], [2,2,2])
    assert batch["finder_step"][0] == 2 and batch["base_step"][0] == 2


@pytest.mark.parametrize("route_failure", [None, "missing_reference", "construction_bug", "oom_rollout", "oom_transfer", "oom_repeated", "runtime_error", "oom_at_one"])
def test_collector_fit_report_pipeline_keeps_episode_splits_and_nested_examples(path_config, tmp_path, monkeypatch, route_failure):
    import json
    import jax
    import swarmecho.env.roadmap_cpu as cpu
    from geo_experiments.shared import path_collection, rollout, fitting, artifacts, dashboard
    from geo_experiments.shared.reports import path_mask
    from html.parser import HTMLParser

    tiny = level()
    tiny.building, _ = building()
    path_config["policy"]["map_source"] = "training"
    path_config["dataset"].update(train_samples=4, validation_samples=2, test_samples=2, max_episodes=10, eval_batch_size=2)
    path_config["definition"]["inputs"]["all_knowing_till_end"]["every_steps"] = 2
    path_config["probe"].update(epochs=1, updates_per_epoch=1)
    if route_failure == "oom_at_one":
        path_config["dataset"]["eval_batch_size"] = 1
    if route_failure == "oom_repeated":
        path_config["dataset"]["eval_batch_size"] = 4
    monkeypatch.setattr(cpu, "building_roadmap", lambda b, cfg, **kw: graph_for(b))
    runtime_keys = []
    def make_collector(l, **kwargs):
        real_collect = rollout.make_collector(l, environment=environment(), **kwargs)
        def collect(model, keys):
            runtime_keys.append(np.asarray(keys).copy())
            if route_failure == "oom_repeated" and len(runtime_keys) in (2, 3):
                raise RuntimeError("RESOURCE_EXHAUSTED: Failed to allocate request")
            if len(runtime_keys) == 2:
                if route_failure in ("oom_rollout", "oom_at_one"):
                    raise RuntimeError("RESOURCE_EXHAUSTED: Out of memory while trying to allocate buffer")
                if route_failure == "runtime_error":
                    raise RuntimeError("CUDA_ERROR_ILLEGAL_ADDRESS")
            return real_collect(model, keys)
        return collect
    monkeypatch.setattr(path_collection, "make_collector", make_collector)
    real_device_get = jax.device_get
    def device_get(tree):
        if route_failure == "oom_transfer" and len(runtime_keys) == 2:
            raise RuntimeError("RESOURCE_EXHAUSTED: Out of memory during asynchronous device transfer")
        return real_device_get(tree)
    monkeypatch.setattr(jax, "device_get", device_get)
    monkeypatch.setattr("swarmecho.training.train._validate_evaluation_pairs", lambda *args: None)
    original_route = geometry.ReferencePlanner.route
    calls = 0
    def route_with_one_failure(planner, base, target):
        nonlocal calls
        calls += 1
        if calls == 1:
            if route_failure == "missing_reference":
                raise geometry.NoReferenceRoute(1, 0)
            if route_failure == "construction_bug":
                raise ValueError("Broken successor reconstruction")
        return original_route(planner, base, target)
    monkeypatch.setattr(geometry.ReferencePlanner, "route", route_with_one_failure)
    if route_failure in ("runtime_error", "oom_at_one"):
        with pytest.raises(RuntimeError, match="CUDA_ERROR_ILLEGAL_ADDRESS" if route_failure == "runtime_error" else "RESOURCE_EXHAUSTED"):
            path_collection.collect_path_samples(Model(), [tiny], path_config, tmp_path)
        assert len(runtime_keys) == 2  # No retry for unrelated errors or batch-one OOM.
        progress = json.loads((tmp_path / "run.json").read_text())["progress"]
        assert progress["accepted"] == progress["attempted"] == path_config["dataset"]["eval_batch_size"]
        assert not progress["oom_reductions"]
        return
    if route_failure == "construction_bug":
        with pytest.raises(ValueError, match=r"Route construction failed on map 0 .*episode 0"):
            path_collection.collect_path_samples(Model(), [tiny], path_config, tmp_path)
        diagnostic = json.loads((tmp_path / "reference_route_failures.jsonl").read_text())
        assert diagnostic["reason"] == "route_construction_error"
        assert diagnostic["error"] == "Broken successor reconstruction"
        return
    arrays, metadata = path_collection.collect_path_samples(Model(), [tiny], path_config, tmp_path)
    rejected = int(route_failure == "missing_reference")
    assert metadata["attempted_episodes"] == 8 + rejected
    assert metadata["reference_route_rejections"] == rejected
    assert metadata["per_map"][0]["reference_route_rejections"] == rejected
    assert set(metadata["timing_seconds"]) == {"rollout_and_transfer", "reference_routes", "row_assembly", "oom_recovery"}
    recovered = route_failure in ("oom_rollout", "oom_transfer", "oom_repeated")
    assert len(metadata["oom_reductions"]) == (2 if route_failure == "oom_repeated" else int(recovered))
    if recovered:
        initial = path_config["dataset"]["eval_batch_size"]
        assert metadata["requested_capture_batch_size"] == metadata["initial_capture_batch_size"] == initial
        assert metadata["capture_batch_size"] == 1
        for index, reduction in enumerate(metadata["oom_reductions"]):
            assert reduction["attempted_episodes"] == initial
            assert reduction["to_batch"] == reduction["from_batch"] // 2
            np.testing.assert_array_equal(runtime_keys[index + 1][:reduction["to_batch"]], runtime_keys[index + 2])
        np.testing.assert_array_equal(arrays["episodes/episode_id"], np.arange(8))
    if rejected:
        diagnostic = json.loads((tmp_path / "reference_route_failures.jsonl").read_text())
        assert diagnostic["episode_id"] == 0 and diagnostic["target_attachments"] == 0
        assert diagnostic["reason"] == "no_finite_reference"
        assert 0 not in arrays["episodes/episode_id"]
    splits = [set(arrays[f"splits/{s}"]) for s in ("train", "validation", "test")]
    assert list(map(len, splits)) == [4,2,2]
    assert not (splits[0] & splits[1] or splits[0] & splits[2] or splits[1] & splits[2])
    for source in path_config["definition"]["inputs"]:
        assert set(arrays[f"inputs/{source}/episode"]) == set(range(8))
    fitted, jobs = fitting.fit_experiment(path_config, arrays, tmp_path)
    assert len(jobs) == 5 and all(job["status"] == "complete" for job in jobs)
    monkeypatch.setattr(path_mask, "build_dashboard", lambda: None)
    monkeypatch.setattr(path_mask, "report_controls", lambda folder: "")
    monkeypatch.setattr(dashboard, "report_id", lambda folder: "fixture")
    manifest = dict(config=path_config, collection=metadata, provenance={"identity": {"maps": [{"name":"fixture"}]}})
    artifacts.update_run(tmp_path, **manifest)
    output = path_mask.generate_report(tmp_path, manifest, {**arrays, **fitted})
    class Structure(HTMLParser):
        def __init__(self):
            super().__init__(); self.external_scripts=[]; self.examples=0; self.open_details=0
        def handle_starttag(self, tag, attrs):
            attrs=dict(attrs)
            if tag=="script" and "src" in attrs: self.external_scripts.append(attrs["src"])
            if tag=="details":
                self.open_details += "open" in attrs
                self.examples += "data-example" in attrs
    page=output.read_text(encoding="utf-8")
    parsed=Structure();parsed.feed(page)
    assert not parsed.external_scripts and parsed.examples >= 5 and parsed.open_details == 0
    assert "Average path similarity" in page and "path-examples" in page
    assert ("sampled episodes had no finite reward-reference route" in page) == bool(rejected)


# First-information attribution

def test_first_informed_actor_states_are_counted_once_and_done_is_unchanged():
    env=environment()
    keys=jax.random.split(jax.random.PRNGKey(7),1)
    ordinary=jax.device_get(make_collector(level(),environment=env)(Model(),keys))
    measured=jax.device_get(make_collector(level(),environment=env,knowledge_capture=True)(Model(),keys))
    for key in ordinary:
        np.testing.assert_array_equal(ordinary[key],measured[key])
    counts=measured['knowledge_first_counts'][0]
    assert counts.sum()==2
    assert counts[1]==1 # The finder saw the target directly.
    assert counts[4]==1 # Its peer learned before the base was informed.
    assert measured['knowledge_informed_steps'].sum()==6
    assert measured['knowledge_unattributed_events'].sum()==0


def test_base_only_acquisition_and_actual_cached_message_are_distinct_counters():
    reset,original_step,observations,_=environment()
    def step(state,action):
        next_state=original_step(state,action)
        # Peer 0 has left at step 4; at the next radio tick peer 1 can only
        # learn from the informed base, whose message was cached at step 2.
        return next_state._replace(target_known=jnp.array([next_state.step>=1,next_state.step>=5]))
    collect=make_collector(level(),environment=(reset,step,observations,None),knowledge_capture=True)
    batch=jax.device_get(collect(Model(),jax.random.split(jax.random.PRNGKey(7),1)))
    assert batch['knowledge_first_counts'][0,1]==1
    assert batch['knowledge_first_counts'][0,2]==1
    assert batch['knowledge_first_base_token_counts'][0,2]==1
    assert batch['knowledge_first_base_token_counts'].sum()==1
    assert batch['knowledge_unattributed_events'].sum()==0


# Probe registry and suite

@pytest.fixture
def schedule(tmp_path, monkeypatch):
    checkpoint = tmp_path / "training/checkpoints/ckpt_000123"
    checkpoint.mkdir(parents=True)
    monkeypatch.setattr(runner, "verify_policy", lambda config: None)
    return {"run": {"name": "example", "output_root": "geo_experiments/outputs", "timezone": "Europe/Berlin"},
            "policies": {"policy": {"run": str(checkpoint.parent.parent), "checkpoint": str(checkpoint)}},
            "execute": [{"policy": "policy", "probe": "target-location/shared-coordinates", "repeats": 3}]}


@pytest.fixture
def config(schedule):
    return suite.execution_plan(schedule)[0][0]["config"]


@pytest.fixture(name="geometry")
def suite_geometry():
    return {"cells": np.array([[0, 0, 0], [1, 0, 0]]), "cell_size_m": np.array(5.),
            "map_offsets": np.array([0, 2]), "map_names": np.array(["map"]),
            "cell_map_id": np.array([0, 0]), "coordinate_values": np.array([[0, 0, 0], [1, 0, 0]]),
            "cell_coordinate_id": np.array([0, 1]), "distances": np.array([[0, 1], [1, 0]])}


def test_active_probes_and_future_own_cell_are_distinct():
    assert set(PROBES) == {"target-location/shared-coordinates", "path-predictor"}
    assert get_probe("target-location/per-map-decoder").definition().data["status"] == "archived"
    assert get_probe("target-location/joint-map-and-cell").definition().data["status"] == "archived"
    path = get_probe("path-predictor").definition().data
    own = get_probe("own-cell/cell-mask").definition().data
    assert path["target"]["kind"] == "optimal_roadmap_path"
    assert set(path["inputs"]) == {"finder_gru", "finder_message", "all_knowing_after_info", "all_knowing_till_end", "base"}
    assert path["inputs"]["all_knowing_after_info"]["agents"] == "active_target_knowing"
    assert path["inputs"]["all_knowing_till_end"]["every_steps"] == 10
    assert path["output"]["size"] == 125 and path["output"]["cardinality"] == "multiple"
    assert path["output"]["rotations"] == "yaw_quarter_turns"
    assert path["target"]["endpoints"] == "base_to_target"
    assert own["target"]["kind"] == "agent_position"
    assert own["inputs"]["agent_gru"]["agents"] == "all_active"
    assert own["output"]["cardinality"] == "exactly_one"
    assert own["output"]["encoding"] == path["output"]["encoding"] == "binary_cell_mask"


@pytest.mark.parametrize("name", ["own-cell/cell-mask",
                                  "target-location/per-map-decoder", "target-location/joint-map-and-cell"])
def test_non_active_definitions_cannot_be_scheduled(schedule, name):
    schedule["execute"][0]["probe"] = name
    with pytest.raises(ValueError, match="cannot be scheduled"):
        suite.execution_plan(schedule)


def test_definition_snapshot_and_repetitions_are_frozen(schedule):
    first, _ = suite.execution_plan(schedule)
    schedule["execute"][0]["repeats"] = 5
    second, _ = suite.execution_plan(schedule)
    assert first[0]["config"]["probe"]["seeds"] == [0, 1, 2]
    assert second[0]["config"]["probe"]["seeds"] == list(range(5))
    assert first[0]["collection_key"] == second[0]["collection_key"]
    assert first[0]["config"]["definition"]["target"]["kind"] == "target_position"


def test_collections_depend_on_input_and_target_requests_but_not_mlp_width(config):
    original = runner.plan_config(config)
    wider = copy.deepcopy(config)
    wider["probe"]["hidden_sizes"] = [512]
    wider["definition"]["model"]["hidden_sizes"] = [512]
    assert runner.plan_config(wider) == original
    gru = copy.deepcopy(config)
    gru["definition"]["inputs"] = {"finder_gru": gru["definition"]["inputs"]["finder_gru"]}
    gru["audit"]["enabled"] = False
    assert runner.plan_config(gru) != original


def test_collector_returns_requested_features_and_checks_only_requested_events(config):
    batch = {"finder_gru": np.ones((2, 4)), "finder_message": np.ones((2, 2)),
             "base_message": np.ones((2, 2)), "finder_step": np.array([1, -1]),
             "base_step": np.array([-1, 2]), "complete_sample": np.array([False, False]),
             "target_position": np.zeros((2, 3)), "episode_id": np.array([10, 11])}
    request = resolve_inputs({"memory": config["definition"]["inputs"]["finder_gru"]})
    selected = project_event_inputs(batch, request)
    assert "memory" in selected
    assert not {"finder_gru", "finder_message", "base_message"} & selected.keys()
    np.testing.assert_array_equal(selected["complete_sample"], [True, False])
    np.testing.assert_array_equal(selected["episode_id"], [10, 11])
    paired = project_event_inputs(batch, input_request(config))
    np.testing.assert_array_equal(paired["complete_sample"], [False, False])


def test_unimplemented_agent_input_cannot_silently_become_finder_input():
    with pytest.raises(ValueError, match="planned or unavailable"):
        resolve_inputs({"agent_gru": {"kind": "gru", "agents": "all_active", "event": "each_step"}})


def test_episode_grouping_is_required_even_for_agent_samples():
    raw = copy.deepcopy(get_probe("own-cell/cell-mask").definition().data)
    raw["sampling"]["split_unit"] = "row"
    with pytest.raises(ValueError, match="same episode"):
        Definition(raw).validate()


def test_coordinate_target_and_encoder_preserve_map_coordinates(config, geometry):
    dataset = {"target_position": np.array([[6., 1., 1.], [1., 1., 1.]]),
               "map_id": np.array([0, 0]), "target_required_agents": np.array([2, 1])}
    labels = build_target(config, dataset, geometry)
    y, vocabulary = encode_output(config, labels, geometry)
    np.testing.assert_array_equal(y, [1, 0])
    np.testing.assert_array_equal(vocabulary, geometry["coordinate_values"])


def test_shared_collection_uses_requested_inputs_and_saves_exact_split_budgets(config, geometry, tmp_path):
    config["audit"]["enabled"] = False
    config["policy"]["map_source"] = "training"
    config["definition"]["inputs"] = {"finder_gru": config["definition"]["inputs"]["finder_gru"]}
    config["dataset"].update(train_samples=2, validation_samples=1, test_samples=1)
    rows = {"target_position": np.array([[1., 1., 1.], [6., 1., 1.]] * 2),
            "map_id": np.zeros(4, dtype=int), "target_required_agents": np.ones(4, dtype=int),
            "episode_id": np.arange(4), "finder_gru": np.ones((4, 3))}
    budgets = collector.map_budgets(config, 1)
    counts = [dict(name="map", attempted=4, accepted=4, delivered=4, successful=3)]
    arrays, metadata = collector.finish_collection([rows], [rows], counts, budgets, config,
                                                    tmp_path, geometry, time.monotonic())
    assert metadata["input_dimensions"] == {"finder_gru": 3}
    splits = [arrays[f"splits/{name}"] for name in ("train", "validation", "test")]
    assert [len(part) for part in splits] == [2, 1, 1]
    assert len(set(np.concatenate(splits))) == 4
    assert "dataset/base_message" not in arrays


def test_shared_fitter_uses_definition_inputs_and_encoded_output(config, geometry, tmp_path, monkeypatch):
    config["definition"]["inputs"] = {"finder_gru": config["definition"]["inputs"]["finder_gru"]}
    config["probe"]["seeds"] = [4, 9]
    arrays = {}
    for prefix, values in {"dataset": {"finder_gru": np.ones((3, 4))},
                           "labels": {"coordinate_cell": np.array([0, 1, 0])},
                           "splits": {"train": np.array([0]), "validation": np.array([1]), "test": np.array([2])},
                           "geometry": geometry}.items():
        artifacts.add_group(arrays, prefix, values)
    calls = []

    def train(task, source, seed, data, labels, splits, graph, settings, *, class_values):
        calls.append((task, source, seed))
        np.testing.assert_array_equal(labels[task], [0, 1, 0])
        np.testing.assert_array_equal(class_values, geometry["coordinate_values"])
        return {"seed": seed}, {"probabilities": np.array([[.8, .2]]), "predicted": np.array([0])}

    monkeypatch.setattr(fitting, "train_probe", train)
    fitted, jobs = fitting.fit_experiment(config, arrays, tmp_path)
    assert calls == [("target_cell", "finder_gru", 4), ("target_cell", "finder_gru", 9)]
    assert len(jobs) == 2
    assert "probes/target_cell/finder_gru/seed_9/predicted" in fitted


@pytest.mark.parametrize("repeats", [0, -1, True, 1.5])
def test_invalid_repetitions(schedule, repeats):
    schedule["execute"][0]["repeats"] = repeats
    with pytest.raises(ValueError, match="positive integer"):
        suite.execution_plan(schedule)


def test_cli_repeat_override(schedule, tmp_path):
    path = tmp_path / "suite.yaml"
    OmegaConf.save(OmegaConf.create(schedule), path)
    assert suite.load_settings(path, ["execute.0.repeats=5"])["execute"][0]["repeats"] == 5


def test_suite_resume_and_job_dispatch_do_not_reload_definitions(schedule, tmp_path, monkeypatch):
    plan, _ = suite.execution_plan(schedule)
    artifacts.update_run(tmp_path, suite_config=schedule, execution_plan=plan)
    monkeypatch.setattr(suite.sys, "argv", ["suite.py", f"resume={tmp_path}"])
    monkeypatch.setattr(suite, "execution_plan", lambda settings: pytest.fail("replanned frozen suite"))
    monkeypatch.setattr(type(get_probe("target-location/shared-coordinates")), "definition",
                        lambda self: pytest.fail("reloaded probe definition"))
    assert job_probe(plan[0]).adapter() is runner
    received = []
    monkeypatch.setattr(suite, "execute_suite", lambda settings, root, jobs: received.extend(jobs))
    suite.main()
    assert received == plan


def test_completed_jobs_are_skipped(schedule, tmp_path, monkeypatch):
    plan, _ = suite.execution_plan(schedule)
    calls = []
    monkeypatch.setattr(runner, "prepare", lambda *args: calls.append("collect"))

    def fit(config, folder, collection):
        calls.append("fit")
        folder.mkdir(parents=True)
        artifacts.update_run(folder, config=config, progress={"stage": "complete"})
        (folder / "report.html").write_text("<html>saved</html>")

    monkeypatch.setattr(runner, "fit_and_report", fit)
    root = tmp_path / "suite"
    root.mkdir()
    suite.execute_suite(schedule, root, plan)
    suite.execute_suite(schedule, root, plan)
    assert calls == ["collect", "fit"]


def test_historical_own_cell_remains_separate_and_readable():
    from geo_experiments.archive.legacy.run import legacy_config_path
    from geo_experiments.archive.target_location.config import experiment_axes, validate_config
    config = OmegaConf.to_container(OmegaConf.merge(OmegaConf.load(legacy_config_path("config.yaml")),
                                                   OmegaConf.load(legacy_config_path("own_cell.yaml"))), resolve=True)
    validate_config(config)
    assert experiment_axes(config) == (("own_cell",), ("agent_gru", "agent_message"))


def test_report_dispatch_uses_explicit_renderer_before_geometry(tmp_path, monkeypatch):
    from geo_experiments import report
    from geo_experiments.shared.reports import dispatch
    config = get_probe("target-location/shared-coordinates").settings()
    monkeypatch.setattr(report, "load_run", lambda folder: ({"config": config}, {}, False))
    selected = []
    monkeypatch.setattr(dispatch, "render", lambda folder, manifest, arrays, dpi=None: selected.append(config["report"]["renderer"]))
    report.generate_report(tmp_path)
    assert selected == ["location"]
