"""Composable probe definitions, shared acquisition and historical result compatibility."""
import copy
import time

import numpy as np
import pytest
from omegaconf import OmegaConf

from geo_experiments import suite
from geo_experiments.registry import PROBES, get_probe, job_probe
from geo_experiments.shared import artifacts, collector, fitting, runner
from geo_experiments.shared.definitions import Definition
from geo_experiments.shared.inputs import input_request, project_event_inputs, resolve_inputs
from geo_experiments.shared.targets import build_target, encode_output


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


@pytest.fixture
def geometry():
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
