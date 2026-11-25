"""G03 curriculum and path-grid contracts without environment rollout."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from swarmecho.analysis.path_targets import path_cell_offsets, path_grid_shape, rasterize
from swarmecho.core.config import LEVEL_DIR, TrainingConfig, ValidationConfig, load_level
from swarmecho.training.checkpoints import validate_checkpoint_contract
from swarmecho.training.run_lifecycle import schedule_due
from swarmecho.training.target_curriculum import ForwardTargetCurriculum


def test_forward_curriculum_uses_validation_group_and_restores_progress(tmp_path):
    curriculum = ForwardTargetCurriculum(7)
    assert curriculum.reset_kwargs() == {"target_agent_budget": 2, "target_min_agents": 1}
    assert curriculum.observe_curriculum("inspection", [2], [True], [1], 10) == {}

    curriculum.observe_curriculum("validation", [2] * 4, [True, True, False, False], [1] * 4, 10)
    assert curriculum.group == 2
    curriculum.observe_curriculum("validation", [2] * 4, [True, True, True, False], [1] * 4, 10)
    assert curriculum.group == 3

    curriculum.save(tmp_path)
    path = tmp_path / "target_curriculum_state.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["config"].pop("start_agents")  # Older forward state used a fixed implicit starting budget.
    path.write_text(json.dumps(saved), encoding="utf-8")
    continued = ForwardTargetCurriculum(7)
    assert continued.restore(tmp_path)
    assert continued.group == 3


def test_authored_validation_can_advance_past_empty_groups():
    curriculum = ForwardTargetCurriculum(7, skip_empty_forward_groups=True)
    curriculum.observe_curriculum("validation", [1, 5], [True, True], [1, 1], 10)
    assert curriculum.group == 5


def test_g03_probe_grid_preserves_resolution_and_order():
    g03 = SimpleNamespace(interior_cells=np.ones((3, 3, 3), dtype=bool))
    wide = SimpleNamespace(interior_cells=np.ones((2, 4, 3), dtype=bool))
    shape = path_grid_shape([g03])
    assert shape == (5, 5, 5)
    assert path_grid_shape([g03, wide]) == (7, 7, 5)
    offsets = path_cell_offsets(shape)
    np.testing.assert_array_equal(offsets[[0, 1, 5, 25]],
                                  [[-2, -2, -2], [-1, -2, -2], [-2, -1, -2], [-2, -2, -1]])
    mask = rasterize(np.array([[0., 0., 0.], [5., 0., 0.]]), 5., grid_shape=shape)
    assert mask.shape == (125,)
    np.testing.assert_array_equal(offsets[mask], [[0, 0, 0], [1, 0, 0]])


def test_uniform_default_and_update_schedule():
    assert not TrainingConfig().forward_target_curriculum.enabled
    assert ValidationConfig().mode is None
    assert [update for update in range(1, 304) if schedule_due(update, 100, 1)] == [101, 201, 301]


def test_g03_preset_loads_current_training_and_validation_mechanisms():
    preset = LEVEL_DIR / "G" / "G03_CLU_random_buildings.yaml"
    level = load_level(preset, config_data=yaml.safe_load(preset.read_text(encoding="utf-8")))

    assert level.training.checkpoint_path is None
    assert level.training.map_pool_path is None
    assert level.training.forward_target_curriculum.enabled
    assert level.training.adaptive_target_spawn.plateau_warning_enabled
    assert level.training.geo_probe_auxiliary.lambda_target == 0.3
    assert level.env.num_agents == 7 and level.env.observe_base_population
    assert level.training.num_maps == 3600 and level.validation.num_maps == 400
    assert level.validation.enabled and level.validation.mode is None
    assert path_grid_shape([level.building]) == (5, 5, 5)


def test_checkpoint_with_removed_vote_input_reports_incompatibility(tmp_path):
    current = {"format": 1, "adaptive_size": True, "recruitment_enabled": False,
               "obs_dim": 140, "vote_observation_slot": False,
               "wall_sensing_ray_max_distance": 5., "value_normalization": "running"}
    model = SimpleNamespace(checkpoint_contract=tuple(current.items()))
    path = tmp_path / "training_contract.json"
    path.write_text(json.dumps(current), encoding="utf-8")
    validate_checkpoint_contract(model, tmp_path)

    old = {**current, "obs_dim": 141}
    old.pop("vote_observation_slot")
    path.write_text(json.dumps(old), encoding="utf-8")
    with pytest.raises(ValueError, match="retired vote observation input"):
        validate_checkpoint_contract(model, tmp_path)


def _preset_data(name="G03_CLU_random_buildings"):
    source = LEVEL_DIR / "G" / (name + ".yaml")
    return source, yaml.safe_load(source.read_text(encoding="utf-8"))


def _historical_config(data):
    from copy import deepcopy

    data = deepcopy(data)
    data["env"]["radar_mode"] = "legacy"
    data["adapt_size"] = data.pop("decommissioning")
    for current, old in (("base_reporter_reward", "finder_reward"),
                         ("target_delivery_reward", "target_found_reward"),
                         ("closed_gap_reward_per_meter", "gap_progress_reward_per_meter")):
        data["reward"][old] = data["reward"].pop(current)
    data["reward"].update(decommission_reward=0, useless_agent_at_success_reward=-3000,
                          useless_agent_at_success_all_agents_reward=-750,
                          useless_agent_at_success_all_agents_reward_cap=-2000)
    adaptive = data["training"]["adaptive_target_spawn"]
    if "initial_target_agent_budget" in adaptive:
        adaptive["curriculum_start_agents"] = adaptive.pop("initial_target_agent_budget")
    inspection = data["inspection_evaluation"]
    inspection["eval_parallel_envs"] = inspection.pop("episodes_per_map")
    if inspection.get("inspection_maps") is not None:
        inspection["random_eval_envs"] = inspection.pop("inspection_maps")
    gate = data["validation"].pop("early_stopping")
    for current, old in (("enabled", "early_exit"), ("success_rate", "early_exit_success_rate"),
                         ("final_success_rate", "final_early_exit_success_rate"),
                         ("min_success_length_reduction", "early_exit_min_success_length_reduction"),
                         ("hold_evals", "early_exit_hold_evals")):
        if current in gate:
            inspection[old] = gate[current]
    return data


@pytest.mark.parametrize("name", ["G01_static", "G01_CLU_static", "G02_static", "G03_CLU_random_buildings"])
def test_renamed_presets_and_historical_configs_resolve_identically(name):
    from copy import deepcopy
    from dataclasses import asdict

    source, data = _preset_data(name)
    historical = _historical_config(data)
    unchanged = deepcopy(historical)
    current = load_level(source, config_data=data)
    previous = load_level(source, config_data=historical)
    assert historical == unchanged
    for field in ("env", "reward", "training", "validation", "evaluation"):
        assert asdict(getattr(current, field)) == asdict(getattr(previous, field))
    assert current.env.radar_mode == "max_proximity"
    assert current.env.decommissioning.enabled


def test_historical_override_names_move_before_config_merge():
    from swarmecho.core.config import normalize_renamed_settings

    normalized = normalize_renamed_settings({"evaluation": {
        "eval_parallel_envs": 12, "random_eval_envs": 2,
        "early_exit": False, "early_exit_success_rate": .9}})
    assert normalized["evaluation"] == {"episodes_per_map": 12, "inspection_maps": 2}
    assert normalized["validation"]["early_stopping"] == {"enabled": False, "success_rate": .9}
    assert normalize_renamed_settings({"env": {"adapt_size": {"enabled": False}}}) == {
        "env": {}, "decommissioning": {"enabled": False}}
    with pytest.raises(ValueError, match="Conflicting settings"):
        normalize_renamed_settings({"evaluation": {"episodes_per_map": 12, "eval_parallel_envs": 4}})


def test_saved_generated_run_metadata_is_readable():
    source, data = _preset_data()
    data.update(name="G03_CLU_random_buildings", building="random_buildings")
    data["env"]["map_names"] = ["random_buildings"]
    level = load_level(source, config_data=_historical_config(data))
    assert level.map_names == ["random_buildings"]
    assert level.env.radar_mode == "max_proximity"


def test_removed_reward_fields_are_absent_from_the_live_schema():
    from swarmecho.core.config import RewardConfig

    removed = {"decommission_reward", "useless_agent_at_success_reward",
               "useless_agent_at_success_all_agents_reward", "useless_agent_at_success_all_agents_reward_cap"}
    assert removed.isdisjoint(RewardConfig.__dataclass_fields__)
    _, data = _preset_data()
    assert removed.isdisjoint(data["reward"])
    assert data["reward"]["non_contributor_decommission_reward"] == 3000


def test_renames_preserve_the_existing_g03_validation_identity(tmp_path):
    from dataclasses import replace
    from pathlib import Path
    from swarmecho.training.validation import ValidationRunner

    source, data = _preset_data()
    level = load_level(source, config_data=data)
    paths = (Path("src/swarmecho/curriculum_config/maps/custom_maps/open_cuboid.yaml"),)
    level = replace(level, validation=replace(level.validation, num_maps=1), map_paths=paths)
    runner = ValidationRunner(level, tmp_path, paths)
    assert runner.identity == "ed973247a7ada1b01bc3e282439aece7bf0e61a37c270ecc26ce82be76e872bf"
    runner.level = replace(level, validation=replace(level.validation,
        early_stopping=replace(level.validation.early_stopping, hold_evals=2)))
    assert not runner.observe_gate({"success_rate": .995})
    assert runner.observe_gate({"success_rate": .995})
    assert not runner.observe_gate({"success_rate": .5})


def test_checkpoint_reader_accepts_historical_radar_and_decommissioning_names(tmp_path):
    current = {"format": 1, "decommissioning_enabled": True, "recruitment_enabled": False,
               "obs_dim": 140, "vote_observation_slot": False,
               "radar_mode": "max_proximity", "radar_layout": "max_proximity_interleaved",
               "wall_sensing_ray_max_distance": 8., "value_normalization": "running"}
    previous = {**current, "adaptive_size": current["decommissioning_enabled"],
                "radar_mode": "legacy", "radar_layout": "legacy_interleaved"}
    previous.pop("decommissioning_enabled")
    (tmp_path / "training_contract.json").write_text(json.dumps(previous), encoding="utf-8")
    validate_checkpoint_contract(SimpleNamespace(checkpoint_contract=tuple(current.items())), tmp_path)


def test_replay_reader_translates_historical_names_without_changing_values(tmp_path):
    from swarmecho.visualize.replay import REPLAY_FORMAT, load_replay

    np.savez(tmp_path / "episode.npz", position=np.zeros((1, 2, 3)), fully_connected=np.array([True]))
    metadata = {"format": REPLAY_FORMAT, "data_file": "episode.npz", "frames": 1,
                "adapt_size": {"enabled": True}, "fields": ["position", "fully_connected"],
                "reward_component_names": ["finder_reward", "target_found_reward", "decommission_reward"]}
    source = tmp_path / "episode.json"
    source.write_text(json.dumps(metadata), encoding="utf-8")
    manifest, arrays = load_replay(source)
    assert manifest["decommissioning"] == {"enabled": True}
    assert "adapt_size" not in manifest and "fully_connected" not in arrays
    assert manifest["reward_component_names"] == [
        "base_reporter_reward", "target_delivery_reward", "decommission_reward"]
    np.testing.assert_array_equal(arrays["base_target_connected"], [True])


def test_frozen_eligibility_migrates_only_matching_checkpoint_settings(tmp_path):
    from dataclasses import replace
    from swarmecho.training.target_spawn_eligibility import TargetSpawnEligibility

    source, data = _preset_data()
    level = load_level(source, config_data=data)
    support = TargetSpawnEligibility(level)
    for histogram in support.histograms:
        histogram[0, 1] = support.config.samples_per_map
    checkpoint = tmp_path / "run" / "checkpoints" / "ckpt_000001"
    checkpoint.mkdir(parents=True)
    (checkpoint.parent.parent / "config.yaml").write_text(yaml.safe_dump(_historical_config(data)), encoding="utf-8")
    support.save(checkpoint)
    path = checkpoint / "target_spawn_eligibility.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["identity"] = "before-source-renames"
    for row in saved["maps"]:
        row["identity"] = saved["identity"]
    path.write_text(json.dumps(saved), encoding="utf-8")
    restored = TargetSpawnEligibility(level)
    restored.restore(path)
    np.testing.assert_array_equal(restored.histograms[0], support.histograms[0])

    changed = replace(level, env=replace(level.env, comm_radius=level.env.comm_radius + 1))
    with pytest.raises(ValueError, match="Frozen target-cell eligibility settings changed"):
        TargetSpawnEligibility(changed).restore(path)
