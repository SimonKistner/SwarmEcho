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
