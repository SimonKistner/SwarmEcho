"""Independent training, inspection and validation goals."""

from dataclasses import replace
from pathlib import Path
import json

import pytest

from swarmecho.core.config import load_level, resolve_evaluation_level, resolve_validation_level
from swarmecho.training.validation import ValidationRunner


CLEANUP = "chain_held_and_cleand_up"


def test_g03_cluster_trains_and_scores_full_mission():
    level = load_level("G03_CLU_random_buildings")
    assert level.env.success_condition == CLEANUP
    assert resolve_evaluation_level(level).env.success_condition == CLEANUP
    assert resolve_validation_level(level).env.success_condition == CLEANUP
    assert level.env.hold_chain_for == 50


@pytest.mark.parametrize("inspection,validation", [
    ("chain_held", None), (CLEANUP, "chain_held"), (None, None),
])
def test_goals_default_to_training_independently(inspection, validation):
    level = load_level("G03_CLU_random_buildings", [
        f"inspection_evaluation.success_condition={inspection or 'null'}",
        f"validation.success_condition={validation or 'null'}",
    ])
    assert resolve_evaluation_level(level).env.success_condition == (inspection or CLEANUP)
    assert resolve_validation_level(level).env.success_condition == (validation or CLEANUP)
    assert level.env.success_condition == CLEANUP


@pytest.mark.parametrize("overrides,match", [
    (["validation.success_condition=typo"], r"validation.success_condition"),
    (["validation.enabled=false", "validation.success_condition=typo"], r"validation.success_condition"),
    (["validation.success_condition=coverage"], "requires a target task"),
    (["validation.success_condition=discovery"], "Adaptive size requires"),
    (["adapt_size.enabled=false", "reward.priveleged_vote_reward.enabled=false",
      "reward.success_reward_as_hold_record=true",
      "validation.success_condition=delivery"], "chain-hold validation success condition"),
])
def test_invalid_validation_goals_are_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        load_level("G03_CLU_random_buildings", overrides)


@pytest.mark.parametrize("mode", ["held_out_maps", "uniform_full_task"])
def test_runner_uses_validation_goal_and_guards_saved_identity(tmp_path, mode):
    level = load_level("G03_CLU_random_buildings", ["validation.success_condition=chain_held"])
    paths = [Path("src/swarmecho/curriculum_config/maps/M00_no_maze_open_cuboid.yaml")]
    level = replace(level, map_paths=tuple(paths), validation=replace(level.validation, num_maps=1, mode=mode))
    runner = ValidationRunner(level, tmp_path, paths)
    assert runner.level.env.success_condition == "chain_held"
    assert level.env.success_condition == CLEANUP
    metadata = json.loads((tmp_path / "validation" / "metadata.json").read_text())
    assert metadata["success_condition"] == "chain_held"

    # An explicit goal equal to training retains the default task identity.
    inherited = replace(level, validation=replace(level.validation, success_condition=None))
    default_runner = ValidationRunner(inherited, tmp_path / "default", paths)
    explicit = replace(level, validation=replace(level.validation, success_condition=CLEANUP))
    explicit_runner = ValidationRunner(explicit, tmp_path / "explicit", paths)
    assert default_runner.identity == explicit_runner.identity
    assert default_runner.identity != runner.identity
    with pytest.raises(ValueError, match="Validation maps, stage or case settings changed"):
        ValidationRunner(inherited, tmp_path, paths)
