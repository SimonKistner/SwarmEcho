from copy import deepcopy

import pytest
import yaml

from swarmecho.core.config import load_level_3d, load_level_3d_cli
from swarmecho.training.artifacts import (
    checkpoint_artifact_suffix,
    eval_checkpoint_replay_root,
    train_replay_root,
)


def test_baseline_level_is_strict_and_solvable():
    level = load_level_3d()
    assert level.name == "M00_no_maze_open_cuboid_3D"
    assert level.env.radar_bins == 8
    assert level.env.num_agents == 4
    assert level.env.max_steps == 700
    assert level.env.coverage_voxel_size == 2.5
    assert level.map_names == ["M00_no_maze_open_cuboid"]
    assert level.training.total_timesteps == 500_000_000
    assert level.training.num_envs == 4000
    assert level.training.num_steps == 100
    assert level.training.num_epochs == 4
    assert level.training.num_minibatches == 20
    assert not level.training.training_noise
    assert level.training.noise_level == level.evaluation.eval_action_noise_max
    assert level.network.actor_memory
    assert level.evaluation.eval_parallel_envs == 4000
    assert level.evaluation.eval_robustness_runs == 5
    assert level.evaluation.eval_action_noise_max == 0.011
    assert level.logging.wandb_mode == "online"
    assert level.ideal_chain_margin_m > 0


def test_tall_level_adds_two_solvable_spawn_layers():
    level = load_level_3d("M00_no_maze_open_cuboid_tall_3D")
    assert level.building.target_exclusion.shape == (4, 4, 6)
    assert level.building.target_exclusion[:, :, :2].all()
    assert not level.building.target_exclusion[:, :, 2:].any()
    assert level.env.num_agents == 4
    assert level.env.max_steps == 700
    assert level.ideal_chain_margin_m > 0


def test_unknown_3d_environment_parameter_is_rejected(tmp_path):
    source = "src/swarmecho/curriculum_config/levels/M00_no_maze_open_cuboid_3D.yaml"
    data = yaml.safe_load(open(source, encoding="utf-8"))
    data = deepcopy(data)
    data["env"]["typo_parameter"] = 1
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown env fields: typo_parameter"):
        load_level_3d(path)


def test_3d_cli_uses_the_same_dotlist_overrides_as_2d():
    level = load_level_3d_cli(
        [
            "level=M00_no_maze_open_cuboid_3D",
            "training.total_timesteps=400000",
            "logging.run_name=inspector_smoke",
            "env.radar_bins=16",
            "env.coverage_voxel_size=2.5",
        ]
    )

    assert level.training.total_timesteps == 400_000
    assert level.logging.run_name == "inspector_smoke"
    assert level.env.radar_bins == 16
    assert level.env.coverage_voxel_size == 2.5
    assert level.num_updates == 1


def test_3d_cli_can_enable_training_only_action_noise():
    level = load_level_3d_cli(
        [
            "level=M00_no_maze_open_cuboid_3D",
            "training.training_noise=true",
            "training.noise_level=0.02",
        ]
    )

    assert level.training.training_noise
    assert level.training.noise_level == 0.02
    # This setting is solely consumed by the training rollout; it must not
    # alter the standalone robust-evaluation configuration.
    assert level.evaluation.eval_action_noise_max == 0.011


def test_3d_replays_use_the_maintained_artifact_hierarchy(tmp_path):
    level = load_level_3d()
    checkpoint = tmp_path / "checkpoints/ckpt_000050"

    assert train_replay_root(tmp_path) == tmp_path / "artifacts/train/replays"
    assert checkpoint_artifact_suffix(checkpoint, level) == "u000050_s00020M"
    assert eval_checkpoint_replay_root(tmp_path, checkpoint, level) == (
        tmp_path / "artifacts/eval/ckpt_u000050_s00020M/replays"
    )
