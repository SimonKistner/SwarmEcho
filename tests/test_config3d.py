from copy import deepcopy

import pytest
import yaml

from swarmecho.core.config3d import load_level_3d, load_level_3d_cli
from swarmecho.training.artifacts import (
    checkpoint_artifact_suffix,
    eval_checkpoint_replay_root,
    train_replay_root,
)


def test_baseline_level_is_strict_and_solvable():
    level = load_level_3d()
    assert level.name == "M00_no_maze_open_cuboid_3D"
    assert level.env.radar_bins == 8
    assert level.env.num_agents == 5
    assert level.map_names == ["M00_no_maze_open_cuboid"]
    assert level.training.total_timesteps == 16_384_000
    assert level.network.actor_memory
    assert level.evaluation.eval_parallel_envs == 8
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
            "training.total_timesteps=327680",
            "logging.run_name=inspector_smoke",
            "env.radar_bins=16",
        ]
    )

    assert level.training.total_timesteps == 327_680
    assert level.logging.run_name == "inspector_smoke"
    assert level.env.radar_bins == 16
    assert level.num_updates == 20


def test_3d_replays_use_the_maintained_artifact_hierarchy(tmp_path):
    level = load_level_3d()
    checkpoint = tmp_path / "checkpoints/ckpt_000050"

    assert train_replay_root(tmp_path) == tmp_path / "artifacts/train/replays"
    assert checkpoint_artifact_suffix(checkpoint, level) == "u000050_s00819k"
    assert eval_checkpoint_replay_root(tmp_path, checkpoint, level) == (
        tmp_path / "artifacts/eval/ckpt_u000050_s00819k/replays"
    )
