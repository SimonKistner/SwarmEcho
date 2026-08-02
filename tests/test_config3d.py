from copy import deepcopy

import pytest
import yaml

from swarmecho.core.config3d import load_level_3d


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
