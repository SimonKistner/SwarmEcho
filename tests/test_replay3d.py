import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from swarmecho.env.baseline3d import Baseline3DConfig, make_baseline_3d_fns
from swarmecho.env.buildings import load_building
from swarmecho.visualize.replay3d import load_replay, write_replay


def test_replay_round_trip_and_atomic_manifest(tmp_path):
    building = load_building(
        "src/swarmecho/curriculum_config/maps/M00_no_maze_open_cuboid.yaml"
    )
    cfg = Baseline3DConfig()
    reset, step, _, _ = make_baseline_3d_fns(building, cfg)
    state = reset(jax.random.PRNGKey(3))
    states = [state]
    for _ in range(3):
        state = step(state, jnp.zeros((cfg.num_agents, 3)))
        states.append(state)

    data_path, manifest_path = write_replay(
        tmp_path / "rollout",
        states,
        map_name="M00_no_maze_open_cuboid",
        dt=cfg.dt,
        reward_terms=np.zeros((4, cfg.num_agents, 2), dtype=np.float32),
    )
    assert data_path.exists()
    assert manifest_path.exists()
    assert not list(tmp_path.glob("*.tmp"))

    metadata, arrays = load_replay(manifest_path)
    assert metadata["frames"] == 4
    assert arrays["position"].shape == (4, cfg.num_agents, 3)
    assert arrays["coverage"].shape == (4, 4, 4, 4)
    assert arrays["reward_terms"].shape == (4, cfg.num_agents, 2)
    assert arrays["obstacle_min"].shape == (4, 0, 3)


def test_replay_rejects_unknown_format(tmp_path):
    manifest = tmp_path / "bad.json"
    manifest.write_text(json.dumps({"format": "unknown"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported replay format"):
        load_replay(manifest)
