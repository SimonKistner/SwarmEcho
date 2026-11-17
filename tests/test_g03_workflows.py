"""Small G03 CLU workflow checks sharing one real training update."""

import json
import math
from dataclasses import replace

import pytest

from swarmecho.core.config import load_level
from swarmecho.curriculum_config.maps.random_building_bank import generate_bank
from swarmecho.training.evaluate import _load_checkpoint_on_map
from swarmecho.training.artifacts import eval_checkpoint_artifact_root
from swarmecho.training.checkpoints import restore_model_checkpoint
from swarmecho.training.train import build_model, evaluate_model, train
from swarmecho.visualize.replay import load_replay


@pytest.fixture(scope="module")
def g03_run(tmp_path_factory):
    # Compile a tiny private pool to exercise the normal G03 map-bank path.
    level = load_level("G03_CLU_random_buildings", [
        "env.max_steps=6",
        "training.total_timesteps=12",
        "training.max_training_seconds=null",
        "training.num_envs=2",
        "training.num_maps=1",
        "training.num_steps=6",
        "training.num_epochs=1",
        "training.num_minibatches=1",
        "training.oom_recovery.enabled=false",
        "training.target_spawn_eligibility.samples_per_map=8",
        "training.geo_probe_auxiliary.sampling.snapshots_per_agent=1",
        "network.hidden_dim=16",
        "network.tarmac_sig_dim=4",
        "network.tarmac_val_dim=4",
        "inspection_evaluation.inspection_maps=1",
        "inspection_evaluation.eval_parallel_envs=2",
        "validation.num_maps=1",
        "validation.num_episodes=2",
        "validation.parallel_envs=2",
        "logging.wandb_mode=disabled",
    ])
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(generate_bank, "POOL_ROOT", tmp_path_factory.mktemp("g03_bank"))
        pool, _ = generate_bank.build_pool(level, seed=456789, count=2, workers=1)
    level = replace(level, training=replace(level.training, map_pool_path=str(pool)))
    run_dir = tmp_path_factory.mktemp("g03_clu")
    checkpoint, stats = train(level, output_dir=run_dir)
    return level, run_dir, checkpoint, stats


def test_training_saves_a_checkpoint_and_finite_loss(g03_run):
    level, run_dir, checkpoint, stats = g03_run
    assert checkpoint.is_dir()
    assert level.env.success_condition == "chain_held_and_cleand_up"
    history = json.loads((checkpoint / "step_history.json").read_text(encoding="utf-8"))
    assert history["update"] == 1
    assert history["total_steps"] == level.training.total_timesteps
    saved = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert math.isfinite(saved["total_loss"])
    assert saved["total_loss"] == stats["total_loss"]


def test_checkpoint_evaluation_uses_the_frozen_inspection_map(g03_run):
    level, run_dir, checkpoint, _ = g03_run
    saved_map = run_dir / "random_eval_maps" / "random_eval_0000.yaml"
    inspection = _load_checkpoint_on_map(run_dir, str(saved_map), [])
    assert inspection.map_paths == (saved_map,)
    restored = build_model(level)
    restore_model_checkpoint(restored, checkpoint)
    states, rewards = evaluate_model(restored, inspection, max_steps=1)
    assert len(states) == 2
    assert rewards.shape == (2, level.env.num_agents, 1)

    artifact_root = eval_checkpoint_artifact_root(run_dir, checkpoint, level)
    replays = list(artifact_root.glob("eval_*/replays/*.json"))
    assert len(replays) == 1
    metadata, arrays = load_replay(replays[0])
    assert metadata["artifact_scope"] == "eval"
    assert metadata["checkpoint"] == str(checkpoint)
    assert metadata["success_condition"] == "chain_held"
    assert arrays["position"].shape[-1] == 3


def test_validation_writes_results_on_separate_maps_and_freezes_cases(g03_run):
    level, run_dir, checkpoint, _ = g03_run
    validation = run_dir / "validation"
    result = json.loads((validation / "results_u000001.json").read_text(encoding="utf-8"))
    assert result["success_condition"] == "chain_held"
    assert result["metrics"]["episodes"] == level.validation.num_episodes
    assert len(result["maps"]) == level.validation.num_maps
    selection = json.loads((checkpoint / "adaptive_map_selection.json").read_text(encoding="utf-8"))
    assert selection["kind"] == "bank"
    assert {entry["id"] for entry in selection["maps"]}.isdisjoint(
        entry["id"] for entry in selection["evaluation_maps"]
    )
    assert (checkpoint.parent / "best_val_checkpoint").is_dir()
    assert (checkpoint / "validation_cases.npz").read_bytes() == (validation / "cases.npz").read_bytes()
