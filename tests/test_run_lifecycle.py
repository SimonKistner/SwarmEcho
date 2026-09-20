import json
from dataclasses import replace

from omegaconf import OmegaConf

from swarmecho.core.config import load_level
from swarmecho.training.run_lifecycle import (
    resolve_resume_state,
    resolve_run_layout,
    schedule_due,
)


def test_run_layout_is_identical_for_dictconfig_and_3d_dataclasses(tmp_path):
    level = load_level()
    logging = replace(level.logging, run_name="shared", log_dir=str(tmp_path))
    evaluation = replace(level.evaluation, checkpoint_dir=None)
    typed = resolve_run_layout(logging, evaluation, media_dir="replays")
    omega = resolve_run_layout(
        OmegaConf.create({"run_name": "shared", "log_dir": str(tmp_path)}),
        OmegaConf.create({"checkpoint_dir": None}),
        media_dir="vids",
    )

    assert typed.run_dir == omega.run_dir == tmp_path / "shared"
    assert typed.checkpoint_dir == omega.checkpoint_dir == tmp_path / "shared/checkpoints"
    assert typed.train_media_dir == tmp_path / "shared/artifacts/train/replays"
    assert omega.train_media_dir == tmp_path / "shared/artifacts/train/vids"


def test_resume_history_and_schedules_are_dimension_agnostic(tmp_path):
    level = load_level()
    checkpoint = tmp_path / "run/checkpoints/ckpt_000020"
    checkpoint.mkdir(parents=True)
    (checkpoint / "step_history.json").write_text(
        json.dumps({"total_steps": 8_000_000, "history": [{"run_name": "run", "steps": 8_000_000}]})
    )
    training = replace(level.training, checkpoint_path=str(checkpoint), ckpt_loading_mode="resume")
    resume = resolve_resume_state(training, "run", 4000, 100)

    assert resume.start_update == 20
    assert resume.step_offset == 0
    assert resume.prior_history == []
    assert schedule_due(21, 20, 1)
    assert not schedule_due(20, 20, 1)


def test_init_restores_weights_without_inheriting_parent_timeline(tmp_path):
    level = load_level()
    checkpoint = tmp_path / "parent/checkpoints/ckpt_001250"
    checkpoint.mkdir(parents=True)
    (checkpoint / "step_history.json").write_text(
        json.dumps({"total_steps": 500_000_000, "history": [{"run_name": "parent", "steps": 500_000_000}]})
    )
    training = replace(level.training, checkpoint_path=str(checkpoint), ckpt_loading_mode="init")

    resume = resolve_resume_state(training, "tall", 4000, 100)

    assert resume.start_update == 0
    assert resume.step_offset == 0
    assert resume.prior_history == []
