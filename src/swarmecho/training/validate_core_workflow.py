"""Run the smallest maintained end-to-end training workflow.

VALIDATION COVERAGE GAPS
------------------------
This is deliberately a workflow validator, not a behavioral or numerical test.
It covers configuration loading, environment/model construction, one recurrent
TarMAC rollout, one PPO update, parallel evaluation, evaluation CSV/heatmap
artifacts, early exit, checkpoint saving, checkpoint restoration, and a
checkpoint-scoped evaluation plus video render outside the training runner.

It does NOT validate:

- learning quality, numerical equivalence, determinism, performance, GPU
  utilization, large-batch memory pressure, or long-run resource cleanup;
- curriculum/multi-train stage handoff, W&B online/offline behavior, training
  resume/branch accounting, interrupted-run recovery, or custom checkpoint
  directories;
- other maps, level families, feed-forward/no-memory configurations, alternate
  reward/observation/connectivity configurations, or targeted-position CLI
  evaluation;
- analysis dashboards, grid search, map/maze authoring and preview tools,
  documentation commands, or any future 3D runtime.

A successful run therefore proves only that this one maintained vertical slice
completed without an exception.  Any omitted surface may still be broken by a
past or future change.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from swarmecho.core.config import load_config, validate_config
from swarmecho.training.evaluate import evaluate_checkpoint
from swarmecho.training.runner import train


def remove_validation_run(run_dir: Path, cfg) -> None:
    """Delete only this validator's timestamped run below the output root."""
    output_root = Path(
        cfg.logging.get("log_dir", "outputs")
    ).absolute().resolve()
    resolved_run_dir = run_dir.absolute().resolve()

    if resolved_run_dir.parent != output_root:
        raise RuntimeError(
            "Refusing to remove validation output outside the configured "
            f"output root: {resolved_run_dir}"
        )
    if not resolved_run_dir.name.startswith("core_workflow_validation_"):
        raise RuntimeError(
            "Refusing to remove a run without the validation name prefix: "
            f"{resolved_run_dir}"
        )

    shutil.rmtree(resolved_run_dir)


def main() -> None:
    cfg = load_config(
        cli_overrides=True,
        overrides=["level=VALIDATION_core_workflow"],
    )
    validate_config(cfg)

    checkpoint = train(cfg)
    if not checkpoint:
        raise RuntimeError(
            "Validation training returned without saving a checkpoint."
        )

    checkpoint_path = Path(checkpoint).absolute()
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Validation checkpoint was not created: {checkpoint_path}"
        )

    run_dir = (
        checkpoint_path.parents[1]
        if checkpoint_path.parent.name == "checkpoints"
        else None
    )
    artifact_root = evaluate_checkpoint(
        cfg,
        checkpoint_path,
        run_dir=run_dir,
        save_video=True,
    )
    print("\nCore workflow validation passed.")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  evaluation artifacts: {artifact_root}")
    if run_dir is None:
        raise RuntimeError(
            "Validation passed, but its run directory could not be resolved "
            "safely for cleanup."
        )
    remove_validation_run(run_dir, cfg)
    print(f"  cleaned validation run: {run_dir}")


if __name__ == "__main__":
    main()
