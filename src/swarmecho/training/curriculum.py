"""Sequential checkpoint-inheriting curriculum entry point for 3D."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys
import time

from swarmecho.core.config import load_level_cli
from swarmecho.training.orchestration import (
    checkpoint_total_steps,
    explicit_override,
    split_levels,
)
from swarmecho.training.train import train


def main() -> None:
    """Train the requested 3D levels in order, handing off final weights."""
    levels, shared_args = split_levels(
        sys.argv[1:], "M00_no_maze_open_cuboid_3D"
    )
    base_level = load_level_cli(shared_args)
    base_run_name = base_level.logging.run_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if base_run_name:
        curriculum_id = (
            f"{base_run_name}_{timestamp}"
            if base_level.logging.use_timestamp_postfix
            else str(base_run_name)
        )
    else:
        curriculum_id = f"curriculum_{timestamp}"

    # Keep each curriculum stage in the same flat run layout as an ordinary
    # 3D run.  The prefix preserves curriculum provenance without requiring
    # consumers such as the inspector to understand a special nested root.
    output_root = Path(base_level.logging.log_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    explicit_group = explicit_override(shared_args, "logging.wandb_group")
    group_name = explicit_group or curriculum_id

    last_checkpoint = base_level.training.checkpoint_path
    initial_checkpoint_mode = str(base_level.training.ckpt_loading_mode).lower()
    if initial_checkpoint_mode not in {"resume", "branch", "init"}:
        raise ValueError(
            "training.ckpt_loading_mode must be resume, branch, or init."
        )
    cumulative_steps = checkpoint_total_steps(
        last_checkpoint,
        num_envs=base_level.training.num_envs,
        num_steps=base_level.training.num_steps,
    )

    print("\n" + "=" * 60)
    print("  SwarmEcho 3D curriculum")
    print(f"  output : {output_root}")
    print(f"  levels : {', '.join(levels)}")
    print("=" * 60)

    for index, level_name in enumerate(levels, start=1):
        level = load_level_cli([f"level={level_name}", *shared_args])
        level_short = level.name.split("_", 1)[0]
        stage_name = f"curr_{curriculum_id}_{level_short}"
        training = level.training
        if last_checkpoint:
            # An explicitly supplied initial checkpoint may either resume the
            # interrupted stage, branch with cumulative history, or initialize
            # a clean new stage from weights only. Subsequent stages always
            # branch from the preceding stage's final checkpoint.
            handoff_mode = (
                initial_checkpoint_mode if index == 1 else "branch"
            )
            handoff_overrides = {
                "checkpoint_path": str(last_checkpoint),
                "ckpt_loading_mode": handoff_mode,
            }
            if handoff_mode == "branch":
                handoff_overrides["checkpoint_step_offset"] = cumulative_steps
            training = replace(
                training,
                **handoff_overrides,
            )
            print(f"  [curriculum] loading weights from: {last_checkpoint}")
            print(f"  [curriculum] handoff mode: {handoff_mode}")
            print(f"  [curriculum] cumulative steps: {cumulative_steps:,}")

        logging = replace(
            level.logging,
            log_dir=str(output_root),
            run_name=stage_name,
            wandb_group=(
                group_name if level.logging.wandb_mode != "disabled" else explicit_group
            ),
        )
        level = replace(level, training=training, logging=logging)

        print(f"\n  [LEVEL {index}/{len(levels)}] {level.name}")
        final_checkpoint, _ = train(level)
        if index < len(levels) and final_checkpoint is None:
            raise RuntimeError(
                f"3D curriculum stage {level.name} produced no final checkpoint; "
                "set evaluation.save_model=true for checkpoint handoff."
            )
        if final_checkpoint is not None:
            last_checkpoint = final_checkpoint
            cumulative_steps = checkpoint_total_steps(
                final_checkpoint,
                num_envs=level.training.num_envs,
                num_steps=level.training.num_steps,
            )
            print(f"  [curriculum] stage complete: {final_checkpoint}")
        time.sleep(2)

    print(f"\n3D curriculum complete: {output_root}")


if __name__ == "__main__":
    main()
