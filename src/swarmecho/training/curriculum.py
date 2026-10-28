"""Sequential checkpoint-inheriting curriculum entry point for the environment."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sys
import time

from swarmecho.core.config import load_level_cli
from swarmecho.training.orchestration import (
    checkpoint_total_steps,
    explicit_override,
    split_levels,
)
from swarmecho.training.launcher import train


def train_agent_counts(level, *, output_dir=None, checkpoint_path=None):
    """Run <=k target budgets with spare slots bounded by env.num_agents."""
    settings = level.training.agent_count_curriculum
    maximum = level.env.num_agents
    start = settings.start_agents
    first_agents = settings.start_agents
    prefix = level.logging.run_name or f"agents_{level.name}_{datetime.now():%Y%m%d_%H%M%S}"
    if level.logging.run_name and level.logging.use_timestamp_postfix:
        prefix += f"_{datetime.now():%Y%m%d_%H%M%S}"
    parent = Path(output_dir or level.logging.log_dir).absolute()
    parent.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_path or level.training.checkpoint_path
    first_mode = level.training.ckpt_loading_mode
    resumed_num_envs = level.training.num_envs
    if checkpoint and first_mode == "resume":
        import yaml
        checkpoint_dir = Path(str(checkpoint).replace("\\", "/"))
        marker = checkpoint_dir / "recovery_ready.json"
        metadata = json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else {}
        source = (Path(metadata["run_dir"]) if metadata.get("run_dir") else checkpoint_dir.parents[1]) / "config.yaml"
        if not source.is_file():
            raise ValueError("Resuming an agent stage requires its saved config.yaml.")
        saved = yaml.safe_load(source.read_text(encoding="utf-8"))
        first_agents = int(saved["training"].get("agent_count_curriculum", {}).get("start_agents", first_agents))
        saved_additional = int(saved["training"].get("agent_count_curriculum", {}).get("additional_agents", 0))
        if saved_additional != settings.additional_agents:
            raise ValueError("Agent curriculum additional_agents changed; branch into a new run instead of resuming.")
        if level.training.oom_recovery.enabled:
            resumed_num_envs = min(resumed_num_envs, int(saved["training"]["num_envs"]))
        saved_budget = saved["env"].get("relay_agent_budget")
        start = int(saved["env"]["num_agents"] if saved_budget is None else saved_budget)
        saved_name = saved["logging"]["run_name"]
        suffix = f"_agents_{start}"
        if not saved_name.endswith(suffix) or not settings.start_agents <= start <= maximum:
            raise ValueError("Checkpoint does not identify a compatible agent curriculum stage.")
        prefix = saved_name[:-len(suffix)]
    journal = parent / f"{prefix}_agents_{first_agents}" / "agent_curriculum.json"
    legacy_journal = parent / f"{prefix}_agent_curriculum.json"
    source_journal = journal if journal.is_file() else legacy_journal
    record = {"level": level.name, "maximum_agents": maximum,
              "additional_agents": settings.additional_agents, "stages": []}
    if source_journal.is_file():
        record = json.loads(source_journal.read_text(encoding="utf-8"))
        if (record["maximum_agents"] != maximum or record["level"] != level.name
                or record.get("additional_agents", 0) != settings.additional_agents):
            raise ValueError("Agent curriculum level, maximum or additional_agents changed; use a new run name.")
        record["additional_agents"] = settings.additional_agents
    journal.parent.mkdir(parents=True, exist_ok=True)
    if source_journal == legacy_journal and legacy_journal.is_file():
        legacy_journal.rename(journal)
    final_metrics = {}
    effective_num_envs = min(resumed_num_envs, int(record.get("effective_num_envs", level.training.num_envs)))
    for agents in range(start, maximum + 1):
        deployable_agents = min(agents + settings.additional_agents, maximum)
        success_rate = level.evaluation.early_exit_success_rate
        if (agents == deployable_agents == maximum
                and level.evaluation.final_early_exit_success_rate is not None):
            success_rate = level.evaluation.final_early_exit_success_rate
        stage_name = f"{prefix}_agents_{agents}"
        adaptive = replace(level.env.adapt_size,
                           initial_agents=min(level.env.adapt_size.initial_agents, deployable_agents))
        training = replace(level.training,
                           num_envs=effective_num_envs,
                           agent_count_curriculum=replace(settings, enabled=False),
                           checkpoint_path=str(checkpoint) if checkpoint else None,
                           ckpt_loading_mode=first_mode if agents == start else "branch")
        if checkpoint and training.ckpt_loading_mode == "branch":
            training = replace(training, checkpoint_step_offset=checkpoint_total_steps(
                checkpoint, num_envs=training.num_envs, num_steps=training.num_steps))
        stage = replace(level, env=replace(level.env, num_agents=deployable_agents,
                            relay_agent_budget=agents if settings.additional_agents else None,
                            adapt_size=adaptive),
                        training=training,
                        evaluation=replace(level.evaluation, replay_targets_from_storey=None,
                            early_exit_success_rate=success_rate,
                            checkpoint_dir=(str(Path(level.evaluation.checkpoint_dir) / stage_name)
                                            if level.evaluation.checkpoint_dir else None)),
                        logging=replace(level.logging, log_dir=str(parent), run_name=stage_name,
                                        use_timestamp_postfix=False,
                                        wandb_group=level.logging.wandb_group or prefix))
        print(f"\n[AGENT CURRICULUM] Budget {agents}/{maximum}; {deployable_agents} deployable agents; "
              f"accepting constructed counts 1..{agents}", flush=True)
        checkpoint, final_metrics = train(stage)
        effective_num_envs = int(final_metrics.get("effective_num_envs", training.num_envs))
        record["effective_num_envs"] = effective_num_envs
        mastered = bool(final_metrics.get("agent_curriculum_mastered", False))
        item = {"agents": agents, "run_name": stage_name,
                "deployable_agents": deployable_agents,
                "checkpoint": str(checkpoint) if checkpoint else None, "mastered": mastered,
                "num_envs": effective_num_envs}
        record["stages"] = [row for row in record["stages"] if row["agents"] != agents] + [item]
        temporary = journal.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        temporary.replace(journal)
        if not mastered:
            print(f"[AGENT CURRICULUM] Stopped at {agents}: evaluation mastery was not reached. Checkpoint: {checkpoint}", flush=True)
            break
        if checkpoint is None:
            raise RuntimeError("Mastered agent stage produced no checkpoint.")
    return checkpoint, final_metrics


def main() -> None:
    """Train the requested levels in order, handing off final weights."""
    levels, shared_args = split_levels(
        sys.argv[1:], "M00_no_maze_open_cuboid"
    )
    feature_override = any(arg.lower() in {
        "training.dynamic_max_geo_distance.enabled=true",
        "training.agent_count_curriculum.enabled=true",
    } for arg in shared_args)
    # The default placeholder may be a coverage level, which cannot validate
    # target-filter overrides intended for the actual first requested level.
    base_level = load_level_cli([f"level={levels[0]}", *shared_args] if feature_override else shared_args)
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
    # run.  The prefix preserves curriculum provenance without requiring
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
    print("  SwarmEcho curriculum")
    print(f"  output : {output_root}")
    print(f"  levels : {', '.join(levels)}")
    print("=" * 60)

    effective_num_envs = None
    for index, level_name in enumerate(levels, start=1):
        level = load_level_cli([f"level={level_name}", *shared_args])
        level_short = level.name.split("_", 1)[0]
        stage_name = f"curr_{curriculum_id}_{level_short}"
        training = level.training
        if training.oom_recovery.enabled and effective_num_envs is not None:
            training = replace(training, num_envs=min(training.num_envs, effective_num_envs))
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
        final_checkpoint, metrics = train(level)
        effective_num_envs = int(metrics.get("effective_num_envs", training.num_envs))
        if index < len(levels) and final_checkpoint is None:
            raise RuntimeError(
                f"curriculum stage {level.name} produced no final checkpoint; "
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
        if level.training.agent_count_curriculum.enabled and not metrics.get("agent_curriculum_mastered", False):
            print("  [curriculum] agent stage did not reach mastery; remaining levels are not started.")
            break
        time.sleep(2)

    print(f"\ncurriculum complete: {output_root}")


if __name__ == "__main__":
    main()
