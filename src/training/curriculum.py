"""
training/curriculum.py
=======================
Orchestrates a sequence of training runs (L0 -> L1 -> ...) where each
stage inherits weights from the previous level's final checkpoint.

Level naming convention
-----------------------
Levels use a prefix string followed by an underscore and a description:
  A00_open_field, A01_warehouse, B05a_open_square, ...

The curriculum list uses just the prefix (e.g. "A00", "B01") and the loader
resolves the matching YAML file automatically (glob: "{prefix}_*.yaml").

Evaluation-based transitions
----------------------------
If a level enables `evaluation.early_exit`, the trainer advances after the
parallel evaluation success rate reaches `evaluation.early_exit_threshold`.
Otherwise the level runs for its configured timestep budget.

Usage
-----
    uv run python training/curriculum.py logging.run_name=stage1_full
    uv run python training/curriculum.py logging.run_name=b_series \\
        logging.wandb_mode=online
"""

import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf

from core.config import load_config, validate_config
from training.runner import train


# ---------------------------------------------------------------------------
# Main curriculum runner
# ---------------------------------------------------------------------------

def run_curriculum():
    # Parse levels from sys.argv if present (e.g. levels=B02,B03)
    levels = ["B05a","B05b"]
    filtered_args = []
    for arg in sys.argv[1:]:
        if arg.startswith("levels="):
            val = arg.split("levels=", 1)[1]
            levels = [lvl.strip() for lvl in val.split(",")]
        else:
            filtered_args.append(arg)

    # Capture raw CLI overrides (forwarded verbatim to each stage)
    cli_overrides = OmegaConf.from_cli(filtered_args)
    # Load the base config once for global settings (comm_radius etc.)
    global_cfg = load_config(cli_overrides=True, overrides=filtered_args)

    # ---- Shared run identity ----------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_run_name = global_cfg.logging.get("run_name", None)
    use_ts = global_cfg.logging.get("use_timestamp_postfix", False)

    if not base_run_name:
        curriculum_id = f"curriculum_{timestamp}"
    else:
        curriculum_id = f"{base_run_name}_{timestamp}" if use_ts else base_run_name

    curriculum_root = Path("outputs") / "curriculum" / curriculum_id
    curriculum_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print(f"  STARTING CURRICULUM: {base_run_name}")
    print(f"  Root Dir: {curriculum_root}")
    print(f"  Levels:   {', '.join(levels)}")
    print("="*60)

    last_checkpoint = global_cfg.training.get("checkpoint_path", None)

    # Track cumulative step offset across curriculum stages
    cumulative_steps = 0
    if last_checkpoint:
        try:
            import re
            path_name = Path(last_checkpoint).name
            match = re.search(r'ckpt_(?:early_|final_)?(\d+)', path_name, re.IGNORECASE)
            if match:
                val = int(match.group(1))
                E = int(global_cfg.training.num_envs)
                T = int(global_cfg.training.num_steps)
                cumulative_steps = val * E * T
                print(f"  [curriculum] Parsed initial step offset from starting checkpoint: {cumulative_steps:,} steps")
        except Exception as e:
            print(f"  [curriculum] Failed to parse initial checkpoint steps: {e}")

    for idx, level_id in enumerate(levels):
        level_name = f"L{level_id}"
        print(f"\n  [LEVEL {idx+1}/{len(levels)}] Starting {level_name}...")

        # 1. Load baseline config defaults
        cfg = load_config(cli_overrides=False)

        # 2. Merge level-specific YAML
        level_dir = Path(__file__).resolve().parents[1] / "curriculum_config" / "levels"
        matches = (
            list(level_dir.glob(f"{level_id}_*.yaml"))
            + list(level_dir.glob(f"{level_id}_*.yml"))
            # Fallback: any file starting with the prefix
            + list(level_dir.glob(f"{level_id}*.yaml"))
            + list(level_dir.glob(f"{level_id}*.yml"))
        )
        # Deduplicate while preserving order
        seen: set = set()
        matches = [m for m in matches if not (m in seen or seen.add(m))]  # type: ignore

        if matches:
            level_cfg_path = matches[0]
            print(f"  Loading level YAML: {level_cfg_path.name}")
            level_overrides_yaml = OmegaConf.load(level_cfg_path)
            cfg = OmegaConf.merge(cfg, level_overrides_yaml)
        else:
            raise FileNotFoundError(
                f"No explicit curriculum level YAML found for prefix {level_id!r} in {level_dir}."
            )

        # 3. Apply CLI overrides (always win)
        OmegaConf.set_readonly(cfg, False)
        cfg = OmegaConf.merge(cfg, cli_overrides)

        # 4. Set checkpoint from previous stage
        cfg.training.checkpoint_path = last_checkpoint
        if last_checkpoint:
            cfg.training.ckpt_loading_mode = "branch"
            cfg.training.checkpoint_step_offset = cumulative_steps
            print(f"  [curriculum] Loading weights from: {last_checkpoint}")
            print(f"  [curriculum] Continuing with step offset: {cumulative_steps:,} steps")

        # 5. Wire logging into the curriculum directory
        cfg.logging.log_dir = str(curriculum_root)
        cfg.logging.run_name = f"{curriculum_id}_{level_id}"

        OmegaConf.set_readonly(cfg, True)
        validate_config(cfg)

        # 6. Run training. It returns after the timestep budget or configured
        # evaluation early exit; either result advances to the next level.
        final_ckpt_path = train(cfg)

        # 7. Update cumulative steps based on updates completed in this stage
        if final_ckpt_path:
            try:
                import re
                path_name = Path(final_ckpt_path).name
                match = re.search(r'ckpt_(?:early_|final_)?(\d+)', path_name, re.IGNORECASE)
                if match:
                    local_updates = int(match.group(1))
                    E = int(cfg.training.num_envs)
                    T = int(cfg.training.num_steps)
                    steps_taken = local_updates * E * T
                    cumulative_steps += steps_taken
                    print(f"  [curriculum] Stage completed {local_updates} updates ({steps_taken:,} steps). "
                          f"New cumulative step offset: {cumulative_steps:,}")
            except Exception as e:
                print(f"  [curriculum] Failed to parse completed stage steps from '{final_ckpt_path}': {e}")

        # 8. Pass checkpoint to the next stage
        last_checkpoint = final_ckpt_path
        print(f"  Stage {level_name} complete.")

        time.sleep(2)  # Brief cooldown for GPU/W&B sync

    print("\n" + "="*60)
    print(f"  CURRICULUM FINISHED SUCCESSFULLY!")
    print(f"  Results: {curriculum_root}")
    print("="*60)


if __name__ == "__main__":
    run_curriculum()
