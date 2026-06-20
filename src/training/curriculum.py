"""
training/curriculum.py
=======================
Orchestrates a sequence of training runs (L0 -> L1 -> ...) where each
stage inherits weights from the previous level's final checkpoint.

Level naming convention
-----------------------
Levels use a prefix string followed by an underscore and a description:
  A00_open_field, A01_warehouse, B01_agents_in_square_50m_comm_50m_base, ...

The curriculum list uses just the prefix (e.g. "A00", "B01") and the loader
resolves the matching YAML file automatically (glob: "{prefix}_*.yaml").

B-series auto-generation
------------------------
If a B-series level YAML (or its map) is missing, the runner automatically
calls generate_b_curriculum to create it before training starts.  The comm
radii are read from the global config so the generated file name matches.

Success-rate based transitions
-------------------------------
If a level YAML contains `curriculum.success_threshold: 0.95`, the trainer
will advance to the next level as soon as the 2000-episode sliding window
success rate reaches that threshold, rather than waiting for total_timesteps.
Omit the key (or set to 0.0) to use timesteps-only transitions.

Usage
-----
    uv run python training/curriculum.py logging.run_name=stage1_full
    uv run python training/curriculum.py logging.run_name=b_series \\
        logging.wandb_mode=online
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
from omegaconf import OmegaConf

from core.config import load_config, validate_config
from training.runner import train


# ---------------------------------------------------------------------------
# B-level auto-generation helper
# ---------------------------------------------------------------------------

def _ensure_b_level_exists(level_id: str, global_cfg) -> None:
    """
    If a B-series level file does not yet exist (matching the current comm
    radii) generate it via generate_b_curriculum.

    The number of agents is parsed from the numeric part of level_id ("B03" -> 3).
    The expected file name encodes the comm radii, so changing radii always
    triggers a fresh generation.
    """
    if not level_id.startswith("B"):
        return

    try:
        n_agents = int(level_id[1:])
    except ValueError:
        print(f"  [curriculum] Cannot parse agent count from '{level_id}', skipping auto-gen")
        return

    comm_r      = float(global_cfg.env.comm_radius)
    comm_r_base = float(global_cfg.env.get("comm_radius_base", comm_r))

    # Build the expected stem -- must match generate_b_curriculum.py logic
    comm_r_int      = int(comm_r)
    comm_r_base_int = int(comm_r_base)
    radius_suffix   = f"{comm_r_int}m_comm_{comm_r_base_int}m_base"
    expected_stem   = f"{level_id}_agents_in_square_{radius_suffix}"

    level_dir = Path(__file__).resolve().parents[1] / "curriculum_config" / "levels"
    if (level_dir / f"{expected_stem}.yaml").exists():
        return  # Already exists with the correct parameters

    print(
        f"  [curriculum] Level '{expected_stem}' not found -- auto-generating "
        f"(n_agents={n_agents}, comm_r={comm_r}, comm_r_base={comm_r_base})"
    )

    from curriculum_config.maps.scripts.generate_b_curriculum import generate_level
    generate_level(n_agents=n_agents, comm_r=comm_r, comm_r_base=comm_r_base)


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

        # Auto-generate B-series files if needed
        _ensure_b_level_exists(level_id, global_cfg)

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

        # 6. Read success threshold (None = timesteps-only transition)
        success_threshold = None
        if hasattr(cfg, "curriculum") and cfg.curriculum is not None:
            raw = cfg.curriculum.get("success_threshold", None)
            if raw is not None:
                success_threshold = float(raw)

        OmegaConf.set_readonly(cfg, True)
        validate_config(cfg)

        # 7. Run training
        final_ckpt_path = train(cfg, success_threshold=success_threshold)

        # 8. Update cumulative steps based on updates completed in this stage
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

        # 9. Pass checkpoint to the next stage
        last_checkpoint = final_ckpt_path
        print(f"  Stage {level_name} complete.")

        time.sleep(2)  # Brief cooldown for GPU/W&B sync

    print("\n" + "="*60)
    print(f"  CURRICULUM FINISHED SUCCESSFULLY!")
    print(f"  Results: {curriculum_root}")
    print("="*60)


if __name__ == "__main__":
    run_curriculum()
