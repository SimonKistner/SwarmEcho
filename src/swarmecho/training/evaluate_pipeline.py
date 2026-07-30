"""
training/evaluate_pipeline.py
=============================
Unified High-Throughput Parallel Evaluation and Failure Analysis Pipeline.

This pipeline performs the following in a single cohesive execution:
  1. Runs the configured parallel JAX evaluation batch to sweep for failures.
  2. Saves every target position, outcome, and target-to-base distance to CSV.
  3. Generates a failed chain targets heatmap overlay.
  4. Optionally generates a target-not-delivered heatmap.
  5. Optionally generates a target-not-visually-found heatmap.
  6. Optionally combines "Not Visually Found" (sky blue) and "Not Delivered" (dark blue) target groups
     into a single found-and-delivered heatmap with a top-padded title/legend layout.

Usage:
------
    # Run pipeline according to top-level toggle configurations:
    uv run swarmecho-evaluate-pipeline checkpoint=outputs/my_run/checkpoints/ckpt_001000
"""

import sys
import csv
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

from swarmecho.core.config import load_config, validate_config
from swarmecho.training.artifacts import (
    checkpoint_artifact_suffix,
    eval_checkpoint_artifact_root,
    save_eval_info_csv,
    write_manifest,
)
from swarmecho.training.evaluation import evaluate_parallel
from swarmecho.training.evaluation_artifacts import (
    load_map_data,
    render_and_save_failed_chain_heatmap,
    render_and_save_found_and_delivered_heatmap,
    render_and_save_not_found_heatmap,
)
from swarmecho.training.runtime import build_evaluation_runtime

# ==============================================================================
# Pipeline Artifact Output Toggles
# ==============================================================================
CREATE_CSV = None                    # None = use evaluation.save_eval_info_as_csv
CREATE_FAILED_CHAIN_HEATMAP = None   # None = use evaluation.eval_failed_chain_heatmap
CREATE_NOT_DELIVERED_HEATMAP = None  # None = use evaluation.eval_not_delivered_or_visually_found_heatmap
CREATE_NOT_VISUALLY_FOUND_HEATMAP = None # None = use evaluation.eval_not_delivered_or_visually_found_heatmap

# ==============================================================================
# Pipeline Configuration Constants
# ==============================================================================
# 1. Parallel simulation parameters
SEED = 42                   # Random seed for env reset and model initialization
SCALE = 8.0                 # Resolution scale (pixels per world-meter) for map image
SHOW_SPAWN_ZONES = False    # Set to False to disable target/base spawn zones overlay

# Resolved from evaluation.eval_not_deliv_not_visual_splitt_in_two in main().
COMBINE_FOUND_AND_DELIVERED_HEATMAPS = True

# ==============================================================================
# Pipeline Operations
# ==============================================================================
def setup_model_and_env(cfg, checkpoint_path):
    """Initialize the shared evaluation runtime and restore its checkpoint."""
    runtime = build_evaluation_runtime(cfg, checkpoint_path, rng_seed=SEED)
    print("Checkpoint loaded successfully ✓")

    environment = runtime.environment
    return (
        runtime.model,
        environment.env_step,
        environment.reset,
        environment.compute_obs,
        environment.compute_reward,
    )


def run_parallel_eval(model, cfg, env_step, reset, compute_obs, compute_reward, track_delivered=True, track_visual=True):
    """Run the public parallel evaluator and extract analysis arrays."""
    num_envs = int(cfg.evaluation.eval_parallel_envs)
    eval_key = jax.random.PRNGKey(SEED)

    start_time = time.time()
    result = evaluate_parallel(
        model, reset, env_step, compute_obs, compute_reward,
        cfg, eval_key, num_envs=num_envs
    )
    result.final_successes.block_until_ready()
    elapsed = time.time() - start_time
    print(f"Simulation completed in {elapsed:.2f} seconds.")

    # Calculate Stats
    num_success = int(jnp.sum(result.final_successes))
    num_fail = num_envs - num_success
    success_rate = (num_success / num_envs) * 100.0
    print(f"         Successes:      {num_success}/{num_envs} ({success_rate:.2f}%)")

    # Extract compact per-episode data once, after the parallel simulation.
    target_positions = np.array(result.final_state.target_pos)
    base_positions = np.array(result.final_state.base_pos)
    success_mask = np.array(result.final_successes, dtype=bool)

    failed_mask = ~success_mask
    failed_positions = target_positions[failed_mask]

    if track_delivered:
        num_delivered = int(jnp.sum(result.final_delivered))
        num_not_delivered = num_envs - num_delivered
        delivered_rate = (num_delivered / num_envs) * 100.0
        print(f"         Delivered:      {num_delivered}/{num_envs} ({delivered_rate:.2f}%)")
        not_delivered_mask = np.array(~result.final_delivered)
        not_delivered_positions = target_positions[not_delivered_mask]
    else:
        num_delivered = 0
        num_not_delivered = 0
        delivered_rate = 0.0
        not_delivered_positions = np.zeros((0, 2))

    if track_visual:
        num_visually_found = int(jnp.sum(result.final_visually_found))
        num_not_visually_found = num_envs - num_visually_found
        visually_found_rate = (num_visually_found / num_envs) * 100.0
        print(f"Results: Visually Found: {num_visually_found}/{num_envs} ({visually_found_rate:.2f}%)")
        not_visually_found_mask = np.array(~result.final_visually_found)
        not_visually_found_positions = target_positions[not_visually_found_mask]
    else:
        num_visually_found = 0
        num_not_visually_found = 0
        visually_found_rate = 0.0
        not_visually_found_positions = np.zeros((0, 2))

    return (
        failed_positions, not_delivered_positions, not_visually_found_positions,
        success_rate, delivered_rate, visually_found_rate,
        num_fail, num_not_delivered, num_not_visually_found,
        target_positions, base_positions, success_mask,
    )


def load_failures_from_csv(csv_path):
    """Load failed target coordinates from a comprehensive evaluation CSV."""
    failed_positions = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required_columns = {"x", "y", "success"}
        if not required_columns.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"Evaluation CSV must contain columns {sorted(required_columns)}: {csv_path}"
            )
        for row in reader:
            try:
                succeeded = row["success"].strip().lower() in {"1", "true", "yes"}
                if not succeeded:
                    failed_positions.append((float(row["x"]), float(row["y"])))
            except (AttributeError, TypeError, ValueError):
                continue
    return failed_positions


def find_eval_csvs(data_dir):
    """Return comprehensive evaluation CSVs."""
    data_dir = Path(data_dir)
    return list(data_dir.glob("eval_info_*.csv"))


def main():
    # Generate Run-Start Timestamp for consistent output file labeling
    run_timestamp = time.strftime("%Y_%m_%d_%H_%M")

    global CREATE_CSV, CREATE_FAILED_CHAIN_HEATMAP, CREATE_NOT_DELIVERED_HEATMAP, CREATE_NOT_VISUALLY_FOUND_HEATMAP, COMBINE_FOUND_AND_DELIVERED_HEATMAPS

    # Parse CLI Arguments
    args = sys.argv[1:]
    checkpoint_path = None
    overrides = []

    for arg in args:
        if arg.startswith("checkpoint="):
            checkpoint_path = Path(arg.split("=", 1)[1].replace("\\", "/"))
        elif arg.lower() in ["connectivity=true", "conn_matrix=true", "--connectivity", "--conn-matrix"]:
            overrides.append("visualize.render_conn_matrix=true")
        elif arg.lower() in ["connectivity=false", "conn_matrix=false", "--no-connectivity", "--no-conn-matrix"]:
            overrides.append("visualize.render_conn_matrix=false")
        elif arg.lower() in ["csv=true", "--csv"]:
            CREATE_CSV = True
        elif arg.lower() in ["csv=false", "--no-csv"]:
            CREATE_CSV = False
        elif arg.lower() in ["heatmap=true", "--heatmap"]:
            CREATE_FAILED_CHAIN_HEATMAP = True
            CREATE_NOT_DELIVERED_HEATMAP = True
            CREATE_NOT_VISUALLY_FOUND_HEATMAP = True
        elif arg.lower() in ["heatmap=false", "--no-heatmap"]:
            CREATE_FAILED_CHAIN_HEATMAP = False
            CREATE_NOT_DELIVERED_HEATMAP = False
            CREATE_NOT_VISUALLY_FOUND_HEATMAP = False
        else:
            overrides.append(arg)

    if checkpoint_path is None:
        print("ERROR: Must specify checkpoint=<path>")
        print("  e.g.  uv run swarmecho-evaluate-pipeline checkpoint=outputs/my_run/checkpoints/ckpt_001000")
        sys.exit(1)

    # Standard layout extraction: outputs/<run_name>/checkpoints/ckpt_XXXXXX
    if checkpoint_path.parent.name == "checkpoints":
        run_dir = checkpoint_path.parents[1]
    else:
        run_dir = Path("outputs")

    # Load Configuration
    run_config_path = run_dir / "config.yaml"
    run_config_loaded = run_config_path.exists()
    cfg = load_config(
        config_path = run_config_path if run_config_loaded else None,
        cli_overrides = True,
        overrides = overrides,
    )
    validate_config(cfg)
    num_envs = int(cfg.evaluation.eval_parallel_envs)
    if CREATE_CSV is None:
        CREATE_CSV = bool(cfg.evaluation.get("save_eval_info_as_csv", False))
    if CREATE_FAILED_CHAIN_HEATMAP is None:
        CREATE_FAILED_CHAIN_HEATMAP = bool(cfg.evaluation.get("eval_failed_chain_heatmap", False))
    if CREATE_NOT_DELIVERED_HEATMAP is None:
        CREATE_NOT_DELIVERED_HEATMAP = bool(
            cfg.evaluation.get("eval_not_delivered_or_visually_found_heatmap", False)
        )
    if CREATE_NOT_VISUALLY_FOUND_HEATMAP is None:
        CREATE_NOT_VISUALLY_FOUND_HEATMAP = bool(
            cfg.evaluation.get("eval_not_delivered_or_visually_found_heatmap", False)
        )
    COMBINE_FOUND_AND_DELIVERED_HEATMAPS = not bool(
        cfg.evaluation.get("eval_not_deliv_not_visual_splitt_in_two", False)
    )

    # Resolve checkpoint-scoped artifact directories.
    artifact_root = eval_checkpoint_artifact_root(run_dir, checkpoint_path, cfg)
    artifact_tag = checkpoint_artifact_suffix(checkpoint_path, cfg)
    chain_heatmaps_dir = artifact_root / "chain_heatmaps"
    found_heatmaps_dir = artifact_root / "found_heatmaps"
    data_dir = artifact_root / "data"
    manifest_dir = artifact_root / "manifests"

    # Map blueprint files
    map_name, map_data, map_def = load_map_data(cfg)

    # Initialize environment, model, and load checkpoint weights
    model, env_step, reset, compute_obs, compute_reward = setup_model_and_env(cfg, checkpoint_path)

    # ── Dependency Resolution & Execution Plan ────────────────────────────
    # Resolve CSV candidates early
    csv_candidates = find_eval_csvs(data_dir)

    # We need to run parallel JAX simulation sweep if:
    #   CREATE_CSV is requested (to get fresh evaluation data) OR we want to render either of the target-not-found heatmaps
    run_sweep = CREATE_CSV or CREATE_NOT_DELIVERED_HEATMAP or CREATE_NOT_VISUALLY_FOUND_HEATMAP

    # PREREQUISITE FALLBACK CHECK:
    # If the user wants to load from CSV (run_sweep = False), but no CSV actually exists:
    # We must force the simulation sweep to run to generate evaluation data.
    if not run_sweep:
        if not csv_candidates:
            print("\n[Prerequisite Warning] CREATE_CSV is False but no pre-existing CSV was found in output directory.")
            print("                       Forcing JAX parallel simulation sweep to generate coordinates.")
            run_sweep = True

    # Load or simulate coordinates
    failed_positions = None
    not_delivered_positions = None
    not_visually_found_positions = None
    success_rate = None
    delivered_rate = None
    visually_found_rate = None
    num_fail = None
    num_not_delivered = None
    num_not_visually_found = None

    if run_sweep:
        (failed_positions, not_delivered_positions, not_visually_found_positions,
         success_rate, delivered_rate, visually_found_rate,
         num_fail, num_not_delivered, num_not_visually_found,
         target_positions, base_positions, success_mask) = run_parallel_eval(
            model, cfg, env_step, reset, compute_obs, compute_reward,
            track_delivered=CREATE_NOT_DELIVERED_HEATMAP,
            track_visual=CREATE_NOT_VISUALLY_FOUND_HEATMAP
        )
        
        print("\n--- Phase 2: Processing Swept Coordinates ---")

        if CREATE_CSV:
            eval_info_path = save_eval_info_csv(
                data_dir / f"eval_info_{artifact_tag}.csv",
                target_positions=target_positions,
                base_positions=base_positions,
                successes=success_mask,
            )
            print(f"Saved {len(target_positions)} evaluation episode(s) to: {eval_info_path.name}")

        # Check if we should combine visually not found and not delivered heatmaps
        if COMBINE_FOUND_AND_DELIVERED_HEATMAPS:
            if CREATE_NOT_DELIVERED_HEATMAP or CREATE_NOT_VISUALLY_FOUND_HEATMAP:
                print("\n--- Phase 2a/b: Generating Found-and-Delivered Heatmap overlay ---")
                render_and_save_found_and_delivered_heatmap(
                    not_delivered_positions=not_delivered_positions,
                    not_visually_found_positions=not_visually_found_positions,
                    map_data=map_data,
                    map_def=map_def,
                    delivered_rate=delivered_rate,
                    visually_found_rate=visually_found_rate,
                    num_not_delivered=num_not_delivered,
                    num_not_visually_found=num_not_visually_found,
                    run_dir=run_dir,
                    video_dir=found_heatmaps_dir,
                    run_timestamp=run_timestamp,
                    total_episodes=num_envs,
                    data_dir=data_dir,
                    manifest_dir=manifest_dir,
                    artifact_stem=f"found_and_delivered_{artifact_tag}"
                )
        else:
            # 1. Visually Found Heatmap (Phase 2a)
            if CREATE_NOT_VISUALLY_FOUND_HEATMAP:
                print("\n--- Phase 2a: Generating Visually-Found Heatmap overlay ---")
                render_and_save_not_found_heatmap(
                    not_visually_found_positions, map_data, map_def, visually_found_rate, num_not_visually_found, 
                    run_dir, found_heatmaps_dir, run_timestamp, "found", "Not Visually Found",
                    total_episodes=num_envs, data_dir=data_dir, manifest_dir=manifest_dir, artifact_stem=f"found_{artifact_tag}"
                )

            # 2. Delivered Heatmap (Phase 2b)
            if CREATE_NOT_DELIVERED_HEATMAP:
                print("\n--- Phase 2b: Generating Delivered-To-Base Heatmap overlay ---")
                render_and_save_not_found_heatmap(
                    not_delivered_positions, map_data, map_def, delivered_rate, num_not_delivered, 
                    run_dir, found_heatmaps_dir, run_timestamp, "delivered", "Not Delivered",
                    total_episodes=num_envs, data_dir=data_dir, manifest_dir=manifest_dir, artifact_stem=f"delivered_{artifact_tag}"
                )

        # 3. Failed Chain Heatmap (Phase 2c)
        if CREATE_FAILED_CHAIN_HEATMAP:
            print("\n--- Phase 2c: Generating Failed Chain Heatmap overlay ---")
            _ = render_and_save_failed_chain_heatmap(
                failed_positions, map_data, map_def, success_rate, num_fail, run_dir, chain_heatmaps_dir, run_timestamp,
                save_png=True, total_episodes=num_envs,
                manifest_dir=manifest_dir, artifact_stem=f"failed_chain_{artifact_tag}"
            )
    else:
        # Load from the latest CSV
        csv_path = max(csv_candidates, key=lambda p: p.stat().st_mtime)
        print(f"\nLoading failed target positions from CSV: {csv_path.name}")
        failed_positions = np.array(load_failures_from_csv(csv_path))
        num_fail = len(failed_positions)

        if CREATE_FAILED_CHAIN_HEATMAP:
            print("\n--- Phase 2: Generating failed chain heatmap overlay from loaded CSV ---")
            _ = render_and_save_failed_chain_heatmap(
                failed_positions, map_data, map_def, success_rate=None, num_fail=num_fail, 
                run_dir=run_dir, video_dir=chain_heatmaps_dir, run_timestamp=run_timestamp,
                save_png=True, total_episodes=num_envs,
                manifest_dir=manifest_dir, artifact_stem=f"failed_chain_{artifact_tag}"
            )

        if CREATE_NOT_DELIVERED_HEATMAP:
            print("\n[Prerequisite Warning] Delivered-to-base heatmap cannot be generated when loading from static CSV.")
        if CREATE_NOT_VISUALLY_FOUND_HEATMAP:
            print("\n[Prerequisite Warning] Visually-found heatmap cannot be generated when loading from static CSV.")

    print("\nEvaluation pipeline complete.")


if __name__ == "__main__":
    main()
