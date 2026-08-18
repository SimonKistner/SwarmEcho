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
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

from swarmecho.core.config import load_config, validate_config
from swarmecho.training.artifacts import (
    checkpoint_artifact_suffix,
    eval_checkpoint_artifact_root,
    load_eval_info_csv,
    save_eval_info_csv,
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
CREATE_FAILED_CHAIN_HEATMAP = True
CREATE_NOT_DELIVERED_HEATMAP = True
CREATE_NOT_VISUALLY_FOUND_HEATMAP = True

# ==============================================================================
# Pipeline Configuration Constants
# ==============================================================================
# Parallel simulation parameters
SEED = 42                   # Random seed for env reset and model initialization

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


def run_parallel_eval(model, cfg, env_step, reset, compute_obs, compute_reward):
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
    success_rate = (num_success / num_envs) * 100.0
    print(f"         Successes:      {num_success}/{num_envs} ({success_rate:.2f}%)")

    # Extract compact per-episode data once, after the parallel simulation.
    target_positions = np.array(result.final_state.physics.target_pos)
    base_positions = np.array(result.final_state.physics.base_pos)
    success_mask = np.array(result.final_successes, dtype=bool)

    delivered_mask = np.array(result.final_delivered, dtype=bool)
    num_delivered = int(np.sum(delivered_mask))
    delivered_rate = (num_delivered / num_envs) * 100.0
    print(f"         Delivered:      {num_delivered}/{num_envs} ({delivered_rate:.2f}%)")

    visually_found_mask = np.array(result.final_visually_found, dtype=bool)
    num_visually_found = int(np.sum(visually_found_mask))
    visually_found_rate = (num_visually_found / num_envs) * 100.0
    print(f"Results: Visually Found: {num_visually_found}/{num_envs} ({visually_found_rate:.2f}%)")

    return (
        target_positions,
        base_positions,
        success_mask,
        delivered_mask,
        visually_found_mask,
    )


def main():
    # Generate Run-Start Timestamp for consistent output file labeling
    run_timestamp = time.strftime("%Y_%m_%d_%H_%M")

    global CREATE_FAILED_CHAIN_HEATMAP, CREATE_NOT_DELIVERED_HEATMAP, CREATE_NOT_VISUALLY_FOUND_HEATMAP, COMBINE_FOUND_AND_DELIVERED_HEATMAPS

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
        elif arg.lower() in ["heatmap=true", "--heatmap"]:
            # Retained as a compatibility no-op: standalone evaluation always
            # creates all heatmaps.
            pass
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
    _, map_data, map_def = load_map_data(cfg)

    # Initialize environment, model, and load checkpoint weights
    model, env_step, reset, compute_obs, compute_reward = setup_model_and_env(cfg, checkpoint_path)

    # ── Dependency Resolution & Execution Plan ────────────────────────────
    (
        target_positions,
        base_positions,
        success_mask,
        delivered_mask,
        visually_found_mask,
    ) = run_parallel_eval(
        model, cfg, env_step, reset, compute_obs, compute_reward,
    )

    eval_info_path = save_eval_info_csv(
        data_dir / f"eval_info_{artifact_tag}.csv",
        target_positions=target_positions,
        base_positions=base_positions,
        successes=success_mask,
        delivered=delivered_mask,
        visually_found=visually_found_mask,
    )
    print(f"Saved {len(target_positions)} evaluation episode(s) to: {eval_info_path.name}")

    # Heatmaps intentionally consume the just-written canonical record rather
    # than parallel in-memory tables, so their inputs remain inspectable.
    records = load_eval_info_csv(eval_info_path)
    target_positions = records["positions"]
    stages = records["stages"]
    total_episodes = len(stages)
    success_mask = stages == "chain_success"
    delivered_mask = np.isin(stages, ("found_and_delivered", "chain_success"))
    visually_found_mask = stages != "not_found"
    failed_positions = target_positions[~success_mask]
    not_delivered_positions = target_positions[~delivered_mask]
    not_visually_found_positions = target_positions[~visually_found_mask]
    visually_found_not_delivered_positions = target_positions[
        (~delivered_mask) & visually_found_mask
    ]
    success_rate = float(np.mean(success_mask) * 100.0)
    delivered_rate = float(np.mean(delivered_mask) * 100.0)
    visually_found_rate = float(np.mean(visually_found_mask) * 100.0)

    if COMBINE_FOUND_AND_DELIVERED_HEATMAPS:
        if CREATE_NOT_DELIVERED_HEATMAP or CREATE_NOT_VISUALLY_FOUND_HEATMAP:
            print("\n--- Generating Found-and-Delivered Heatmap overlay ---")
            render_and_save_found_and_delivered_heatmap(
                visually_found_not_delivered_positions=(
                    visually_found_not_delivered_positions
                ),
                not_visually_found_positions=not_visually_found_positions,
                map_data=map_data,
                map_def=map_def,
                delivered_rate=delivered_rate,
                visually_found_rate=visually_found_rate,
                num_not_delivered=len(not_delivered_positions),
                num_not_visually_found=len(not_visually_found_positions),
                run_dir=run_dir,
                video_dir=found_heatmaps_dir,
                run_timestamp=run_timestamp,
                total_episodes=total_episodes,
                manifest_dir=manifest_dir,
                artifact_stem=f"found_and_delivered_{artifact_tag}",
                source_csv=eval_info_path,
            )
    else:
        if CREATE_NOT_VISUALLY_FOUND_HEATMAP:
            print("\n--- Generating Visually-Found Heatmap overlay ---")
            render_and_save_not_found_heatmap(
                not_visually_found_positions, map_data, map_def,
                visually_found_rate, len(not_visually_found_positions),
                run_dir, found_heatmaps_dir, run_timestamp, "found",
                "Not Visually Found", total_episodes=total_episodes,
                manifest_dir=manifest_dir, artifact_stem=f"found_{artifact_tag}",
                source_csv=eval_info_path,
            )
        if CREATE_NOT_DELIVERED_HEATMAP:
            print("\n--- Generating Delivered-To-Base Heatmap overlay ---")
            render_and_save_not_found_heatmap(
                not_delivered_positions, map_data, map_def,
                delivered_rate, len(not_delivered_positions), run_dir,
                found_heatmaps_dir, run_timestamp, "delivered", "Not Delivered",
                total_episodes=total_episodes, manifest_dir=manifest_dir,
                artifact_stem=f"delivered_{artifact_tag}", source_csv=eval_info_path,
            )

    if CREATE_FAILED_CHAIN_HEATMAP:
        print("\n--- Generating Failed Chain Heatmap overlay ---")
        render_and_save_failed_chain_heatmap(
            failed_positions, map_data, map_def, success_rate,
            len(failed_positions), run_dir, chain_heatmaps_dir, run_timestamp,
            save_png=True, total_episodes=total_episodes,
            manifest_dir=manifest_dir, artifact_stem=f"failed_chain_{artifact_tag}",
            source_csv=eval_info_path,
        )

    print("\nEvaluation pipeline complete.")


if __name__ == "__main__":
    main()
