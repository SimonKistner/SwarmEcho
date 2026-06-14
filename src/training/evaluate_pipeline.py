"""
training/evaluate_pipeline.py
=============================
Unified High-Throughput Parallel Evaluation and Failure Analysis Pipeline.

This pipeline performs the following in a single cohesive execution:
  1. Runs parallel JAX evaluation rollouts across 4096 environments to sweep for failures.
  2. Saves failed target positions to a CSV and generates a failed chain targets heatmap overlay.
  3. Optionally generates a target-not-delivered heatmap.
  4. Optionally generates a target-not-visually-found heatmap.
  5. Optionally merges "Not Visually Found" (sky blue) and "Not Delivered" (dark blue) target groups
     into a single merged heatmap with a top-padded title/legend layout.
  6. Clusters failed positions using HDBSCAN (or BFS connected components) to detect spatial patterns.
  7. Plots clustered failures and highlights cluster representatives on a map blueprint.
  8. Optionally simulates and renders video rollouts of the cluster representatives.

Usage:
------
    # Run pipeline according to top-level toggle configurations:
    uv run python src/training/evaluate_pipeline.py checkpoint=outputs/my_run/checkpoints/ckpt_001000

    # Render specific failed target videos from CSV (legacy evaluate_heatmap.py feature):
    uv run python src/training/evaluate_pipeline.py checkpoint=outputs/my_run/checkpoints/ckpt_001000 --render-failed-csv=5
"""

import sys
import re
import csv
import time
from pathlib import Path
import dataclasses
import yaml
import numpy as np
import cv2
import jax
import jax.numpy as jnp
from flax import nnx

# Add project src root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import load_config, validate_config, compute_obs_dim, compute_action_dim
from env.physics import make_env_fns
from env.observations import make_obs_fns
from env.rewards import make_reward_fn
from models.mappo import MAPPOModel
from training.runner import _evaluate, _evaluate_parallel
from training.video_worker import render_eval_video
from env.maps import MapDefinition
from visualize.render_preview import render_png, _resolve_map_path

# ==============================================================================
# Pipeline Artifact Output Toggles
# ==============================================================================
CREATE_CSV = True                    # Save coordinates of failed episodes to CSV log (False = load from latest CSV)
CREATE_FAILED_CHAIN_HEATMAP = True   # Render failed targets chain heatmap overlay image
CREATE_NOT_DELIVERED_HEATMAP = True  # Render heatmap showing target positions as dots when NOT delivered to base
CREATE_NOT_VISUALLY_FOUND_HEATMAP = True # Render heatmap showing target positions as dots when NOT visually found by any drone
CREATE_CLUSTER_MAP = True            # Run failure clustering and save colored overlay image
CREATE_CLUSTER_VIDEOS = True         # Simulate and render rollout videos for cluster representatives

# ==============================================================================
# Pipeline Configuration Constants
# ==============================================================================
# 1. Parallel simulation parameters
NUM_ENVS = 4096             # Number of environments to evaluate in parallel
SEED = 42                   # Random seed for env reset and model initialization
SCALE = 8.0                 # Resolution scale (pixels per world-meter) for map image
SHOW_SPAWN_ZONES = False    # Set to False to disable target/base spawn zones overlay

# Merge "Not Visually Found" and "Not Delivered" heatmaps into a single heatmap
MERGE_TARGET_FOUND_HEATMAPS = True

# 2. Heatmap overlay parameters
FAILED_CHAIN_HEATMAP_ALPHA = 0.7   # Transparency of overlay dots in the failed chain heatmap (0.0 to 1.0)
FAILED_CHAIN_HEATMAP_DOT_RADIUS = 2 # Radius in pixels of failed chain heatmap dots
HEATMAP_ALPHA = 0.7         # Transparency of overlay dots in the not-found heatmaps (0.0 to 1.0)
HEATMAP_DOT_RADIUS = 2      # Radius in pixels of not-found heatmap dots

# 3. Clustering parameters
CLUSTERING_METHOD = "hdbscan"  # Active method: "hdbscan" or "bfs"

# BFS Connected Components parameters
EPSILON = 5.0                 # Distance threshold (in meters) for points to be neighbors
MIN_CLUSTER_FRACTION = 0.08   # Min fraction of total failures to form a cluster (e.g. 0.08 = 8%)

# HDBSCAN parameters (requires scikit-learn or standalone hdbscan)
HDBSCAN_MIN_CLUSTER_FRACTION = 0.035 # Min fraction of total failures to form a cluster
HDBSCAN_MIN_SAMPLES = None          # Min samples for core points (None defaults to min_cluster_size)
HDBSCAN_EPSILON = 0.0               # cluster_selection_epsilon (0.0 means no threshold)

# Cluster heatmap visualization parameters
CLUSTER_HEATMAP_ALPHA = 0.5   # Opacity blending factor for cluster overlay
CLUSTER_COLORS = [
    (235, 99, 37),    # Blue (BGR format)
    (74, 163, 22),    # Green
    (234, 51, 147),   # Purple
    (6, 119, 217),    # Orange
    (136, 148, 13),   # Teal
    (72, 29, 225),    # Rose
    (8, 179, 234),    # Yellow
    (68, 68, 239),    # Red
]

# 4. Representative rollout parameters
RENDER_NUM_CLUSTERS = 5    # Number of cluster representatives to render. None = all.


# ==============================================================================
# Helper Algorithms
# ==============================================================================
def cluster_points_bfs(points, eps, min_fraction):
    """
    Density-based connected components clustering using Breadth-First Search (BFS).
    Guarantees execution without external dependencies.
    """
    n = len(points)
    if n == 0:
        return [], [], []
    min_samples = max(1, int(min_fraction * n))

    # Calculate pairwise Euclidean distances
    diff = points[:, None, :] - points[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)

    # Neighbors adjacency graph
    adj = dists <= eps

    visited = np.zeros(n, dtype=bool)
    raw_clusters = []

    for i in range(n):
        if not visited[i]:
            queue = [i]
            visited[i] = True
            cluster = []
            head = 0
            while head < len(queue):
                curr = queue[head]
                cluster.append(curr)
                # Find unvisited neighbors within epsilon
                neighbors = np.where(adj[curr] & ~visited)[0]
                for neb in neighbors:
                    visited[neb] = True
                    queue.append(neb)
                head += 1
            raw_clusters.append(cluster)

    # Filter clusters by minimum size requirement
    valid_clusters = [c for c in raw_clusters if len(c) >= min_samples]

    # Find the representative point for each valid cluster
    reps = []
    cluster_indices = []
    for c in valid_clusters:
        coords = points[c]
        center = np.mean(coords, axis=0)
        # Find the actual point closest to the geometric center
        d_to_center = np.linalg.norm(coords - center, axis=-1)
        rep_idx = c[np.argmin(d_to_center)]
        reps.append(rep_idx)
        cluster_indices.append(c)

    # Identify outliers (noise points not belonging to any valid cluster)
    in_valid_cluster = np.zeros(n, dtype=bool)
    for c in valid_clusters:
        in_valid_cluster[c] = True
    outliers = np.where(~in_valid_cluster)[0].tolist()

    return cluster_indices, reps, outliers


def cluster_points_hdbscan(points, min_cluster_size, min_samples=None, epsilon=0.0):
    """
    Cluster points using HDBSCAN (either from scikit-learn or the standalone hdbscan library).
    Returns cluster_indices, reps, outliers, or None if no library is available.
    """
    labels = None

    # Try using scikit-learn first
    try:
        from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
        clusterer = SklearnHDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=epsilon
        )
        labels = clusterer.fit_predict(points)
    except (ImportError, TypeError) as e:
        if isinstance(e, TypeError):
            print(f"\n[HDBSCAN Warning] scikit-learn failed with TypeError: {e}")
            print("This is a known compatibility issue between NumPy 2.x and scikit-learn's epsilon-search code path.")
            print("Attempting to fall back to the standalone 'hdbscan' library...")

        # Fall back to standalone hdbscan
        try:
            import hdbscan as standalone_hdbscan
            clusterer = standalone_hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=epsilon
            )
            labels = clusterer.fit_predict(points)
        except ImportError:
            if isinstance(e, TypeError):
                print("\n[HDBSCAN ERROR] Standalone 'hdbscan' is not installed.")
                print("To run with non-zero HDBSCAN_EPSILON under NumPy 2.x, please either:")
                print("  1. Install standalone hdbscan: pip install hdbscan")
                print("  2. Set HDBSCAN_EPSILON = 0.0 to bypass the buggy code path.")
            else:
                print("\n[HDBSCAN ERROR] Neither 'scikit-learn' (sklearn.cluster.HDBSCAN) nor standalone 'hdbscan' is installed.")
                print("To run with scikit-learn on-the-fly via uv, use:")
                print("  uv run --with scikit-learn python src/training/evaluate_pipeline.py checkpoint=<path>")
                print("Or add it to your environment: pip install scikit-learn")
            return None

    unique_labels = sorted(list(set(labels)))
    cluster_indices = []
    reps = []

    for label in unique_labels:
        if label == -1:
            continue
        members = np.where(labels == label)[0].tolist()
        cluster_indices.append(members)

        # Representative: closest actual point to geometric mean
        coords = points[members]
        center = np.mean(coords, axis=0)
        d_to_center = np.linalg.norm(coords - center, axis=-1)
        rep_idx = members[np.argmin(d_to_center)]
        reps.append(rep_idx)

    outliers = np.where(labels == -1)[0].tolist()
    return cluster_indices, reps, outliers


def make_override_reset(r_fn, target_x, target_y):
    """Wraps environment reset to override the target_pos field."""
    def override_reset(k):
        s = r_fn(k)
        return s.replace(target_pos=jnp.array([target_x, target_y], dtype=jnp.float32))
    return override_reset


# ==============================================================================
# Pipeline Operations
# ==============================================================================
def setup_model_and_env(cfg, checkpoint_path):
    """Initializes environment functions and loads model from checkpoint."""
    # 1. Environment
    env_step, reset, _, (resolved_W, resolved_H, occ_grid, comm_occ_grid) = make_env_fns(cfg)
    compute_obs, _ = make_obs_fns(cfg, resolved_W, resolved_H, occ_grid, comm_occ_grid)
    compute_reward = make_reward_fn(cfg)

    # 2. Model
    obs_dim = compute_obs_dim(cfg)
    act_dim = compute_action_dim(cfg)
    N = int(cfg.env.num_agents)
    critic_type = str(cfg.network.critic_type)

    rngs = nnx.Rngs(SEED)
    model = MAPPOModel(
        obs_dim          = obs_dim,
        act_dim          = act_dim,
        num_agents       = N,
        hidden_dim       = int(cfg.network.hidden_dim),
        num_layers       = int(cfg.network.num_layers),
        actor_num_layers = int(cfg.network.actor_num_layers),
        critic_type      = critic_type,
        actor_memory     = bool(cfg.network.get("actor_memory", False)),
        critic_memory    = bool(cfg.network.get("critic_memory", False)),
        rngs             = rngs,
        memory_comm_enabled = bool(cfg.network.get("memory_comm_enabled", False)),
        memory_comm_gradient_mode = str(cfg.network.get("memory_comm_gradient_mode", "rial")),
        memory_comm_every_k_steps = int(cfg.network.get("memory_comm_every_k_steps", 5)),
        memory_comm_num_heads = int(cfg.network.get("memory_comm_num_heads", 4)),
    )

    # 3. Load Checkpoint
    import orbax.checkpoint as ocp
    _, empty_state = nnx.split(model)
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    restored_state = checkpointer.restore(
        str(checkpoint_path.absolute()),
        args=ocp.args.StandardRestore(empty_state),
    )
    nnx.update(model, restored_state)
    print("Checkpoint loaded successfully ✓")

    return model, env_step, reset, compute_obs, compute_reward


def run_parallel_eval(model, cfg, env_step, reset, compute_obs, compute_reward, track_delivered=True, track_visual=True):
    """Unified wrapper that runs _evaluate_parallel to satisfy DRY compliance and fix memory-communication evaluation."""
    max_steps = int(cfg.env.max_steps)
    eval_key = jax.random.PRNGKey(SEED)

    start_time = time.time()
    (rets, lengths, gaps, progs, succs, fnds,
     final_state, final_succs, final_delivered, final_visual) = _evaluate_parallel(
        model, reset, env_step, compute_obs, compute_reward,
        cfg, eval_key, num_envs=NUM_ENVS
    )
    final_succs.block_until_ready()
    elapsed = time.time() - start_time
    print(f"Simulation completed in {elapsed:.2f} seconds.")

    # Calculate Stats
    num_success = int(jnp.sum(final_succs))
    num_fail = NUM_ENVS - num_success
    success_rate = (num_success / NUM_ENVS) * 100.0
    print(f"         Successes:      {num_success}/{NUM_ENVS} ({success_rate:.2f}%)")

    # Extract target positions
    target_positions = np.array(final_state.target_pos)
    if target_positions.ndim == 3:
        target_positions = target_positions[:, 0, :]  # fallback for MEM_T8

    failed_mask = np.array(~final_succs)
    failed_positions = target_positions[failed_mask]

    if track_delivered:
        num_delivered = int(jnp.sum(final_delivered))
        num_not_delivered = NUM_ENVS - num_delivered
        delivered_rate = (num_delivered / NUM_ENVS) * 100.0
        print(f"         Delivered:      {num_delivered}/{NUM_ENVS} ({delivered_rate:.2f}%)")
        not_delivered_mask = np.array(~final_delivered)
        not_delivered_positions = target_positions[not_delivered_mask]
    else:
        num_delivered = 0
        num_not_delivered = 0
        delivered_rate = 0.0
        not_delivered_positions = np.zeros((0, 2))

    if track_visual:
        num_visually_found = int(jnp.sum(final_visual))
        num_not_visually_found = NUM_ENVS - num_visually_found
        visually_found_rate = (num_visually_found / NUM_ENVS) * 100.0
        print(f"Results: Visually Found: {num_visually_found}/{NUM_ENVS} ({visually_found_rate:.2f}%)")
        not_visually_found_mask = np.array(~final_visual)
        not_visually_found_positions = target_positions[not_visually_found_mask]
    else:
        num_visually_found = 0
        num_not_visually_found = 0
        visually_found_rate = 0.0
        not_visually_found_positions = np.zeros((0, 2))

    return (
        failed_positions, not_delivered_positions, not_visually_found_positions,
        success_rate, delivered_rate, visually_found_rate,
        num_fail, num_not_delivered, num_not_visually_found
    )


def load_map_data(cfg):
    """Loads map definition and blueprint configs."""
    map_names = cfg.env.get("map_names", [])
    if not map_names:
        print("ERROR: No map specified in configuration.")
        sys.exit(1)
    map_name = map_names[0]

    map_path = _resolve_map_path(map_name)
    if not map_path.exists():
        print(f"ERROR: Map definition not found at {map_path}")
        sys.exit(1)

    with open(map_path, "r") as f:
        map_data = yaml.safe_load(f)

    try:
        map_def = MapDefinition.load(map_path, cell_size=1.0)
    except Exception as e:
        print(f"Warning: Could not load MapDefinition: {e}")
        map_def = None

    return map_name, map_data, map_def


def render_and_save_failed_chain_heatmap(failed_positions, map_data, map_def, success_rate, num_fail, run_dir, video_dir, run_timestamp, save_csv=True, save_png=True, total_episodes=4096):
    """
    Generates the failed targets CSV and/or renders the failed chain heatmap image.
    The heatmap is generated with a 60px white border at the top displaying the 
    sliding window size/total episodes, failed chain counts, success rate, and legend.
    """
    csv_path = video_dir / f"{run_timestamp}_failed_target_positions.csv"
    heatmap_path = video_dir / f"{run_timestamp}_failed_chain_targets_heatmap.png"

    # Save CSV
    if save_csv:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])
            for pos in failed_positions:
                writer.writerow([f"{pos[0]:.6f}", f"{pos[1]:.6f}"])
        print(f"Saved failed target position(s) to: {csv_path.name}")

    # Generate Heatmap image
    if save_png:
        background_img = render_png(
            data=map_data,
            map_def=map_def,
            state=None,
            show_zones=SHOW_SPAWN_ZONES,
            show_spawns=False,
            scale=SCALE,
        )

        overlay = background_img.copy()
        height = float(map_data["height"])

        for pos in failed_positions:
            px = int(pos[0] * SCALE)
            py = int((height - pos[1]) * SCALE)
            cv2.circle(overlay, (px, py), FAILED_CHAIN_HEATMAP_DOT_RADIUS, (68, 68, 239), -1, cv2.LINE_AA)

        heatmap_img = cv2.addWeighted(overlay, FAILED_CHAIN_HEATMAP_ALPHA, background_img, 1.0 - FAILED_CHAIN_HEATMAP_ALPHA, 0)
        
        # Add 60px top padding for title and legend
        padded_img = cv2.copyMakeBorder(heatmap_img, 60, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        
        rate_str = f"{success_rate:.1f}%" if success_rate is not None else "?"
        info_str = f"Run: {run_dir.name} | Failed Chain: {num_fail}/{total_episodes} | Success Rate: {rate_str}"
        cv2.putText(padded_img, info_str, (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.42, (55, 41, 31), 1, cv2.LINE_AA)
        
        # Draw Legend
        cv2.circle(padded_img, (15, 46), 4, (68, 68, 239), -1, cv2.LINE_AA)
        cv2.putText(padded_img, "Failed Chain Target", (25, 50), cv2.FONT_HERSHEY_DUPLEX, 0.38, (55, 41, 31), 1, cv2.LINE_AA)

        cv2.imwrite(str(heatmap_path), padded_img)
        print(f"Saved failed chain targets heatmap to: {heatmap_path.name}")

    return csv_path


def render_and_save_not_found_heatmap(not_found_positions, map_data, map_def, found_rate, num_not_found, run_dir, video_dir, run_timestamp, filename_prefix, label, total_episodes=4096):
    """
    Generates a secondary heatmap plotting target coordinates that were not found/delivered.
    Uses blue dots on the map, with a 60px top padding containing title stats and a color legend.
    """
    heatmap_path = video_dir / f"{run_timestamp}_{filename_prefix}.png"

    # Generate heatmap background
    background_img = render_png(
        data=map_data,
        map_def=map_def,
        state=None,
        show_zones=SHOW_SPAWN_ZONES,
        show_spawns=False,
        scale=SCALE,
    )

    overlay = background_img.copy()
    height = float(map_data["height"])

    for pos in not_found_positions:
        px = int(pos[0] * SCALE)
        py = int((height - pos[1]) * SCALE)
        # Blue color in BGR is (235, 99, 37) (Harmonious Blue)
        cv2.circle(overlay, (px, py), HEATMAP_DOT_RADIUS, (235, 99, 37), -1, cv2.LINE_AA)

    heatmap_img = cv2.addWeighted(overlay, HEATMAP_ALPHA, background_img, 1.0 - HEATMAP_ALPHA, 0)
    
    # Add 60px top padding for title and legend
    padded_img = cv2.copyMakeBorder(heatmap_img, 60, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    info_str = f"Run: {run_dir.name} | {label}: {num_not_found}/{total_episodes} | Rate: {found_rate:.1f}%"
    cv2.putText(padded_img, info_str, (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.42, (55, 41, 31), 1, cv2.LINE_AA)
    
    # Draw Legend
    cv2.circle(padded_img, (15, 46), 4, (235, 99, 37), -1, cv2.LINE_AA)
    cv2.putText(padded_img, label, (25, 50), cv2.FONT_HERSHEY_DUPLEX, 0.38, (55, 41, 31), 1, cv2.LINE_AA)

    cv2.imwrite(str(heatmap_path), padded_img)
    print(f"Saved {label.lower()} heatmap to: {heatmap_path.name}")


def render_and_save_merged_heatmap(not_delivered_positions, not_visually_found_positions, map_data, map_def, delivered_rate, visually_found_rate, num_not_delivered, num_not_visually_found, run_dir, video_dir, run_timestamp, total_episodes=4096):
    """
    Generates a merged heatmap combining 'Not Visually Found' (sky blue) and 
    'Visually Found, Not Delivered' (dark blue) target coordinate groups.
    Includes a validation check warning if 'Not Visually Found' count exceeds 
    'Not Delivered', and formats title stats and a dual-color legend in a 60px top white border.
    """
    heatmap_path = video_dir / f"{run_timestamp}_merged_targets_heatmap.png"

    # Math safety check
    if num_not_visually_found > num_not_delivered:
        print(f"WARNING: num_not_visually_found ({num_not_visually_found}) is greater than num_not_delivered ({num_not_delivered})! "
              f"This does not make sense since a target must be visually found to be delivered.")

    # Generate heatmap background
    background_img = render_png(
        data=map_data,
        map_def=map_def,
        state=None,
        show_zones=SHOW_SPAWN_ZONES,
        show_spawns=False,
        scale=SCALE,
    )

    overlay = background_img.copy()
    height = float(map_data["height"])

    # Convert not_visually_found positions to set of tuples for fast matching
    not_visually_found_set = {tuple(pos) for pos in not_visually_found_positions}

    for pos in not_delivered_positions:
        px = int(pos[0] * SCALE)
        py = int((height - pos[1]) * SCALE)
        
        # Check matching
        if tuple(pos) in not_visually_found_set:
            # Blue color in BGR is (235, 99, 37) (Harmonious Blue)
            color = (235, 99, 37)
        else:
            # Orange color in BGR is (6, 119, 217) (Harmonious Orange)
            color = (6, 119, 217)
            
        cv2.circle(overlay, (px, py), HEATMAP_DOT_RADIUS, color, -1, cv2.LINE_AA)

    heatmap_img = cv2.addWeighted(overlay, HEATMAP_ALPHA, background_img, 1.0 - HEATMAP_ALPHA, 0)
    
    # Add 60px top padding for title and legend
    padded_img = cv2.copyMakeBorder(heatmap_img, 60, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    # Title stats
    info_str = f"Run: {run_dir.name} | Not Delivered: {num_not_delivered}/{total_episodes} | Not Visually Found: {num_not_visually_found}/{total_episodes}"
    cv2.putText(padded_img, info_str, (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.40, (55, 41, 31), 1, cv2.LINE_AA)
    
    # Dual color legend items
    # 1. Visually Found, Not Delivered (Orange)
    cv2.circle(padded_img, (15, 46), 4, (6, 119, 217), -1, cv2.LINE_AA)
    cv2.putText(padded_img, "Visually Found, Not Delivered", (25, 50), cv2.FONT_HERSHEY_DUPLEX, 0.38, (55, 41, 31), 1, cv2.LINE_AA)
    
    # 2. Not Visually Found (Blue)
    cv2.circle(padded_img, (260, 46), 4, (235, 99, 37), -1, cv2.LINE_AA)
    cv2.putText(padded_img, "Not Visually Found", (270, 50), cv2.FONT_HERSHEY_DUPLEX, 0.38, (55, 41, 31), 1, cv2.LINE_AA)

    cv2.imwrite(str(heatmap_path), padded_img)
    print(f"Saved merged targets heatmap to: {heatmap_path.name}")


def load_failures_from_csv(csv_path):
    """Loads target coordinates from a previously generated CSV file."""
    failed_positions = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header row if present
        if header:
            try:
                x, y = float(header[0]), float(header[1])
                failed_positions.append((x, y))
            except ValueError:
                pass
        for row in reader:
            if len(row) >= 2:
                try:
                    failed_positions.append((float(row[0]), float(row[1])))
                except ValueError:
                    continue
    return failed_positions


def run_failure_clustering(failed_positions, map_data, map_def, video_dir, run_timestamp, render_png_flag=True):
    """Clusters failed positions, draws visualization if toggled, and returns representative coordinates."""
    n_failures = len(failed_positions)
    if n_failures == 0:
        print("No failures to cluster.")
        return []

    used_hdbscan = False
    min_size = 0

    if CLUSTERING_METHOD.lower() == "hdbscan":
        min_size = max(2, int(HDBSCAN_MIN_CLUSTER_FRACTION * n_failures))
        print(f"Clustering failures using HDBSCAN (min_cluster_size={min_size}, epsilon={HDBSCAN_EPSILON})...")
        hdb_res = cluster_points_hdbscan(
            failed_positions,
            min_cluster_size=min_size,
            min_samples=HDBSCAN_MIN_SAMPLES,
            epsilon=HDBSCAN_EPSILON
        )
        if hdb_res is not None:
            cluster_indices, reps, outliers = hdb_res
            used_hdbscan = True
        else:
            print("Falling back to BFS Clustering...")
            cluster_indices, reps, outliers = cluster_points_bfs(failed_positions, EPSILON, MIN_CLUSTER_FRACTION)
            min_size = max(1, int(MIN_CLUSTER_FRACTION * n_failures))
    else:
        print(f"Clustering failures using BFS (EPSILON={EPSILON}m, MIN_CLUSTER_FRACTION={MIN_CLUSTER_FRACTION*100:.1f}%)...")
        cluster_indices, reps, outliers = cluster_points_bfs(failed_positions, EPSILON, MIN_CLUSTER_FRACTION)
        min_size = max(1, int(MIN_CLUSTER_FRACTION * n_failures))

    n_clusters = len(cluster_indices)
    print(f"Clustering complete: found {n_clusters} cluster(s).")
    if n_clusters == 0:
        print("No valid clusters found under specified threshold parameters.")
        return []

    # Draw and overlay clustered targets
    background_img = render_png(
        data=map_data,
        map_def=map_def,
        state=None,
        show_zones=SHOW_SPAWN_ZONES,
        show_spawns=False,
        scale=SCALE,
    )

    overlay = background_img.copy()
    height = float(map_data["height"])

    # 1. Draw outliers (noise points) in light gray
    for idx in outliers:
        pos = failed_positions[idx]
        px = int(pos[0] * SCALE)
        py = int((height - pos[1]) * SCALE)
        cv2.circle(overlay, (px, py), 2, (200, 200, 200), -1, cv2.LINE_AA)

    # 2. Draw clusters
    for c_idx, c_members in enumerate(cluster_indices):
        color = CLUSTER_COLORS[c_idx % len(CLUSTER_COLORS)]
        for idx in c_members:
            if idx == reps[c_idx]:
                continue
            pos = failed_positions[idx]
            px = int(pos[0] * SCALE)
            py = int((height - pos[1]) * SCALE)
            cv2.circle(overlay, (px, py), 2, color, -1, cv2.LINE_AA)

    # Blend overlay with blueprint
    img = cv2.addWeighted(overlay, CLUSTER_HEATMAP_ALPHA, background_img, 1.0 - CLUSTER_HEATMAP_ALPHA, 0)

    # 3. Draw cluster representatives (opaque, black borders)
    rep_coords = []
    for c_idx, rep_idx in enumerate(reps):
        color = CLUSTER_COLORS[c_idx % len(CLUSTER_COLORS)]
        pos = failed_positions[rep_idx]
        rep_coords.append(pos)
        px = int(pos[0] * SCALE)
        py = int((height - pos[1]) * SCALE)

        # Draw a larger circle with black border
        cv2.circle(img, (px, py), 6, color, -1, cv2.LINE_AA)
        cv2.circle(img, (px, py), 6, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Text label
        cv2.putText(img, f"C{c_idx}", (px + 9, py + 4), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    # Draw stats title
    if used_hdbscan:
        info_str = f"HDBSCAN: {n_clusters} clusters found | MinSize={min_size} | Eps={HDBSCAN_EPSILON}"
    else:
        info_str = f"BFS: {n_clusters} clusters found | EPS={EPSILON}m | MinFract={MIN_CLUSTER_FRACTION*100:.0f}% (MinSize={min_size})"
    cv2.putText(img, info_str, (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.45, (55, 41, 31), 1, cv2.LINE_AA)

    # Save image if toggled
    if render_png_flag:
        clustered_path = video_dir / f"{run_timestamp}_failed_targets_clustered.png"
        cv2.imwrite(str(clustered_path), img)
        print(f"Saved clustered failed targets map to: {clustered_path.name}")

    return rep_coords


def render_cluster_videos(model, cfg, env_step, reset, compute_obs, compute_reward, rep_coords, video_dir, run_timestamp):
    """Simulates and renders rollout videos for identified representatives (explicitly timestamped)."""
    num_to_render = RENDER_NUM_CLUSTERS if RENDER_NUM_CLUSTERS is not None else len(rep_coords)
    num_to_render = min(num_to_render, len(rep_coords))
    print(f"Simulating rollout videos for the first {num_to_render} cluster representative(s)...")

    renderer = str(cfg.visualize.get("final_eval_renderer", "slow"))
    key = jax.random.PRNGKey(SEED)

    for c_idx in range(num_to_render):
        tx, ty = rep_coords[c_idx]
        print(f"  Cluster {c_idx} Representative Target: ({tx:.2f}, {ty:.2f})")

        custom_reset = make_override_reset(reset, tx, ty)

        # Run rollout simulation
        (all_states, all_rewards, all_metrics,
         _, _, _, _, _, _) = _evaluate(
            model=model,
            reset_fn=jax.jit(custom_reset),
            env_step_fn=jax.jit(env_step),
            obs_fn=jax.jit(compute_obs),
            reward_fn=jax.jit(compute_reward),
            cfg=cfg,
            key=key,
            num_episodes=1,
        )

        filename_stem = f"{run_timestamp}_FAIL_cluster_{c_idx}_rep_{tx:.2f}_{ty:.2f}"
        vid_path = render_eval_video(
            ep_states=all_states[0],
            ep_rewards=all_rewards[0],
            ep_metrics=all_metrics[0],
            cfg=cfg,
            out_dir=video_dir,
            filename_stem=filename_stem,
            renderer=renderer,
        )
        print(f"    ✓ Video saved: {Path(vid_path).name}")


def run_legacy_csv_rendering(model, cfg, env_step, reset, compute_obs, compute_reward, video_dir, limit, run_timestamp):
    """Legacy feature: Simulates and renders failed videos directly from a previously saved CSV file."""
    csv_candidates = list(video_dir.glob("*failed_target_positions*.csv"))
    if not csv_candidates:
        print(f"ERROR: No failed target positions CSV found in: {video_dir}")
        sys.exit(1)
    csv_path = max(csv_candidates, key=lambda p: p.stat().st_mtime)
    print(f"Loading failed target positions from CSV: {csv_path.name}")

    # Load failed targets
    failed_positions = load_failures_from_csv(csv_path)
    print(f"Loaded {len(failed_positions)} failed target positions from CSV.")

    # Identify already rendered targets
    rendered_positions = set()
    if video_dir.exists():
        for f in video_dir.glob("*FAIL_target_*.mp4"):
            # Check for name match including timestamp suffixes
            match = re.search(r"FAIL_target_([0-9\.\-]+)_([0-9\.\-]+)", f.name)
            if match:
                rx = float(match.group(1))
                ry = float(match.group(2))
                rendered_positions.add((round(rx, 2), round(ry, 2)))

    # Filter targets to render
    to_render = []
    for pos in failed_positions:
        tx, ty = pos
        if (round(tx, 2), round(ry, 2)) not in rendered_positions:
            to_render.append(pos)
            if len(to_render) >= limit:
                break

    if not to_render:
        print("All failed targets from the CSV have already been rendered! ✓")
        return

    print(f"Identified {len(to_render)} new target position(s) to render.")
    renderer = str(cfg.visualize.get("final_eval_renderer", "slow"))
    key = jax.random.PRNGKey(SEED)

    for idx, (tx, ty) in enumerate(to_render):
        print(f"  [{idx+1}/{len(to_render)}] Simulating rollout for target: ({tx:.2f}, {ty:.2f})")
        custom_reset = make_override_reset(reset, tx, ty)

        (all_states, all_rewards, all_metrics,
         _, _, _, _, _, _) = _evaluate(
            model=model,
            reset_fn=jax.jit(custom_reset),
            env_step_fn=jax.jit(env_step),
            obs_fn=jax.jit(compute_obs),
            reward_fn=jax.jit(compute_reward),
            cfg=cfg,
            key=key,
            num_episodes=1,
        )

        filename_stem = f"{run_timestamp}_FAIL_target_{tx:.2f}_{ty:.2f}"
        vid_path = render_eval_video(
            ep_states=all_states[0],
            ep_rewards=all_rewards[0],
            ep_metrics=all_metrics[0],
            cfg=cfg,
            out_dir=video_dir,
            filename_stem=filename_stem,
            renderer=renderer,
        )
        print(f"    ✓ Video saved: {Path(vid_path).name}")


# ==============================================================================
# Main Execution Entry Point
# ==============================================================================
def main():
    # Generate Run-Start Timestamp for consistent output file labeling
    run_timestamp = time.strftime("%Y_%m_%d_%H_%M")

    global CREATE_CSV, CREATE_FAILED_CHAIN_HEATMAP, CREATE_NOT_DELIVERED_HEATMAP, CREATE_NOT_VISUALLY_FOUND_HEATMAP, CREATE_CLUSTER_MAP, CREATE_CLUSTER_VIDEOS

    # Parse CLI Arguments
    args = sys.argv[1:]
    checkpoint_path = None
    render_failed_csv = None
    overrides = []

    for arg in args:
        if arg.startswith("checkpoint="):
            checkpoint_path = Path(arg.split("=", 1)[1].replace("\\", "/"))
        elif arg.startswith("--render-failed-csv="):
            render_failed_csv = int(arg.split("=", 1)[1])
        elif arg.startswith("render_failed_csv="):
            render_failed_csv = int(arg.split("=", 1)[1])
        elif arg.lower() in ["obs_log=true", "obs_saving=true", "--obs-log", "--obs-saving"]:
            overrides.append("logging.obs_log=true")
        elif arg.lower() in ["obs_log=false", "obs_saving=false", "--no-obs-log", "--no-obs-saving"]:
            overrides.append("logging.obs_log=false")
        elif arg.lower() in ["connectivity=true", "conn_matrix=true", "--connectivity", "--conn-matrix"]:
            overrides.append("visualize.render_conn_matrix=true")
            overrides.append("env.log_adjacency_matrix=true")
        elif arg.lower() in ["connectivity=false", "conn_matrix=false", "--no-connectivity", "--no-conn-matrix"]:
            overrides.append("visualize.render_conn_matrix=false")
            overrides.append("env.log_adjacency_matrix=false")
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
        elif arg.lower() in ["cluster=true", "--cluster"]:
            CREATE_CLUSTER_MAP = True
        elif arg.lower() in ["cluster=false", "--no-cluster"]:
            CREATE_CLUSTER_MAP = False
        elif arg.lower() in ["videos=true", "--videos"]:
            CREATE_CLUSTER_VIDEOS = True
        elif arg.lower() in ["videos=false", "--no-videos"]:
            CREATE_CLUSTER_VIDEOS = False
        else:
            overrides.append(arg)

    if checkpoint_path is None:
        print("ERROR: Must specify checkpoint=<path>")
        print("  e.g.  uv run python src/training/evaluate_pipeline.py checkpoint=outputs/my_run/checkpoints/ckpt_001000")
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

    # Resolve Video / Output Directory
    video_dir = run_dir / "videos" / "eval"
    video_dir.mkdir(parents=True, exist_ok=True)

    # Map blueprint files
    map_name, map_data, map_def = load_map_data(cfg)

    # Initialize environment, model, and load checkpoint weights
    model, env_step, reset, compute_obs, compute_reward = setup_model_and_env(cfg, checkpoint_path)

    # Execute Selected Mode
    if render_failed_csv is not None:
        # Mode A: Legacy mode to render failed target videos from CSV
        print(f"\n--- Running Legacy CSV Target Replays (limit: {render_failed_csv}) ---")
        run_legacy_csv_rendering(model, cfg, env_step, reset, compute_obs, compute_reward, video_dir, render_failed_csv, run_timestamp)
        print("\nEvaluation pipeline complete.")
        return

    # ── Dependency Resolution & Execution Plan ────────────────────────────
    # Resolve CSV candidates early
    csv_candidates = list(video_dir.glob("*failed_target_positions*.csv"))

    # We need to run parallel JAX simulation sweep if:
    #   CREATE_CSV is requested (to get fresh coordinate logs) OR we want to render either of the target-not-found heatmaps
    run_sweep = CREATE_CSV or CREATE_NOT_DELIVERED_HEATMAP or CREATE_NOT_VISUALLY_FOUND_HEATMAP

    # PREREQUISITE FALLBACK CHECK:
    # If the user wants to load from CSV (run_sweep = False), but no CSV actually exists:
    # We must force the simulation sweep to run to generate the target coordinate log.
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
         num_fail, num_not_delivered, num_not_visually_found) = run_parallel_eval(
            model, cfg, env_step, reset, compute_obs, compute_reward,
            track_delivered=CREATE_NOT_DELIVERED_HEATMAP,
            track_visual=CREATE_NOT_VISUALLY_FOUND_HEATMAP
        )
        
        print("\n--- Phase 2: Processing Swept Coordinates ---")

        # Check if we should merge visually not found and not delivered heatmaps
        if MERGE_TARGET_FOUND_HEATMAPS:
            if CREATE_NOT_DELIVERED_HEATMAP or CREATE_NOT_VISUALLY_FOUND_HEATMAP:
                print("\n--- Phase 2a/b: Generating Merged Targets Heatmap overlay ---")
                render_and_save_merged_heatmap(
                    not_delivered_positions=not_delivered_positions,
                    not_visually_found_positions=not_visually_found_positions,
                    map_data=map_data,
                    map_def=map_def,
                    delivered_rate=delivered_rate,
                    visually_found_rate=visually_found_rate,
                    num_not_delivered=num_not_delivered,
                    num_not_visually_found=num_not_visually_found,
                    run_dir=run_dir,
                    video_dir=video_dir,
                    run_timestamp=run_timestamp,
                    total_episodes=NUM_ENVS
                )
        else:
            # 1. Visually Found Heatmap (Phase 2a)
            if CREATE_NOT_VISUALLY_FOUND_HEATMAP:
                print("\n--- Phase 2a: Generating Visually-Found Heatmap overlay ---")
                render_and_save_not_found_heatmap(
                    not_visually_found_positions, map_data, map_def, visually_found_rate, num_not_visually_found, 
                    run_dir, video_dir, run_timestamp, "not_visually_found_targets_heatmap", "Not Visually Found",
                    total_episodes=NUM_ENVS
                )

            # 2. Delivered Heatmap (Phase 2b)
            if CREATE_NOT_DELIVERED_HEATMAP:
                print("\n--- Phase 2b: Generating Delivered-To-Base Heatmap overlay ---")
                render_and_save_not_found_heatmap(
                    not_delivered_positions, map_data, map_def, delivered_rate, num_not_delivered, 
                    run_dir, video_dir, run_timestamp, "not_delivered_targets_heatmap", "Not Delivered",
                    total_episodes=NUM_ENVS
                )

        # 3. CSV and Failed Chain Heatmap (Phase 2c)
        if CREATE_CSV or CREATE_FAILED_CHAIN_HEATMAP:
            print("\n--- Phase 2c: Saving Coordinate Log & Failed Chain Heatmap overlay ---")
            _ = render_and_save_failed_chain_heatmap(
                failed_positions, map_data, map_def, success_rate, num_fail, run_dir, video_dir, run_timestamp,
                save_csv=CREATE_CSV, save_png=CREATE_FAILED_CHAIN_HEATMAP, total_episodes=NUM_ENVS
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
                run_dir=run_dir, video_dir=video_dir, run_timestamp=run_timestamp,
                save_csv=False, save_png=True, total_episodes=NUM_ENVS
            )

        if CREATE_NOT_DELIVERED_HEATMAP:
            print("\n[Prerequisite Warning] Delivered-to-base heatmap cannot be generated when loading from static CSV.")
        if CREATE_NOT_VISUALLY_FOUND_HEATMAP:
            print("\n[Prerequisite Warning] Visually-found heatmap cannot be generated when loading from static CSV.")

    # 5. Clustering and representative rollouts
    if not (CREATE_CLUSTER_MAP or CREATE_CLUSTER_VIDEOS):
        print("\nClustering phase is disabled via config toggles. Skipping Phase 3 & 4.")
    else:
        if len(failed_positions) > 0:
            print("\n--- Phase 3: Spatial Clustering Analysis ---")
            rep_coords = run_failure_clustering(
                failed_positions, map_data, map_def, video_dir, run_timestamp, 
                render_png_flag=CREATE_CLUSTER_MAP
            )

            if CREATE_CLUSTER_VIDEOS:
                if len(rep_coords) > 0:
                    print("\n--- Phase 4: Simulating & Rendering Representative Rollout Videos ---")
                    render_cluster_videos(model, cfg, env_step, reset, compute_obs, compute_reward, rep_coords, video_dir, run_timestamp)
        else:
            print("\nPerfect success rate! No failures to cluster.")

    print("\nEvaluation pipeline complete.")


if __name__ == "__main__":
    main()
