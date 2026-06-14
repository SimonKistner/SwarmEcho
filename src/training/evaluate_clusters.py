"""
training/evaluate_clusters.py
==============================
Cluster failed target positions from SwarmEcho runs, visualize colored clusters
on the map blueprint with highlighted representatives, and render evaluation
rollout videos of representative positions.

Usage
-----
    uv run python src/training/evaluate_clusters.py \
        checkpoint=outputs/my_run/checkpoints/ckpt_001000
"""

import sys
import re
import csv
import time
from pathlib import Path
import dataclasses
import numpy as np
import cv2
import yaml
import jax
import jax.numpy as jnp
from flax import nnx
from omegaconf import OmegaConf

# Add project src root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import load_config, validate_config, compute_obs_dim, compute_action_dim
from env.physics import make_env_fns
from env.observations import make_obs_fns
from env.rewards import make_reward_fn
from models.mappo import MAPPOModel
from training.runner import _evaluate
from training.video_worker import render_eval_video
from env.maps import MapDefinition
from visualize.render_preview import render_png, _resolve_map_path

# ==============================================================================
# Configuration Constants
# ==============================================================================
# Pick your active clustering method: "bfs" or "hdbscan"
CLUSTERING_METHOD = "hdbscan"

# ------------------------------------------------------------------------------
# BFS Connected Components Parameters
# ------------------------------------------------------------------------------
EPSILON = 5.0                 # Distance threshold (in meters) for points to be neighbors
MIN_CLUSTER_FRACTION = 0.08   # Min fraction of total failures to form a cluster (e.g. 0.15 = 15%)

# ------------------------------------------------------------------------------
# HDBSCAN Parameters (requires scikit-learn dependency)
# ------------------------------------------------------------------------------# HDBSCAN parameters
HDBSCAN_MIN_CLUSTER_FRACTION = 0.035 # Min fraction of total failures to form a cluster (float)
HDBSCAN_MIN_SAMPLES = None          # Min samples for core points (None defaults to min_cluster_size)
HDBSCAN_EPSILON = 0.0               # cluster_selection_epsilon (0.0 means no threshold)

RENDER_VIDEOS = True           # Set to False to skip simulating and rendering rollout videos
RENDER_NUM_CLUSTERS = None    # Number of cluster representatives to render. None = all. (Ignored if RENDER_VIDEOS=False)
SEED = 42                     # Random seed for env reset and evaluations
HEATMAP_ALPHA = 0.5          # Heatmap opacity blending factor
SCALE = 8.0                  # Resolution scale (pixels per world-meter) for map image
SHOW_SPAWN_ZONES = False     # Set to False to disable the red target/base spawn zones overlay

# Distinct colors in BGR format for plotting clusters
CLUSTER_COLORS = [
    (235, 99, 37),    # Blue
    (74, 163, 22),    # Green
    (234, 51, 147),   # Purple
    (6, 119, 217),    # Orange
    (136, 148, 13),   # Teal
    (72, 29, 225),    # Rose
    (8, 179, 234),    # Yellow
    (68, 68, 239),    # Red
]


def cluster_points(points, eps, min_fraction):
    """
    Density-based connected components clustering using pure BFS.
    Guarantees zero external dependency issues.
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
    outliers = np.where(~in_valid_cluster)[0]

    return cluster_indices, reps, outliers


def cluster_points_hdbscan(points, min_cluster_size, min_samples=None, epsilon=0.0):
    """
    Cluster points using HDBSCAN (either from scikit-learn or the hdbscan library).
    Returns cluster_indices, reps, outliers or None if not importable.
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
                print("  uv run --with scikit-learn python src/training/evaluate_clusters.py checkpoint=<path>")
                print("Or add it to your environment: pip install scikit-learn")
            return None

    n_samples = len(points)
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


def main():
    # ── Parse CLI args ────────────────────────────────────────────────────
    args = sys.argv[1:]
    checkpoint_path = None
    overrides = []

    for arg in args:
        if arg.startswith("checkpoint="):
            checkpoint_path = Path(arg.split("=", 1)[1].replace("\\", "/"))
        else:
            overrides.append(arg)

    if checkpoint_path is None:
        print("ERROR: Must specify checkpoint=<path>")
        print("  e.g.  uv run python src/training/evaluate_clusters.py checkpoint=outputs/my_run/checkpoints/ckpt_001000")
        sys.exit(1)

    # Standard layout: outputs/<run_name>/checkpoints/ckpt_XXXXXX
    if checkpoint_path.parent.name == "checkpoints":
        run_dir = checkpoint_path.parents[1]
    else:
        run_dir = Path("outputs")

    # ── Config ────────────────────────────────────────────────────────────
    run_config_path = run_dir / "config.yaml"
    run_config_loaded = run_config_path.exists()
    cfg = load_config(
        config_path = run_config_path if run_config_loaded else None,
        cli_overrides = True,
        overrides = overrides,
    )
    validate_config(cfg)

    # ── Build Model ───────────────────────────────────────────────────────
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
        memory_comm_merge = str(cfg.network.get("memory_comm_merge", "residual")),
        memory_comm_attention_mode = str(cfg.network.get("memory_comm_attention_mode", "attend_global_learned_query")),
    )

    # ── Load Checkpoint ───────────────────────────────────────────────────
    import orbax.checkpoint as ocp
    graphdef, empty_state = nnx.split(model)
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    restored_state = checkpointer.restore(
        str(checkpoint_path.absolute()),
        args=ocp.args.StandardRestore(empty_state),
    )
    nnx.update(model, restored_state)
    print("Checkpoint loaded successfully ✓")

    map_names = cfg.env.get("map_names", [])
    if not map_names:
        print("ERROR: No map specified in configuration.")
        sys.exit(1)
    map_name = map_names[0]

    video_dir = run_dir / "videos" / "eval"
    video_dir.mkdir(parents=True, exist_ok=True)

    # ── Load CSV ──────────────────────────────────────────────────────────
    csv_candidates = list(video_dir.glob("failed_target_positions*.csv"))
    if not csv_candidates:
        print(f"ERROR: No failed target positions CSV found in: {video_dir}")
        print("Please run parallel evaluation first using evaluate_heatmap.py to generate CSV.")
        sys.exit(1)
    csv_path = max(csv_candidates, key=lambda p: p.stat().st_mtime)
    print(f"Loading failed target positions from CSV: {csv_path.name}")

    failed_positions = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header row if present
        if header:
            try:
                x, y = float(header[0]), float(header[1])
                failed_positions.append((x, y))
            except ValueError:
                pass  # text header
        for row in reader:
            if len(row) >= 2:
                try:
                    failed_positions.append((float(row[0]), float(row[1])))
                except ValueError:
                    continue

    n_failures = len(failed_positions)
    print(f"Found {n_failures} total failures.")
    if n_failures == 0:
        print("No failures to cluster! Exiting.")
        return

    failed_positions_np = np.array(failed_positions)

    # ── Perform Clustering ────────────────────────────────────────────────
    used_hdbscan = False
    method = globals().get("CLUSTERING_METHOD", "bfs").lower()
    
    eps = globals().get("EPSILON", 10.0)
    min_fract = globals().get("MIN_CLUSTER_FRACTION", 0.15)

    if method == "hdbscan":
        hdb_min_fract = globals().get("HDBSCAN_MIN_CLUSTER_FRACTION", 0.15)
        hdb_min_samples = globals().get("HDBSCAN_MIN_SAMPLES", None)
        hdb_eps = globals().get("HDBSCAN_EPSILON", 0.0)

        # Determine min_size
        min_size = max(2, int(hdb_min_fract * n_failures))
            
        print(f"Clustering failures using HDBSCAN (min_cluster_size={min_size}, epsilon={hdb_eps})...")
        hdb_res = cluster_points_hdbscan(
            failed_positions_np,
            min_cluster_size=min_size,
            min_samples=hdb_min_samples,
            epsilon=hdb_eps
        )
        
        if hdb_res is not None:
            cluster_indices, reps, outliers = hdb_res
            used_hdbscan = True
        else:
            print("Falling back to BFS Clustering...")
            cluster_indices, reps, outliers = cluster_points(failed_positions_np, eps, min_fract)
            min_size = max(1, int(min_fract * n_failures))
    else:
        print(f"Clustering failures using BFS (EPSILON={eps}m, MIN_CLUSTER_FRACTION={min_fract*100:.1f}%)...")
        cluster_indices, reps, outliers = cluster_points(failed_positions_np, eps, min_fract)
        min_size = max(1, int(min_fract * n_failures))

    n_clusters = len(cluster_indices)
    print(f"Clustering complete: found {n_clusters} cluster(s).")
    if n_clusters == 0:
        print("No valid clusters found under specified threshold parameters. Exiting.")
        return

    # ── Draw and Overlay Clustered Targets ────────────────────────────────
    print("Rendering clustered map preview...")
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

    # 1. Draw outliers in light gray
    for idx in outliers:
        pos = failed_positions_np[idx]
        px = int(pos[0] * SCALE)
        py = int((height - pos[1]) * SCALE)
        cv2.circle(overlay, (px, py), 2, (200, 200, 200), -1, cv2.LINE_AA)

    # 2. Draw clusters and highlight representatives
    rep_coords = []
    for c_idx, c_members in enumerate(cluster_indices):
        color = CLUSTER_COLORS[c_idx % len(CLUSTER_COLORS)]
        
        # Draw all member points of this cluster
        for idx in c_members:
            if idx == reps[c_idx]:
                continue  # Draw representative separately
            pos = failed_positions_np[idx]
            px = int(pos[0] * SCALE)
            py = int((height - pos[1]) * SCALE)
            cv2.circle(overlay, (px, py), 2, color, -1, cv2.LINE_AA)

    # Blend overlay with background image
    img = cv2.addWeighted(overlay, HEATMAP_ALPHA, background_img, 1.0 - HEATMAP_ALPHA, 0)

    # 3. Draw representatives directly on the final image (opaque, sharp outlines)
    for c_idx, rep_idx in enumerate(reps):
        color = CLUSTER_COLORS[c_idx % len(CLUSTER_COLORS)]
        pos = failed_positions_np[rep_idx]
        rep_coords.append(pos)
        px = int(pos[0] * SCALE)
        py = int((height - pos[1]) * SCALE)

        # Draw a larger circle with black border
        cv2.circle(img, (px, py), 6, color, -1, cv2.LINE_AA)
        cv2.circle(img, (px, py), 6, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Text label
        cv2.putText(img, f"C{c_idx}", (px + 9, py + 4), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    # Draw description on clustered heatmap image
    if used_hdbscan:
        hdb_eps = globals().get("HDBSCAN_EPSILON", 0.0)
        info_str = f"HDBSCAN: {n_clusters} clusters found | MinSize={min_size} | Eps={hdb_eps}"
    else:
        info_str = f"BFS: {n_clusters} clusters found | EPS={eps}m | MinFract={min_fract*100:.0f}% (MinSize={min_size})"
    cv2.putText(img, info_str, (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.45, (55, 41, 31), 1, cv2.LINE_AA)

    clustered_path = video_dir / "failed_targets_clustered.png"
    if clustered_path.exists():
        ts = time.strftime("%Y%m%d_%H%M%S")
        clustered_path = video_dir / f"failed_targets_clustered_{ts}.png"
        print("Default clustered heatmap file already exists. Saving with timestamp suffix solver.")
    cv2.imwrite(str(clustered_path), img)
    print(f"Saved clustered failed targets map to: {clustered_path}")

    # ── Simulation & Rendering for Reps ──────────────────────────────────
    if not RENDER_VIDEOS:
        print("Video rendering is disabled (RENDER_VIDEOS = False). Skipping rollout simulations.")
        return

    num_to_render = RENDER_NUM_CLUSTERS if RENDER_NUM_CLUSTERS is not None else n_clusters
    num_to_render = min(num_to_render, n_clusters)
    print(f"Simulating rollout videos for the first {num_to_render} cluster representative(s)...")

    # Environment fns
    env_step, reset, _, (resolved_W, resolved_H, occ_grid, comm_occ_grid) = make_env_fns(cfg)
    compute_obs, _ = make_obs_fns(cfg, resolved_W, resolved_H, occ_grid, comm_occ_grid)
    compute_reward = make_reward_fn(cfg)

    renderer = str(cfg.visualize.get("final_eval_renderer", "slow"))
    key = jax.random.PRNGKey(SEED)

    for c_idx in range(num_to_render):
        tx, ty = rep_coords[c_idx]
        print(f"  Cluster {c_idx} Representative Target: ({tx:.2f}, {ty:.2f})")

        # Wrapper reset_fn to force the target position
        def make_override_reset(r_fn, target_x, target_y):
            def override_reset(k):
                s = r_fn(k)
                return s.replace(target_pos=jnp.array([target_x, target_y], dtype=jnp.float32))
            return override_reset

        custom_reset = make_override_reset(reset, tx, ty)

        # Run 1 evaluation episode
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

        # Render video
        filename_stem = f"FAIL_cluster_{c_idx}_rep_{tx:.2f}_{ty:.2f}"
        vid_path = render_eval_video(
            ep_states=all_states[0],
            ep_rewards=all_rewards[0],
            ep_metrics=all_metrics[0],
            cfg=cfg,
            out_dir=video_dir,
            filename_stem=filename_stem,
            renderer=renderer,
        )
        print(f"    ✓ Video saved: {vid_path}")


if __name__ == "__main__":
    main()
