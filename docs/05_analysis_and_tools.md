# Analysis Tools & Dashboard

Monitoring SwarmEcho requires visualizing multi-agent behaviors dynamically. We employ specialized scripts embedded inside `src/analysis/` and `src/visualize/`.

## Local Intelligence Dashboard
*(Run via: `uv run streamlit run src/analysis/dashboard.py`)*

The `dashboard.py` script boots a robust Streamlit UI designed for auditing curricula directly targeting the `outputs/` folder. It provides:
- **Core Intelligence Profiles:** Detailed metric extraction.
- **Hierarchical Parameter Matrices:** Intelligently grouping config sets logically (`ENV`, `REWARD`, `TRAINING`) and highlighting hyperparameter deviations between training runs.
- **Video Inspection:** Watch baked JAX rollouts at specific curriculum levels.

## Visual Rendering 
*(Powered by `src/visualize/`)*

We avoid using pure JAX tools for final video exports since they lack fine GUI adjustments. 
- `renderer_cv2.py` focuses on highly performant OpenCV rasterization.
- `renderer_mpl.py` acts as a deeper fallback.
- `renderer.py` coordinates the unified interface to turn purely numerical JAX environments into the rich mp4 evaluation files found in `outputs/{run}/videos/`.

## Spatial Failure Analysis Pipeline
*(Run via: `uv run python src/training/evaluate_pipeline.py checkpoint=<path>`)*

The high-throughput evaluation pipeline runs parallel JAX simulation sweeps across 4096 environments to isolate failure coordinates. The script generates coordinate logs and diagnostic overlays inside the run's evaluation folder:

- **Failed Chain Targets Heatmap (`[timestamp]_failed_chain_targets_heatmap.png`):** Plots red dots for target positions where the shortest-chain relay to the base station could not be completed and held.
- **Failed Chain Targets CSV (`[timestamp]_failed_target_positions.csv`):** CSV log file containing the coordinates of target spawn failures.
- **Found-and-Delivered Heatmap (`found_and_delivered_u000700_s00070M.png`):** Combines two key target-spawner failure metrics onto a single blueprint by default:
  - **Not Visually Found (Sky Blue BGR `(235, 99, 37)`):** Target coordinates that were never visually seen by any drone in the swarm.
  - **Visually Found, Not Delivered (Dark Blue BGR `(6, 119, 217)`):** Target coordinates that were successfully seen by a drone (and updated in `target_known`), but never successfully routed back to the base.
- **Top Padded Legend Layout:** All heatmaps utilize a 60px top margin to print title stats and visual color legends, ensuring the blueprint remains un-cluttered.
- **Validation Check:** During execution, a verification warning is printed to the console if the number of targets not visually found exceeds those not delivered.

### Spatial Failure Clustering & Diagnostic Rollouts
To isolate geographical patterns, `evaluate_pipeline.py` supports clustering spatial failures using **HDBSCAN** or **Breadth-First Search (BFS)** connected components:
- **Clustered Failures Overlay (`[timestamp]_failed_targets_clustered.png`):** Automatically maps failed coordinates to identified density clusters (color-coded). Outliers / noise are marked in light gray. The geometric center / representative of each cluster is highlighted with a larger, black-bordered circle.
- **Cluster Representative Videos (`[timestamp]_FAIL_cluster_[idx]_rep_[x]_[y].mp4`):** When enabled (`CREATE_CLUSTER_VIDEOS = True`), the pipeline isolates the representative coordinate of the top failure clusters, runs a single-episode deterministic rollout with the target forced to those exact coordinates, and renders a video showing the failure trajectory to aid direct visual debugging.
- **Legacy Video Re-simulation:** You can also manually re-simulate and render rollout videos for the first N failures found in a previously generated CSV file by running the pipeline with the `--render-failed-csv=N` argument.

## W&B Dictionary Reference
When training on Weights and Biases (`wandb`), the key metrics include:
| Metric | Description |
|---|---|
| `train/ep_return` | Mean unified team return across the 1024 parallel environments. |
| `train/success_rate` | Fraction of the batched episodes resolving with a full chain link. |
| `train/target_found_rate` | Fraction of environments where the swarm found the target. |
| `train/chain_gap_dist` | Physical distance (metres) between the Base network subset and Target network subset. |
| `ppo/policy_loss` | PPO clipped surrogate loss algorithm output. |
| `ppo/value_loss` | MSE accuracy estimate of the generalized Critic module. |
| `ppo/clip_fraction` | Fraction of PPO samples whose policy ratio was clipped. Useful for spotting overly aggressive recurrent updates. |
| `ppo/approx_kl` | Approximate KL divergence between old and updated policies. |
| `perf/sps` | System Steps per second. |
