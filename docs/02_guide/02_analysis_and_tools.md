# Analysis Tools & Dashboard

Monitoring SwarmEcho requires visualizing multi-agent behaviors dynamically. The packaged tools live under `src/swarmecho/analysis/` and `src/swarmecho/visualize/`.

## Local Intelligence Dashboard
*(Run via: `uv run swarmecho-dashboard`)*

The `dashboard.py` script boots a robust Streamlit UI designed for auditing curricula directly targeting the `outputs/` folder. It provides:
- **Core Intelligence Profiles:** Detailed metric extraction.
- **Hierarchical Parameter Matrices:** Intelligently grouping config sets logically (`ENV`, `REWARD`, `TRAINING`) and highlighting hyperparameter deviations between training runs.
- **Video Inspection:** Watch rendered evaluation rollouts and checkpoint artifacts.

The separate `swarmecho-eval-dashboard` entrypoint focuses on checkpoint-scoped
evaluation CSVs, heatmaps, and videos; `swarmecho-dashboard` compares training
runs and their configurations.

## Visual Rendering 
*(Powered by `src/swarmecho/visualize/`)*

Video exports use the OpenCV rasterizer in `renderer_cv2.py`.
`renderer.py` coordinates the interface that turns numerical JAX environment
trajectories into MP4 evaluation files.

## Spatial Failure Analysis Pipeline
*(Run via: `uv run swarmecho-evaluate-pipeline checkpoint=<path>`)*

The high-throughput evaluation pipeline runs the configured parallel JAX evaluation batch to isolate spatial behavior. It follows the `evaluation` CSV and heatmap settings (which can also be overridden by its CLI flags) and writes enabled artifacts inside the run's evaluation folder:

- **Failed Chain Targets Heatmap (`failed_chain_<update>_<steps>.png`):** Plots red dots for target positions where the shortest-chain relay to the base station could not be completed and held.
- **Evaluation Information CSV (`eval_info_<update>_<steps>.csv`):** Records every evaluated target position, its terminal stage (`not_found`, `visually_found`, `found_and_delivered`, or `chain_success`), and Euclidean target-to-base distance. Heatmaps are filtered from this CSV rather than saving separate point tables.
- **Found-and-Delivered Heatmap (`found_and_delivered_<update>_<steps>.png`):** Combines two key target-spawner failure metrics onto a single blueprint by default:
  - **Not Visually Found (Sky Blue BGR `(235, 99, 37)`):** Target coordinates that were never visually seen by any drone in the swarm.
  - **Visually Found, Not Delivered (Dark Blue BGR `(6, 119, 217)`):** Target coordinates that were successfully seen by a drone (and updated in `target_known`), but never successfully routed back to the base.
- **Top Padded Legend Layout:** All heatmaps utilize a 60px top margin to print title stats and visual color legends, ensuring the blueprint remains un-cluttered.
- **Validation Check:** During execution, a verification warning is printed to the console if the number of targets not visually found exceeds those not delivered.

## W&B Dictionary Reference
When training on Weights and Biases (`wandb`), the key metrics include:
| Metric | Description |
|---|---|
| `train/ep_return` | Mean unified team return across completed training episodes. |
| `train/success_rate` | Fraction of the batched episodes resolving with a full chain link. |
| `train/target_found_rate` | Fraction of environments where the swarm found the target. |
| `train/chain_progress_pct` | Mean chain progress percentage across completed training episodes. |
| `train/map_coverage_pct` | Mean explored-map coverage percentage across completed training episodes. |
| `ppo/policy_loss` | PPO clipped surrogate loss algorithm output. |
| `ppo/value_loss` | MSE accuracy estimate of the generalized Critic module. |
| `ppo/clip_fraction` | Fraction of PPO samples whose policy ratio was clipped. Useful for spotting overly aggressive recurrent updates. |
| `ppo/approx_kl` | Approximate KL divergence between old and updated policies. |
| `perf/sps` | Environment steps per second across the configured training batch (default 4,000 environments). |
