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
