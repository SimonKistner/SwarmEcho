# Configuration & Curriculum Guide

SwarmEcho uses a modular, layered configuration ecosystem. Default values are defined directly in Python dataclasses, and custom settings or difficulty levels are layered on top via YAML files.

## 1. Parameters & Where to Find Them
Rather than a global YAML file, the baseline configurations are declared inside the structured JAX/Flax-compatible dataclasses in `src/core/config.py` (specifically `SwarmEchoConfig`). The configuration structure is separated into semantic domains:

| Domain | Description / Usage |
|---|---|
| `env` | World physics, drone radius, vision limits, total agent count, and map name. |
| `reward` | Scaling coefficients for exploration, chain gaps, and success bonuses. |
| `training` | Central RL config (learning rates, PPO clipping ratios, entropy coefficient scaling, GAE parameters `gamma`/`lambda`). |
| `network` | Hidden layer dimensions, number of layers, activation types. |
| `logging` | Interval lengths for weights & biases syncs, evaluative video renders, and spatial failure heatmaps. |
| `visualize` | Video and preview rendering controls (DPI, alpha settings, rendering backends). |

The recurrent MAPPO extension is controlled from `network`:

| Key | Default | Effect |
|---|---:|---|
| `actor_memory` | `false` | Replaces the feed-forward actor with a per-agent GRU actor. |
| `critic_memory` | `false` | Replaces the agent-centric critic with a per-agent GRU critic before cross-agent attention. Requires `critic_type: agent_centric`. |
| `memory_comm_enabled` | `false` | Enables TarMAC communication for the recurrent actor. Requires `actor_memory: true`. |
| `memory_comm_every_k_steps` | `5` | Applies wall-aware actor communication every k environment steps. Non-communication steps mask agent-agent and base replay messages. |
| `tarmac_sig_dim` | `64` | Query/signature dimension used for TarMAC sender addressing. |
| `tarmac_val_dim` | `128` | Value/message dimension used for TarMAC communicated payloads. |
| `tarmac_include_self` | `true` | Keeps each agent's own previous signature/value in its attention sender set, allowing it to ignore incoming messages when useful. |

When recurrent actor communication is enabled, SwarmEcho uses a single-round TarMAC-style mechanism rather than the older configurable hidden-state attention path. Agents carry GRU hidden state plus previous `(signature, value)` communication state; the base relay stores the first target-knowing reporter's emitted TarMAC signature/value and replays that token to eligible non-knowing agents in base range.

### 1.1 Heatmap and Video Evaluation Toggles
Under the `logging` domain, the following parameters control mid-training and evaluation-time artifacts (stored under `outputs/.../videos/train/` and `outputs/.../videos/eval/`):

| Key | Default | Description / Effect |
|---|---:|---|
| `eval_video` | `true` | Renders a rollout video for evaluation episodes. |
| `eval_failed_chain_heatmap` | `true` | Generates a heatmap mapping target positions for episodes where the shortest chain was not successfully completed/held. |
| `eval_not_delivered_or_visually_found_heatmap` | `true` | Generates a heatmap mapping target positions for episodes where target was not successfully delivered or not visually found. |
| `eval_not_deliv_not_visual_splitt_in_two` | `false` | If true, splits the not-delivered/not-visual heatmap into two separate files instead of one merged heatmap. |

These heatmaps dynamically use the actual count of completed episodes in the sliding window as the denominator. Heatmap and video filenames are prefix-synchronized using a shared timestamp prefix (`YYYY_MM_DD_hh_mm`).

### 1.2 Visualize Settings & Selective Rendering
The `visualize` domain defines the parameters for both evaluation videos and map previews:

| Key | Default | Description / Effect |
|---|---:|---|
| `train_eval_renderer` | `"fast"` | Renderer backend used for mid-training rollout videos (`"fast"` or `"slow"`). |
| `final_eval_renderer` | `"fast"` | Renderer backend used for final / standalone evaluation (`"fast"` or `"slow"`). |
| `selective_eval_render` | `false` | If `true`, enables selective bucket rendering (filtering episodes by outcome instead of rendering all). |
| `eval_render_videos` | `1` | Number of episodes to render in legacy mode (when `selective_eval_render` is `false`). |
| `eval_max_compute_episodes` | `100` | Hard ceiling on episodes simulated when `selective_eval_render` is `true`. |
| `eval_render_successes` | `0` | Number of successful episodes to render in selective mode. |
| `eval_render_failures` | `3` | Number of failed episodes to render in selective mode. |

Under selective rendering (`selective_eval_render: true`), the pipeline runs up to `eval_max_compute_episodes` rollout environments but only exports videos for `eval_render_failures` failed runs and `eval_render_successes` successful runs. Both can be set to `0` to completely skip video rendering while still computing performance statistics.

Additionally, when `eval_not_delivered_or_visually_found_heatmap` is enabled, the pipeline merges "Not Visually Found" (sky blue) and "Not Delivered" (dark blue) target coordinate groups into a single combined heatmap `_merged_targets_heatmap.png` by default (unless `eval_not_deliv_not_visual_splitt_in_two = True`), drawing title statistics and legends in a clean 60px top padding.

## 2. Curriculum Overrides & Level Structure
The training system scales difficulty sequentially via the `levels/` directory.
- The structured defaults in `src/core/config.py` act as the overarching default configuration.
- Level override files are located in `src/curriculum_config/levels/` and act as "patches". When a run scales to a new level, the configuration parameters from the level YAML overwrite the base definitions.
- **Level Naming Conventions**:
  - **A-Series (Hand-designed Levels)**: E.g., `A00_open_field.yaml` and `A01_warehouse.yaml` define manually curated maps and agent counts.
  - **B-Series (Auto-scaling Levels)**: E.g., `B02_comm50_N5.yaml`, `B03_comm50_N6.yaml`, etc. These scale agent counts, communication radii, and spawn properties dynamically.
- **Dynamic B-Series Generation**: If a B-series YAML file or its corresponding map is missing when the training runner starts, the curriculum loader calls `generate_b_curriculum` to generate the level YAML and the square map representation automatically on the fly.
- Example: Turning off the target task in early levels to foster raw spatial awareness before enforcing the full chain-relay task.

## 3. Map Geometry (`src/curriculum_config/maps/`)
Maps are fundamentally defined via YAML layouts (rooms, hallways, walls) which are then rasterized into boolean occupancy grids for simulation.
- **The Map Builder:** A Streamlit-based UI to visually place geometric elements. (`src/curriculum_config/maps/scripts/map_builder.py`)
- **Validation Tools:** We bake and prep the maps using scripts inside `src/curriculum_config/maps/scripts/` to ensure full compatibility with JAX raycasting logic.

### Map Gallery Preview
The curriculum dynamically scales by transitioning through these layouts:

| Level | Map Layout | N Agents | Description |
| :---: | :---: | :---: | :---: |
| **A00** | `open_field` | 8 | Open space, target and base are placed. |
| **A01** | `warehouse` | 15 | Bulky obstacles (crates), requires coverage exploration. |
| **B02** | `square_N5` | 5 | Square boundary box containing 5 agents. |
| **B03+** | `square_N6+` | 6+ | Scaling agent counts and arena dimensions. |
