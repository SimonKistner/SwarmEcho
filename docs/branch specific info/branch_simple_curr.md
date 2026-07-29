# `simpler_curr` Compared with `main`

The branch mainly adds a new hard-gated adaptive target-spawn curriculum, plus supporting environment, diagnostics, visualization, and finder-path fixes.

## New Configuration Parameters

The following parameters were added to `env` in `src/core/config.py`:

| Parameter | Default | Effect |
|---|---:|---|
| `adaptive_target_spawn_mode` | `"soft_gate"` | Selects the legacy probability-based curriculum or the new `"hard_gate"` curriculum. |
| `adaptive_spawn_success_lower` | `0.0` | In hard-gate mode, removes the newest active path category when aggregate training success remains below this value. |
| `adaptive_spawn_success_upper` | `0.8` | Adds the next path category when aggregate training success remains at or above this value. |
| `adaptive_spawn_threshold_hold_updates` | `3` | Number of consecutive adaptive evaluations for which a threshold must hold before the curriculum changes. |

The following parameter was added to `logging`:

| Parameter | Default | Effect |
|---|---:|---|
| `train_target_spawn_heatmap` | `false` | At each evaluation interval, saves a heatmap of target positions from the most recently completed training rollout. It also generates one at the final update. |

The parameter validation happens when `AdaptiveTargetSpawnController` is constructed: the mode must be `soft_gate` or `hard_gate`, thresholds must satisfy `0 <= lower <= upper <= 1`, and the hold count must be at least one.

## Hard-Gated Adaptive Spawning

Target cells are categorized by their shortest four-neighbor maze path distance from the base. Categories are sorted from easiest or nearest to hardest or farthest, and hard-gate mode begins with only the first category active.

At each adaptive update:

- Success is aggregated across all currently active categories.
- If success stays at or above the upper threshold for the configured hold count, the next category becomes active.
- If success stays below the lower threshold for that long, the newest category is removed.
- At least one category always remains active.
- Target positions are sampled uniformly across valid positions in the active categories.
- Threshold counters reset whenever performance returns between the thresholds or a category transition occurs.

Adaptive updates occur every `ceil(max_steps / num_steps)` PPO updates. For M04, that is `ceil(700 / 100) = 7`; with a hold count of three, a category change requires the condition to persist across three adaptive checks, or roughly 21 PPO updates.

The old `soft_gate` behavior remains the global default. It keeps every category available but adjusts per-cell probabilities according to relative category success, assigning more probability to categories performing below the overall average.

## Dynamic Environment Gating

Hard-gate mode makes inactive curriculum regions physically inaccessible:

- The perimeter edges of inactive, target-spawn-valid maze cells are converted into temporary wall segments.
- These walls affect movement, visibility, coverage raycasting, and communication line of sight.
- Existing wall segments are deduplicated.
- Cells that were already excluded from target spawning are deliberately not enclosed.
- Target positions too close to temporary walls are removed according to the map's `target_wall_clearance`; configuration fails clearly if that leaves no valid active positions.

The environment factory now accepts two internal inputs:

- `extra_walls`, which injects temporary category walls into map rasterization.
- `exploration_reward_mask`, which prevents exploration rewards from being earned inside inactive valid target cells.

The coverage map still records visibility everywhere, and cells excluded only by ordinary no-spawn rules remain exploration-reward eligible.

Whenever the active category set changes, the runner rebuilds and JIT-compiles the training environment with the new walls and reward mask. Evaluation continues using the normal ungated environment.

## Diagnostics and Visual Tools

Adaptive diagnostics now report:

- Current and previous active categories.
- A per-category active or inactive indicator in W&B.
- Category transitions in stdout, such as `[1, 2] -> [1, 2, 3]`.
- Existing per-category success, configured probability, actual spawn percentage, and count metrics.

The map preview tool gained:

- `--show-cell-categories`, which labels cells with their BFS path category.
- `--block-categories`, supporting specifications such as `4+`, `4-6`, or `2,4-6`.
- Rendering of temporary walls in PNG, SVG, video, and GIF previews.
- Spawn sampling and simulation against the same temporary walls shown in the preview.

## Training Spawn Heatmaps

When `logging.train_target_spawn_heatmap` is enabled, the runner uses target coordinates from the latest training rollout rather than an accumulated historical list and writes a green-dot heatmap under the training artifacts directory. The heatmap also displays the currently generated category walls, so the sampled curriculum region can be checked visually.

## Finder-Path Fix

The global finder path previously froze only on the exact step when target knowledge reached the base. If that transition happened before renderer-relevant state was sampled, the path could remain invalid permanently.

The branch now falls back to any preserved target-known path after the base already knows the target. This keeps the discrete finder-path reward state and evaluation highlighting available in resumed or delayed-render trajectories.

## M04-Specific Configuration Changes

In `src/curriculum_config/levels/M04_tiny_grid_maze.yaml`:

- Hard-gate settings are declared as lower `0.0`, upper `0.8`, and hold count `3`.
- `adaptive_target_spawn` itself remains `false`, so the feature is configured but not currently enabled for M04.
- The curriculum success threshold increases from `0.90` to `0.99`.
- Training spawn heatmaps are declared but remain disabled.
- Evaluation video frequency changes from `50` to `9,999,999`, effectively suppressing periodic videos while preserving final-run rendering behavior.

In the M04 map, three cross-shaped target exclusion rectangles were removed, leaving only the central `20-30 x 20-30` exclusion zone. This expands the set of target-spawn-valid cells that the adaptive curriculum can manage.

Finally, `.codex/` was added to `.gitignore`.
