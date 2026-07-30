# `simpler_curr` Compared with `main`

The remaining branch-specific changes concern finder-path state preservation
and M04 evaluation settings.

## Finder-Path Fix

The global finder path previously froze only on the exact step when target
knowledge reached the base. If that transition happened before
renderer-relevant state was sampled, the path could remain invalid permanently.

The branch now falls back to any preserved target-known path after the base
already knows the target. This keeps the discrete finder-path reward state and
evaluation highlighting available in resumed or delayed-render trajectories.

## M04-Specific Configuration Changes

In `src/curriculum_config/levels/M04_tiny_grid_maze.yaml`:

- The evaluation early-exit success threshold is `0.99`.
- Evaluation video frequency is `9,999,999`, effectively suppressing periodic
  videos while preserving final-run rendering behavior.
