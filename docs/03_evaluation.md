# 03 Evaluation and inspection

Validation measures the task on a defined set of cases and can select the best checkpoint or stop training. Inspection measures behavior on a smaller set and produces map-level diagnostics and media. The table shows how G03 uses these paths.

| Path | G03 success criterion | Cases and outputs |
| --- | --- | --- |
| Training | Hold the chain for 30 consecutive steps and decommission every active surplus drone | Training maps, rollout metrics and checkpoint artifacts |
| Validation | Hold the chain for 30 consecutive steps | 10,000 total episodes from 400 separate validation buildings, fixed across passes |
| Inspection | Hold the chain for 30 consecutive steps | Three buildings from the validation set with independent full-budget sampling and 1,000 episodes per map |

## Configure validation

`validation.num_maps` specifies how many generated buildings are reserved for validation. `validation.num_episodes` specifies the total number of cases across the set. The validator uses `validation.seed` to assign maps and reset keys. With `fixed_targets=true`, it saves and reuses the cases across passes. With `fixed_targets=false` on generated maps, later passes draw new cases. `validation.parallel_envs` controls how many cases run at a time, independently of the episode count.

The overall `validation/success_rate` weights each episode equally. The validator also reports success by estimated required-agent count and a map-balanced metric that weights map scores equally. Per-map results can be written under `validation/`. When `validation.save_best_checkpoint` is enabled, an improvement in the overall rate replaces `checkpoints/best_val_checkpoint`. Keep the recorded map assignment and case plan when comparing checkpoints on the same cases.

`validation.val_freq` counts PPO updates between passes. G03 runs it every 50 updates. `validation.val_on_finalization` schedules a final pass. The 75% rate on the exact current required-agent group advances G03's training curriculum. A separate full-task validation threshold controls early stopping when `evaluation.early_exit` is enabled. G03 sets that threshold to 99%. For a single authored building, `validation.mode=uniform_full_task` evaluates the same map on fixed solvable targets at the full budget. This mode requires one authored map, `validation.num_maps=1` and `validation.fixed_targets=true`.

[Validator and case handling](../src/swarmecho/training/validation.py) · [Training schedule](../src/swarmecho/training/train.py) · [Configuration constraints](../src/swarmecho/core/config.py)

## Configure inspection and media

Use the `inspection_evaluation` section or its alias `evaluation`. The inspection seed controls resets for that protocol. `num_agents` and `success_condition` can set the inspection fleet cap and goal independently of training. Null values inherit the training setup. Standalone overrides use the same settings.

`evaluation.target_sampling=independent` draws from the full task independently of the training sampler. G03 uses this to keep inspection comparable as the forward curriculum advances. The `training` inspection mode reuses adaptive cell weights on a compatible single authored training map. The `curriculum` inspection mode samples uniformly within the current budget of an adaptive target-spawn curriculum. It requires that curriculum and full-task validation.

`evaluation.eval_freq` schedules scalar inspection in PPO updates. `training_heatmap_creation` controls in-training heatmap creation, while `eval_video` and `eval_video_freq` control replay generation and scheduling. For supported random map suites, `final_artifacts_only` keeps media generation for finalization. G03 reports inspection metrics every 100 updates and writes final heatmaps and replays. `inspection_maps` controls the number of maps in the validation subset used for inspection. `eval_parallel_envs` sets the per-map episode budget for that G03 inspection suite.

Inspection heatmaps show where targets are never found, seen without delivery, delivered to base, or connected by a held chain. Replays expose trajectory and relay behavior for selected episodes. Use the heatmap to locate failure regions, then the replay to see how an individual attempt unfolded.

[Map-suite inspection](../src/swarmecho/training/random_evaluation.py) · [Inspection CLI and heatmaps](../src/swarmecho/training/evaluate.py) · [Artifact schema](../src/swarmecho/training/artifacts.py)

## Evaluate or inspect a checkpoint

Run these commands from the repository root in the project's Linux environment. This example evaluates the checkpoint from the example run in [training](01_training.md) on an authored open-cuboid map.

```bash
uv run swarmecho-evaluate checkpoint=outputs/g03_example/checkpoints/best_val_checkpoint eval_map=open_cuboid eval_samples=8
```

The evaluator reuses the checkpoint run's saved config and adapts it to independent inspection on the requested map. `eval_samples` sets the number of episodes per map in parallel mode. The policy selects actions deterministically across the sampled reset cases. Supply a name from the [authored map library](../src/swarmecho/curriculum_config/maps/custom_maps) or a path to your own building YAML. Quote map names and paths that contain spaces. This path works for maps outside the training pool.

Without `eval_map`, the CLI loads the level named in the checkpoint metadata when available, or a level you select explicitly. It can use the saved inspection suite. Pass `map_suite_map_id` to select a particular saved suite member. `eval_map` cannot be combined with `level` or `map_suite_map_id`. Keep the checkpoint's observation layout and model dimensions when specifying overrides. `replay_after=true` requests replays in addition to evaluation.

```bash
uv run swarmecho-inspect root=outputs
uv run swarmecho-maze-builder --port 8766
```

Training artifacts live under `outputs/<run>/artifacts/train/`. Standalone evaluation writes checkpoint-scoped artifacts under `outputs/<run>/artifacts/eval/`. The inspector reads those artifacts. The maze builder authors building YAMLs, which can be checked with `uv run swarmecho-validate-building <map.yaml>`.

[Evaluation CLI](../src/swarmecho/training/evaluate.py) · [Artifact paths](../src/swarmecho/training/artifacts.py) · [Inspector](../src/swarmecho/visualize/inspector.py)

## Share an inspector artifact

Use the inspector export command to build a self-contained public-inspector bundle from a saved artifact manifest.

```bash
uv run swarmecho-inspect export="path/to/artifact-manifest.json" name=page-name
```

See the [public inspector](../public-inspector/README.md) for the collection layout.
