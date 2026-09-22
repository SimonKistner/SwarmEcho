# configuration

`load_level` and `load_level_cli` in `core/config.py` load a YAML
level and apply OmegaConf `key=value` overrides. Fields are validated against
the dataclasses. Unknown fields or sections fail explicitly.

| Section | Definition / purpose |
| --- | --- |
| `env` | `EnvConfig`: motion, perception, spawning, coverage, and obstacles |
| `reward` | `RewardConfig`: relay objectives and reward weights |
| `training` | `TrainingConfig`: PPO, noise, rollout sizes, and checkpoint loading |
| `network` | `NetworkConfig`: actor/critic memory, architecture, and TarMAC |
| `evaluation` | `EvaluationConfig`: schedules, robustness, maps, and artifacts |
| `logging` | `LoggingConfig`: run names, output roots, W&B, and terminal output |

There is no visualization configuration section in a level. Inspector controls
belong to the independent inspector.

## Levels and maps

Levels live in `src/swarmecho/curriculum_config/levels/`; maps live in the
adjacent `maps/` directory. `env.map_names` must contain exactly one map.
Use a level name or an existing YAML path:

```bash
uv run swarmecho-train level=B01a_office_find_only logging.run_name=office
```

The supported M-series names are `M00_no_maze_open_cuboid`,
`M01_no_maze_open_cuboid_tall`, and `M02_random_cuboid_obstacles`.
The B-series names are `B00_test`, `B01a_office_find_only`, and `B01b_office`.

Building cell size controls authoring geometry. `env.coverage_voxel_size`
controls coverage resolution and must divide every world dimension; null uses
the building cell size. Optional actor observation flags change network input
width, so they must agree with a loaded checkpoint.

## Training and checkpoints

Recurrent training requires `training.num_envs` divisible by
`training.num_minibatches`. The timestep budget must cover at least one rollout.
Production actor and critic memory must both be enabled.

`training.checkpoint_path` loads a checkpoint. `ckpt_loading_mode=resume`
continues run accounting, `branch` carries cumulative progress into another run,
and `init` loads weights with fresh accounting. Use compatible architecture,
observation settings, critic type, and normalization.

See [PPO controls](../01_reference/05_ppo_controls.md) for clipping, entropy,
normalization, and privileged critics. Training action perturbations use
`training.training_noise` and `training.noise_level`.

## Evaluation

Evaluation can use a different map through the existing
`evaluation.eval_differes_from_training_map` and `evaluation.eval_map`
fields. The spelling is retained for existing configuration compatibility.

`evaluation.training_robustness` enables robust periodic metrics.
`eval_robustness_runs` and `eval_action_noise_max` control robustness.
Standalone parallel checkpoint evaluation uses the robust evaluation path.

Evaluation, replay, and checkpoint schedules each have frequency and offset
settings. The retained `eval_video`/frequency fields schedule replay
artifacts. `training_heatmap_creation` controls evaluation heatmap artifacts.
Early-exit settings gate training on evaluation success, optionally requiring
a hold period.

## Reward defaults

The defaults now live directly in `RewardConfig`:
delivery required; Euclidean gap shaping; exploration 0.25; collision 0.5;
finder 50; maximum gap 5; target found 100; success 500; idle termination
-1000. Optional redundancy and efficiency rewards are disabled by default;
the efficiency bonus is 0.5. Level overrides take precedence.

The cleanup does not change any of these values or any bundled level.
