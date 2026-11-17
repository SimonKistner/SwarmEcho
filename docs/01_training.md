# 01 Training

## Start a run

Run commands from the repository root on Linux or WSL2. The project uses Python 3.11 or newer and declares the CUDA 12 build of JAX. Install the project with `uv sync`, then choose the level explicitly. CLU G03 is the default level and the generated-building reference used throughout this guide.

```bash
uv sync
uv run swarmecho-train level=G03_CLU_random_buildings logging.run_name=g03_example logging.wandb_mode=offline
```

A **map** is one building the drones can explore. Its **map file** defines the geometry, base position, and cells where targets may spawn. A **level** specifies a training task. Its **level file** selects or generates maps and defines settings such as swarm size, rewards, model dimensions, PPO, and validation.

Set `level=` to a level name or the path to its YAML file. A level that uses authored maps lists their names or paths in `env.map_names`. A level that uses generated maps configures `random_buildings` instead. Use the same map in different levels to compare training settings, or use several maps in one level to train one policy across buildings.

Use `section.field=value` to override settings for the run. Missing values come from the configuration schema. Use `logging.wandb_mode=online` for remote logging or `disabled` to turn W&B off. The run’s `config.yaml` records the resolved settings, including launch overrides. Keep it with the source revision and the saved map selection to reproduce an experiment.

The [G03 preset](../src/swarmecho/curriculum_config/levels/G/G03_CLU_random_buildings.yaml) supplies a complete generated-building example. For training on hand-authored maps, start with [G01](../src/swarmecho/curriculum_config/levels/G/G01_static.yaml) or [G02](../src/swarmecho/curriculum_config/levels/G/G02_static.yaml). G01 uses one map and G02 uses five. Both read their YAMLs from [maps/custom_maps](../src/swarmecho/curriculum_config/maps/custom_maps). Adjust the level to suit the experiment using the [environment guide](02_environment.md), and see [evaluation](03_evaluation.md) to use a checkpoint on a new map.

[Configuration parser and defaults](../src/swarmecho/core/config.py) · [Training launcher](../src/swarmecho/training/launcher.py)

## Buildings and target curriculum

G03 trains on 3,600 compiled random buildings and validates on 400 separate buildings. `training.num_maps` controls the size of the training set, while `training.num_envs` controls the number of concurrently simulated episodes. With uniform map sampling, episode resets can select the same map across multiple lanes.

Choose the map source according to how you want to control the layouts.

| Configuration | What the trainer does |
| --- | --- |
| `random_buildings.enabled=true`, `load_maps_from_bank=true` | Selects a compatible compiled bank or generates one if automatic selection finds none. A fixed `training.map_pool_path` pins the bank and must pass compatibility checks. The run and its checkpoints retain the map assignment. Changing geometry or settings used during bank compilation can require a different bank. `random_buildings.save_training_maps=true` copies the selected layouts into the run directory. |
| `random_buildings.enabled=true`, `load_maps_from_bank=false` | Generates the buildings and prepares their simulation data without using a precompiled bank. With `random_buildings.save_training_maps=true`, the generated layouts are written as individual map YAMLs. |
| `random_buildings.enabled=false` | Loads the building YAMLs listed in `env.map_names`. For authored training, leave `training.num_maps` unset or make it match the number of listed maps. |

The global relay estimator builds a set of legal candidate positions and checks line of sight and radio or sight range along links. The shortest candidate chain gives the estimated number of drones needed for a base–target pair. Candidate density affects the cost of preparing each map and which routes the estimator can find. When `training.target_spawn_eligibility.enabled` is on, startup probes build per-cell histograms of feasible target pairs. `samples_per_map` controls the accepted probe budget, and `minimum_cell_fraction` filters cells with too little accepted support.

The forward curriculum starts with targets requiring at most two drones. Each time validation success on targets in the exact current group reaches 75%, the training budget advances by one, up to the configured fleet size. All configured drone slots remain available throughout training. Validation and independent inspection can sample the full target budget throughout the run. Setting `training.forward_target_curriculum.enabled=false` opens the full target budget from the start. With adaptive samplers also disabled, target resets use uniform sampling within the eligible volume.

`training.adaptive_target_spawn` can give more weight to target cells with recent failures. `training.adaptive_map_sampling` applies that idea to map selection. They track completed episodes and keep a uniform probability share so successful cells or maps can still recur. Their window lengths determine how much outcome history feeds the weights, and the smoothing controls how quickly new estimates change the sampling probabilities. Both samplers are off in G03, and the forward curriculum requires both to be off.

[Map preparation](../src/swarmecho/curriculum_config/maps/random_building_bank/generate_bank.py) · [Relay estimator](../src/swarmecho/env/relay_global.py) · [Eligibility](../src/swarmecho/training/target_spawn_eligibility.py) · [Target curriculum](../src/swarmecho/training/target_curriculum.py)

## Policy and PPO

G03 uses recurrent MAPPO with centralized training and decentralized execution. The actor encodes its local observation and combines it with the TarMAC context before its GRU. The centralized critic keeps separate per-agent recurrent state and attends across active agents’ observation and memory tokens to estimate one value per drone. Runtime actions use the actor. The `network.hidden_dim` setting changes shared layer width. `actor_num_layers` and `num_layers` set actor and critic depth respectively. TarMAC compares learned signatures to compute attention weights and mixes the corresponding value vectors into a message. `tarmac_sig_dim` and `tarmac_val_dim` change those vector widths. These settings affect checkpoint shapes.

Each update collects `num_steps` transitions in each of `num_envs` lanes. PPO splits the resulting rollout into `num_minibatches` across environment lanes, retaining time order and team context within a sequence. `num_epochs` controls how many optimization passes reuse that rollout. G03 collects 50 steps in each of 6,000 lanes, then makes three passes with six minibatches per pass. Base-cache writes are replayed in the recurrent computation using the recorded writer indices and episode resets.

The buffer computes one generalized advantage estimate per agent, masks inactive slots, and cuts returns across episode ends and decommissioning. `gamma` discounts future rewards, while `gae_lambda` controls how far future temporal-difference errors contribute to each advantage. `actor_clip_eps` bounds the policy ratio used in PPO's clipped objective. `value_clip_eps` controls the permitted value change in normalized critic units for the clipped value objective. Entropy is computed for the squashed action distribution, and running value statistics scale the critic's targets.

`rollout_batch_size` chunks the simulation work across device lanes. `optimizer_microbatches` splits one PPO minibatch into chunks and accumulates gradients before one optimizer step. These control peak memory at the configured rollout and minibatch sizes. Changing `num_envs`, `num_steps` or `num_minibatches` changes how much data is collected or grouped for optimization. G03's OOM recovery increases optimizer microbatching on retry.

`total_timesteps` caps environment steps across lanes. PPO counts each environment transition once regardless of fleet size. `max_training_seconds` starts after setup and includes first compilation and in-loop evaluation. `max_wall_seconds` includes setup as well. Final saving and media generation can run after the time limit.

[Model](../src/swarmecho/models/mappo.py) · [Critic](../src/swarmecho/models/critic.py) · [Buffer](../src/swarmecho/training/mappo_buffer.py) · [PPO](../src/swarmecho/training/ppo.py) · [Optimizer execution](../src/swarmecho/training/ppo_execution.py)

## Auxiliary geodesic path prediction

A separate decoder reads the actor GRU state and predicts a binary mask of building cells crossed by the reference collision-free base-to-target geodesic. Labels use a base-centred frame with a canonical choice among the four horizontal rotations. The output grid fits every possible cell offset for the maps used when the model is constructed, including rotated extents. An occupied horizontal span of three cells requires five output offsets from −2 to +2. G03 therefore uses a 5 × 5 × 5 decoder grid for its 3 × 3 × 3 buildings. Other map dimensions produce different grids automatically. The grid is part of the decoder's model shape, and an existing checkpoint retains its trained output shape. The mask supplies auxiliary supervision and diagnostics. The actor selects actions directly from its recurrent representation.

`sampling.snapshots_per_agent` limits labeled frames per drone per rollout. The sampler divides eligible time steps into chronological bands, then draws one frame per band. A frame is eligible while the drone is active and knows the target. `sampling.every_updates` controls how often a rollout contributes labels. The head uses its own optimizer and binary cross-entropy loss. Warmup and ramp count supervised updates. In G03 the decoder learns from the beginning of supervision while the auxiliary gradient into the actor stays at zero for 10 supervised updates. This gradient then ramps to `lambda_target=0.3` over 100 more. Setting `lambda_target=0` keeps the decoder training on detached actor features.

[Path labels and grid](../src/swarmecho/analysis/path_targets.py) · [Auxiliary decoder and sampling](../src/swarmecho/training/geo_auxiliary.py)

## Fresh starts and checkpoints

A fresh G03 launch has `training.checkpoint_path=null` and `training.map_pool_path=null`. By default the preset uses seed 456789 and an explicit map-selection seed of 456789. With `training.map_selection_seed=null`, the training seed supplies the selection seed. `training.seed_map_pool=true` also keys the map bank by the training seed.

Set `training.checkpoint_path` to the checkpoint directory and choose a loading mode. Use `training.ckpt_loading_mode=resume` to continue an interrupted run with its optimizer state, step history, map selection and matching controller state. The same bank and compatible settings are required. Use `branch` to start a new run from a checkpoint and inherit compatible progress. Use `init` for weights only. Checkpoint metadata guards model shape and observation semantics.

[Configuration schema](../src/swarmecho/core/config.py) · [Run lifecycle](../src/swarmecho/training/run_lifecycle.py) · [Checkpoint contract](../src/swarmecho/training/checkpoints.py)
