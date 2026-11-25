# 03 MARL and training

## Recurrent MAPPO

SwarmEcho uses centralized training and decentralized execution with shared actor parameters. Each drone maintains its own recurrent state, receives a local observation and learned messages, and produces three movement commands. The critic processes the team jointly during training and estimates a separate value for each drone's reward stream. This supports individual reward attribution while sharing an experience-generating policy across the fleet. [MAPPO model](../src/swarmecho/models/mappo.py)

### Actor and TarMAC

The actor encodes the local observation with linear layers, LayerNorm and tanh, combines the encoded features with received communication context, and updates a GRU. Its movement head produces a Gaussian mean and learned log standard deviation for each axis. The implementation clips log standard deviations to [−5, 2], samples unbounded commands during training, and tanh squashes them for the environment. Deterministic evaluation uses the policy's mean command. [Actor](../src/swarmecho/models/actor.py) · [Action distribution](../src/swarmecho/models/actions.py)

TarMAC gives each sender a learned signature and value. A recipient derives a query from its recurrent state, compares it with available signatures, and uses attention weights to mix their values. The radio mask admits active peers within range and line of sight, excluding the recipient itself. Signatures and values come from the previous step, which makes message timing explicit. The base cache supplies an additional sender for eligible uninformed recipients in direct base contact.

Attention uses a softmax over query–signature dot products scaled by the square root of signature width. With no valid sender, the context is zero. Identical values from several otherwise identical senders can average to the same context as a single sender, so packet aggregation alone need not reveal multiplicity.

G03 uses `hidden_dim=256`, three actor layers, signature width 32 and value width 64. `actor_num_layers` controls actor depth, while `tarmac_sig_dim` and `tarmac_val_dim` determine the sizes of transmitted vectors. Changing these dimensions changes model and checkpoint shapes. The actor observation width is derived from the enabled [sensors](02_environment_and_physics.md#observations).

### Centralized critic

The critic has its own observation encoder and per-agent GRU. Tokens combine observation features with recurrent memory. Multi-head attention mixes information across the active team, and a shared value head produces one scalar per agent. G03 uses three critic layers and the same hidden width of 256. The critic can condition a drone's estimate on other agents' observations and memory. Its inputs follow the model's observation and recurrent-state contract. Simulator coordinates, full maps and reference-path labels reach the learning system through their separately defined uses in observations, rewards or auxiliary supervision. [Critic](../src/swarmecho/models/critic.py)

The critic uses four attention heads and excludes self from cross-agent attention while retaining the drone's own token for its value head. Shared token processing makes the value rule permutation equivariant, so renaming agents renames their outputs. A per-agent estimate can condition on relevant teammates while matching that agent's reward stream. Recurrent state persists between rollout segments, while direct gradient propagation follows the replayed segment.

## Rollout collection and PPO

An update collects `num_steps` transitions in each of `num_envs` independent environment lanes. Team slots stay together, and each slot has observation, action, log probability, value, reward, active mask and recurrent state. Rollouts also record reset boundaries and the writer identities needed to reconstruct base-memory updates. [Rollout loop](../src/swarmecho/training/train.py) · [Buffer](../src/swarmecho/training/mappo_buffer.py)

For an active agent, generalized advantage estimation uses temporal-difference errors `delta_t = reward_t + gamma * value_(t+1) - value_t` and discounts their future contributions by `gamma * gae_lambda`. Terminal transitions cut bootstrapping. Decommissioning and inactive masks also cut returns, keeping separate slot lifetimes independent. The buffer uses the terminal flag belonging to the transition being evaluated. A rollout boundary can bootstrap from its final critic estimate when the episode continues. Episode horizons are terminal boundaries.

PPO retains each lane's time order and team context, shuffles lanes, and divides them into `num_minibatches`. `num_epochs` repeats optimization over the collected rollout. G03 collects 50 steps in each of 6,000 lanes, or 300,000 environment transitions per update, then makes three epochs with six minibatches per epoch.

The policy objective uses the ratio of new to stored action probabilities and the clipped surrogate with `actor_clip_eps`. Stored pre-squash actions allow consistent Gaussian probability ratios, with the common tanh change-of-variables factor cancelling. Advantages are normalized over valid rollout entries. Entropy uses a fresh reparameterized sample and the tanh Jacobian correction for the squashed policy. [PPO objectives](../src/swarmecho/training/ppo.py)

The critic uses cumulative running mean and variance statistics for value targets. Statistics update once per rollout and stay fixed during its PPO epochs. Rewards and GAE remain in their raw units. `value_clip_eps` limits critic changes in normalized value units, and setting it to zero clamps changes to the stored prediction. `vf_coef`, `ent_coef`, `lr` and `max_grad_norm` respectively weight value loss, weight entropy, set the Adam learning rate and bound gradient norm. The auxiliary decoder has its own optimizer. [Value normalization and optimizer state](../src/swarmecho/training/ppo.py)

Recurrent replay begins with the recorded state and propagates through the rollout. Base-cache writer indices reconstruct writes from the corresponding actor outputs, allowing the replay to connect gradients through those writes within the sequence. Episode resets and inactive masks clear the relevant state.

### Execution size, recovery and budgets

`rollout_batch_size` chunks rollout execution across lanes. `optimizer_microbatches` divides one PPO minibatch into execution chunks, accumulates their appropriately weighted gradients and applies one optimizer step. Normalization statistics and clipping follow the logical minibatch. These controls reduce peak memory at a fixed scientific batch. G03's OOM recovery increases optimizer microbatching on retry. [Execution planning](../src/swarmecho/training/ppo_execution.py) · [Memory execution](../src/swarmecho/training/memory_execution.py)

The retained OOM strategies can also change `num_envs`, `num_steps` or `num_minibatches`. Such retries change collection or optimization grouping and must be recorded when comparing runs. `oom_recovery.factor`, `decrement` and `max_retries` control the retry sequence. The launcher runs training in a worker and recovers from saved state when retrying. `profile_timing`, terminal initialization logs and `diagnostics_every` control diagnostic overhead and reporting. [Launcher](../src/swarmecho/training/launcher.py) · [Recovery lifecycle](../src/swarmecho/training/run_lifecycle.py)

`total_timesteps` counts environment transitions across lanes, independently of fleet size. `max_training_seconds` starts after preparation and includes first compilation, optimization and in-loop measurement. `max_wall_seconds` includes setup. The applicable limit ends the training loop when checked, and final saving or media can extend elapsed process time. G03 uses one billion environment steps and 24 hours of training time. These limits are separate from validation-based success stopping.

## Buildings and target curriculum

### Map sources and compiled banks

`training.num_maps` controls the number of persistent training buildings, while `training.num_envs` controls concurrent episode lanes. G03 uses 3,600 maps across 6,000 lanes. Generated-map resets select from the persistent set, so several lanes can use the same building. Authored suites distribute lanes over the listed maps, and adaptive map sampling can reassign maps as episodes reset. [Map selection](../src/swarmecho/training/map_selection.py) · [Training setup](../src/swarmecho/training/train.py)

| Configuration | Runtime preparation |
| --- | --- |
| `random_buildings.enabled=true`, `load_maps_from_bank=true` | Select a compatible compiled pool or generate one when automatic selection finds none |
| `random_buildings.enabled=true`, `load_maps_from_bank=false` | Generate and prepare buildings directly without loading a compiled pool |
| `random_buildings.enabled=false` | Load authored YAMLs from `env.map_names` |

An explicit `training.map_pool_path` pins a pool and requires it to pass compatibility checks. Automatic selection supports fresh pools and new random maps. `save_training_maps` optionally writes selected/generated layouts to the run. Authored suites require `training.num_maps` to be unset or match the listed maps. Map selection and compiled geometry are retained for checkpoint recovery. [Bank generation](../src/swarmecho/curriculum_config/maps/random_building_bank/generate_bank.py)

Compiled pools require shared cell size and wall/tile thickness. Map arrays are padded to a common envelope for batched simulation while retaining each building's own bounds. Larger geometry and denser roadmaps increase preparation and device-memory costs, including shortest-path matrices. Compatibility checks include geometry settings and source fingerprints, so changes to compiler sources can require recompilation. [Building bank](../src/swarmecho/env/building_bank.py)

Pool generation appends unique map records under a lock and uses a bounded CPU worker pool. The default worker count is capped at four, with an explicit worker override in the bank-generation tool. Preparing a large map collection can be substantial work before learning starts. Persistent map identities keep geometry, exclusions, support histograms and checkpoint selection aligned. The unselected remainder of a pool stays on disk. [Pool preparation](../src/swarmecho/curriculum_config/maps/random_building_bank/generate_bank.py)

### Relay difficulty and target eligibility

The global relay estimator constructs legal candidate drone positions, connects positions with valid radio links, and searches for the shortest chain joining a base and target sighting. The reported required-agent count includes the finder. Candidate positions combine seeded uniform samples with legal roadmap and launch anchors. `candidate_spacing_fraction` controls the random budget as `ceil(world_volume / (cell_size * fraction)^3)`. `max_candidates_per_map` rejects an excessive preparation request explicitly. [Candidates](../src/swarmecho/env/relay_candidates.py) · [Relay estimator](../src/swarmecho/env/relay_global.py)

The estimator checks body reachability and communication visibility for their respective roles. Its minimum count is conditional on its finite candidate graph. It provides a geometric feasibility estimate and difficulty label. It does not predict whether the policy will discover and hold that chain within the episode horizon.

`training.target_spawn_eligibility.enabled` prepares empirical per-cell support at the full fleet budget. `samples_per_map` counts accepted solvable probes, and `minimum_cell_fraction` removes cells whose accepted support is too small. G03 retains cells with at least eight of 800 accepted probes. The cached histograms also organize targets by required-agent group. This support estimate is a sampling-based filter. The live reset sampler still enforces the requested target budget. Fixed bases are required for this prepared eligibility. [Eligibility preparation and persistence](../src/swarmecho/training/target_spawn_eligibility.py)

### Forward target curriculum

`training.forward_target_curriculum.enabled=true` begins with targets requiring at most two agents. Validation success of at least 75% on the exact current required-agent group advances the budget by one until the configured fleet cap is reached. Generated-map validation needs a nonempty exact group. Same-map validation can instead gate on the hardest available group within the current budget and advance to the next available group, avoiding a stall on absent difficulty classes. All fleet slots remain available while target difficulty changes. Independent validation and inspection can cover the full task throughout training. [Forward controller](../src/swarmecho/training/target_curriculum.py)

Setting this toggle to false opens the full fleet budget immediately. With adaptive samplers also disabled, target sampling is uniform over the allowed volume after any configured eligibility restrictions. The forward controller requires validation and excludes simultaneous adaptive target or map sampling. G03 also enables prepared eligibility to restrict supported cells. The 75% progression threshold is separate from the run's full-task stopping threshold.

### Success-weighted cell and map sampling

Both adaptive samplers remain supported and are disabled in G03. `adaptive_target_spawn` keeps recent completed-episode outcomes for each eligible target cell, while `adaptive_map_sampling` keeps outcomes for each map. These are reset-time distributions, so an update changes future episode draws. G01 and G02 demonstrate authored-map uses. [Cell controller](../src/swarmecho/training/adaptive_spawn.py) · [Map controller](../src/swarmecho/training/adaptive_maps.py)

Let `b_i` be the baseline probability, `s_i` the recent success rate, `d_i = 1 - s_i`, and `m = max(d_i)`. For cells, `b_i` is proportional to eligible free volume. For maps, it is uniform. When failures exist, the failure distribution is `q_i = b_i * d_i / sum(b_j * d_j)`. With `f = min_probability_fraction`, the desired probability is `f*b_i + (1-f)*((1-m)*b_i + m*q_i)`. When all entries succeed, the desired distribution is the baseline.

The retained baseline share keeps successful cells and maps in circulation, while the remaining mass favours failures. `probability_smoothing_half_life_updates` smooths toward the desired weights, with update rate `1 - 2^(-1/half_life)`. Success windows retain the configured number of completed episodes per cell or map. Within a selected cell, free boxes receive probability proportional to their volume.

Adaptive map sampling first fills its outcome windows across maps, accounting for episodes already in flight, before failure-based weighting starts. Cell-support discovery has its own `eligibility_samples_per_map` preparation budget. Live lane counts and discovery samples serve separate purposes. Cell adaptation requires supported fixed-base setups. Compatibility checks reject combinations whose distributions or saved state cannot be reconstructed.

`adaptive_target_spawn.initial_target_agent_budget` retains a separate target-budget curriculum usable with authored or generated maps. It uses validation outcomes from the exact current required-agent group and the configured success, hold and optional completion-speed gate. A group needs validation cases to advance. Its inspection can use the current budget. G01 uses this facility, while G03 uses the dedicated forward controller. Plateau warnings compare cell success progress over a configured update horizon and improvement threshold. They produce diagnostics. The sampler continues running. [Adaptive curriculum](../src/swarmecho/training/adaptive_spawn.py) · [Constraints](../src/swarmecho/core/config.py)

## Auxiliary geodesic path prediction

The auxiliary decoder reads actor GRU state and predicts a binary mask of building cells crossed by a reference collision-free base-to-target geodesic. Labels use a base-centred frame and a canonical choice among four horizontal rotations. The output grid covers the cell offsets required by the maps used when the model is constructed, including rotated extents. G03's 3 × 3 × 3 buildings therefore use a 5 × 5 × 5 output grid, with horizontal offsets −2 through +2. Other map sizes derive their grid automatically. An existing checkpoint retains its decoder shape. [Path labels and grid derivation](../src/swarmecho/analysis/path_targets.py)

Canonicalization compares the transformed solids, interior cells, base and target to resolve orientation and base-cell boundary ties consistently. The planner chooses a deterministic reference route among equal alternatives, then rasterizes its traversed cells. Thus the teacher specifies one path mask in a scene-derived coordinate frame. Different valid physical routes can receive different masks.

Frames are eligible while the drone is active and knows the target. `snapshots_per_agent` caps supervised frames per drone in a rollout. The sampler divides eligible time steps into chronological bands and draws one distinct frame per band, giving coverage across the rollout. `sampling.every_updates` sets which outer PPO updates can supply supervision. Unavailable reference labels are omitted from the auxiliary sample set. PPO can still use those transitions. [Supervision and sampling](../src/swarmecho/training/geo_auxiliary.py)

The head uses binary cross-entropy and its own Adam learning rate and gradient clipping. Its gradient into the actor is scaled by the schedule's lambda. Warmup and ramp count updates with valid supervision. G03 trains the head while keeping actor auxiliary gradient zero for 10 supervised updates, then ramps to `lambda_target=0.3` over 100 updates. The scaled representation gradient is combined with PPO's actor gradient. `lambda_target=0` trains a detached decoder using the same path. The configurable hidden sizes and learning rate control decoder fitting independently of the policy optimizer.

Supervised weights give equal total mass to represented episodes, then equal mass to represented agents within each episode and snapshots within each agent. This prevents a longer informed trajectory from dominating solely through more sampled frames. Labels are cached by map, base and target. Missing sparse-roadmap reference routes are recorded under `geo_probe_auxiliary/missing_reference.jsonl`, even when the denser relay candidate graph certified the target as feasible.

The path target is privileged supervision. The movement head still selects actions from recurrent actor features. The auxiliary prediction is used for loss and diagnostic metrics. Online decoder performance reflects both the changing policy representation and decoder training. Frozen-policy probe studies answer a different question by fitting readouts after policy training. [Metrics and interpretation](04_validation_evaluation_and_inspection.md#auxiliary-and-frozen-policy-diagnostics)

## Fresh starts, checkpoints and reproducibility

`training.seed` controls training randomness. `map_selection_seed=null` follows that seed, while an explicit selection seed fixes map assignment across training seeds. `seed_map_pool=true` additionally keys the generated pool by training seed. G03 leaves `map_pool_path` and `checkpoint_path` null so a fresh launch can select or generate a compatible pool. It sets an explicit map-selection seed of 456789 by default. [Selection and seed handling](../src/swarmecho/training/map_selection.py)

| Loading mode | Intended continuation |
| --- | --- |
| `resume` | Continue an interrupted run, restoring optimizer state, counters, map selection and matching controller state |
| `branch` | Start another run from compatible checkpoint weights and inherited progress |
| `init` | Initialize weights with a fresh run timeline |

Checkpoint metadata validates observation semantics and model dimensions. A checkpoint carries the actor, critic, auxiliary head and value-normalization state, with optimizer and controller records needed for continuation. Recovery requires compatible map identities and settings. The mode and controller-specific restore checks determine which state can be inherited. Reset episodes begin fresh after loading. [Checkpoint contract](../src/swarmecho/training/checkpoints.py) · [Lifecycle](../src/swarmecho/training/run_lifecycle.py)

Observation changes need particular care. The historical best checkpoint has a separate converted 140-input copy that drops an always-zero retired input and its associated encoder optimizer entries. The [experimental record](04_validation_evaluation_and_inspection.md#experimental-records) identifies that copy and its conversion evidence. New models use the current observation function.

## Sequential level curricula and cluster tools

The curriculum launcher accepts a list of levels. Each stage runs its configured training task, then carries compatible weights to the next stage. Validation-based early exit can finish a stage at its configured threshold. Step or time limits can also end a stage before that threshold is reached. This facility supports user-defined sequences such as G01 followed by an office map. Map changes must respect model and decoder compatibility. [Sequential curriculum](../src/swarmecho/training/curriculum.py) · [Command in the README](../README.md#6-start-curriculum-training)

Multi-seed execution launches sequential runs with the same configuration and a seed override. Cluster infrastructure adds source snapshots, submission records and retrieval around these training entry points. It also retains specialized tuning tools for bounded PPO-parameter searches. Their worker assumptions and fixed overrides need checking for a chosen task before reuse. The [cluster guide](../cluster/README/README.md) and [tuning implementation](../src/swarmecho/training/cluster_tuning.py) describe that separate workflow.
