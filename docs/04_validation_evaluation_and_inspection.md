# 04 Validation, evaluation and inspection

Validation measures a defined case set and can select a checkpoint, advance target difficulty or stop training. Inspection measures behaviour on a smaller map suite and generates diagnostics and media. Standalone evaluation applies a checkpoint to selected maps. Their success criteria, sampling distributions and episode budgets determine what each reported score means.

| G03 path | Success criterion | Measurement population |
| --- | --- | --- |
| Training | Hold the chain for 30 steps and decommission every active surplus drone | Episodes drawn from 3,600 training buildings under the current target budget |
| Validation | Hold the chain for 30 steps | 10,000 full-task cases across 400 separate validation buildings |
| Inspection | Hold the chain for 30 steps | Three validation buildings, 1,000 independent full-task episodes per building |

## Validation protocol

### Maps and cases

`validation.num_maps` specifies the number of generated validation buildings. These are separate from training-map assignments. `num_episodes` is the total episode budget across the validation set. Seeded case generation chooses map IDs and reset keys, so the number of episodes per map can vary. `parallel_envs` controls execution batch size independently of that case budget. Padded execution slots are discarded from statistics. [Validator](../src/swarmecho/training/validation.py)

With `fixed_targets=true`, the validator saves and reuses its case plan across passes. The run's validation records include metadata, case arrays, result JSONs and history. Preserve the map assignment and case files to compare checkpoints on identical cases. With `fixed_targets=false` on generated maps, later passes sample new cases. Validation uses the full task budget throughout G03's forward curriculum.

`validation.mode=uniform_full_task` supports same-map validation for one authored building. It requires one authored map, `num_maps=1`, fixed targets and compatible fixed-base settings. Its targets are solvable under the full fleet budget. G01 uses this mode. A same-map score measures performance on new cases within that building. G03's separate-building protocol additionally measures transfer across its validation layouts. [Configuration constraints](../src/swarmecho/core/config.py)

### Metrics and checkpoint selection

`validation/success_rate` pools episodes with equal weight. The map-balanced rate gives equal weight to observed map scores. Per-map summaries include the distribution of observed map rates, and required-agent groups stratify outcomes by the relay estimator's difficulty label. A map with no cases has no empirical rate for that pass. Wilson intervals describe uncertainty of the pooled episode proportion under the sampling model. Variation across separately trained seeds requires per-seed results.

`save_per_map_stats` retains map-level detail. `save_best_checkpoint` replaces `checkpoints/best_val_checkpoint` when the pooled validation rate improves. Its state records the evaluated model and continuation information. Metric histories and case identities establish which validation pass produced that checkpoint. [Validation summaries and best-checkpoint handling](../src/swarmecho/training/validation.py)

`val_freq` counts PPO updates between passes, and `val_on_finalization` schedules a final pass. G03 uses every 50 updates. The validator's `success_condition` is independent of inspection's override. A null validation criterion inherits the environment criterion.

### Progression and stopping

G03's forward curriculum advances at 75% success on the exact current required-agent group. Full-task stopping uses `validation.early_stopping.enabled` with the overall validation rate and a 99% threshold in G03. [Training gate](../src/swarmecho/training/train.py)

`hold_evals` requires consecutive qualifying passes, with at least one pass used. G03 uses one. `min_success_length_reduction` can additionally require successful episodes to finish sufficiently below the horizon. `final_success_rate` supplies the full-task threshold used with the applicable curriculum. These gates require validation to be enabled. Training-rollout and inspection rates remain diagnostic measurements.

## Inspection protocol and scheduling

The `inspection_evaluation` section, also accepted as `evaluation`, configures inspection. `seed` controls its reset sequence. `num_agents` and `success_condition` can override the fleet cap and goal, with null values inheriting the training environment. This makes comparisons meaningful only when their criteria and fleet caps are recorded.

| `target_sampling` | Reset distribution |
| --- | --- |
| `independent` | Full task, independently of live training weights and curriculum budget |
| `training` | Adaptive target-cell weights on a compatible single authored training map |
| `curriculum` | Uniform sampling within the current adaptive target-spawn curriculum budget |

The latter modes require their corresponding controller state. Standalone inspection restores that state when needed and rejects missing state. Explicitly supplied targets remain fixed. G03 uses `independent`. [Inspection sampling](../src/swarmecho/training/inspection_sampling.py)

Generated-map inspection selects `inspection_maps` buildings from the saved validation suite. Authored suites use their configured inspection maps, and alternate evaluation maps can be selected explicitly. `random_eval` and its map list configure a separate inspection suite. G03 uses three fixed validation buildings. In its map-suite protocol, `episodes_per_map=1000` gives 1,000 episodes per map. Map-suite aggregate metrics average map scores, while successful-episode population summaries use their own success denominators. [Map-suite evaluation](../src/swarmecho/training/random_evaluation.py)

| Control | Effect |
| --- | --- |
| `eval_freq`, `eval_offset` | Schedule scalar inspection in PPO updates |
| `eval_min_train_success` | Delay scheduled inspection until rolling training success reaches a threshold, with finalization allowed to bypass the gate |
| `training_heatmap_creation` | Request intermediate evaluation heatmap artifacts |
| `eval_video`, `eval_video_freq`, `eval_video_offset` | Enable and schedule replay recording |
| `final_artifacts_only` | Retain intermediate metrics and postpone media to finalization in the supported suite workflow |
| `map_suite_replay_limit` | Limit maps receiving replays while retaining metrics across the suite |
| `checkpoint_freq`, `checkpoint_offset`, `save_model` | Schedule periodic checkpoint saving |
| `replay_targets_from_storey` | Restrict replay target selection to a 1-based storey |

G03 runs scalar inspection every 100 updates, leaves `eval_min_train_success=0`, disables intermediate heatmaps and creates final media. G02 retains intermediate heatmaps. These schedules count updates, so changing rollout sizes changes environment steps between measurements. [Scheduling](../src/swarmecho/training/train.py)

## Training diagnostics and interpretation

Training success uses a rolling completed-episode window tied to the environment count. Per-required-agent statistics maintain separate windows controlled by `logging.required_agents_metrics_window_episodes`. A rare difficulty group can therefore describe a different span of training history from the overall metric. Always identify the success criterion and denominator when comparing these curves. [Training metrics](../src/swarmecho/training/train.py) · [Deployment metrics](../src/swarmecho/training/decommissioning_metrics.py)

Episode return sums individual reward contributions over drones and time. A larger total contribution can result from repeated per-step rewards, more recipients or longer episodes. Its magnitude alone does not determine its influence on policy gradients. Recorded reward breakdowns help connect return changes to events such as delivery, hold success or decommissioning.

Information-source diagnostics classify first knowledge acquisition using direct sight, base and peer conditions. They measure the Boolean propagation events in the simulator. Learned message content needs separate decoding or intervention evidence. [Information metrics](../src/swarmecho/training/information_metrics.py)

`logging.adaptive_cell_diagnostics` controls adaptive-cell histories and reports. `logging.individual_map_diagnostics` controls individual-map traces. Disabling these diagnostics leaves the sampler state needed for operation and checkpoints. The adaptive-spawn dashboard reads recorded histories and exposes sampling mass, outcomes and plateau warnings. Its analysis entry point is documented by the [tool source](../src/swarmecho/analysis/adaptive_spawn_dashboard.py).

When adaptive-cell diagnostics are enabled, `adaptive_spawn/metadata.json` describes the cells and `history.jsonl` appends one CPU-side record after each update. Records distinguish the probabilities used during that rollout, measured end-of-rollout outcomes and the weights prepared for the next rollout. The dashboard supports timelines, per-cell curves and live refresh. Resume markers let it follow a checkpoint rewind without treating superseded records as a new uninterrupted trajectory. [Journal](../src/swarmecho/training/adaptive_spawn.py)

The general run dashboard compares saved configurations and telemetry across runs. Roadmap scans, bank-coverage checks, random-map generation and spawn-pair feasibility checks inspect preparation without a policy-training experiment. They use the same building and relay contracts as training. `swarmecho-benchmark` measures environment execution, while timing logs distinguish compilation, rollout/transfer, optimization and final artifact costs. [Run dashboard](../src/swarmecho/analysis/dashboard.py) · [Analysis tools](../src/swarmecho/analysis) · [Benchmark](../src/swarmecho/training/benchmark.py)

### Auxiliary and frozen-policy diagnostics

The online auxiliary head reports binary cross-entropy, overlap and thresholded path-mask metrics. `training.geo_probe_auxiliary.metric_threshold` sets the cutoff for binary mask metrics, including exact equality of the complete mask. Soft overlap and BCE use the continuous prediction. G03 uses a cutoff of 0.8 over its 125-cell decoder grid. Report eligible and successfully labeled samples alongside scores, since missing reference labels change the population measured. Updates without valid supervision do not provide a fresh auxiliary fit measurement. [Auxiliary metrics](../src/swarmecho/training/geo_auxiliary.py)

Thresholded overlap is Dice similarity, `2 * intersection / (predicted_cell_count + reference_cell_count)`. Soft overlap replaces the binary prediction with probabilities. Exact-mask accuracy requires every cell to match the reference mask, so a single false positive or missing cell makes that snapshot inexact. These measures describe the chosen teacher route and coordinate convention.

`gradient_diagnostics` samples PPO and auxiliary gradient diagnostics on the configured diagnostic schedule. These help assess the relative gradients into the actor representation. Validation measures the head on its evaluation samples without optimizing it.

Frozen-policy probes fit separate readouts from saved actor states or messages. Their collection splits, target labels, map distributions and normalization define a separate experiment. Target-cell localization, own-cell localization and whole-path prediction answer different questions. Shared-coordinate probes can pool maps, while map-specific readouts can exploit a known map identity. See the [geometry experiment documentation](../geo_experiments/README.md) for the maintained protocols. Historical binary-sighting and map-specific results are indexed under [removed observation experiments](ablations/README.md).

## Evaluate or inspect a checkpoint

The [README](../README.md#4-evaluate-and-inspect-a-checkpoint) gives launch commands. `swarmecho-evaluate` accepts `checkpoint`, an optional `eval_map` and an episode count through `eval_samples`. `eval_map` accepts an authored-map name or a building YAML path, including maps outside the training pool. This mode loads the run's saved configuration and applies independent inspection on the selected building. [Evaluation CLI](../src/swarmecho/training/evaluate.py) · [Argument handling](../src/swarmecho/training/evaluate_cli.py)

Without `eval_map`, evaluation selects the checkpoint's named level when available or an explicitly supplied level and can use the saved inspection suite. `map_suite_map_id` selects a saved suite member. `eval_map` cannot be combined with `level` or `map_suite_map_id`. `eval_samples` sets episodes per map in parallel evaluation, and actions are deterministic across the sampled resets. `replay_after=true` requests replay artifacts as well.

Checkpoint observation and model dimensions must match. A different building can have a new layout or physical extent while using the same local observation contract. The auxiliary decoder retains its trained grid, so a map change must also respect that grid when requesting reference-path scores. Fresh models derive their grid from the configured maps. Quote map paths that contain spaces.

Training artifacts are under `outputs/<run>/artifacts/train/`, and standalone evaluation uses checkpoint-scoped folders under `artifacts/eval/`. The inspector discovers artifact manifests and displays replays, heatmaps and validation summaries. The building editor produces map YAMLs, and `swarmecho-validate-building` checks those files. The test suite and CPU workflow smoke test are described in the [README tests section](../README.md#tests). [Artifact paths](../src/swarmecho/training/artifacts.py) · [Inspector](../src/swarmecho/visualize/inspector.py) · [Validation viewer](../src/swarmecho/visualize/validation_view.py)

### Heatmaps and replay evidence

Evaluation CSVs retain per-episode target positions and outcomes. Heatmaps distinguish targets never found, seen without base delivery, delivered without held-chain success, and successful under the selected goal. Stage-rate views can show cumulative discovery, delivery and success separately. A replay records the episode trajectory, connectivity, events and reward components for its selected target. Difficulty-stratified replay selection helps inspect rare groups that an aggregate rate can hide. [Evaluation outputs](../src/swarmecho/training/evaluate.py) · [Replay recorder](../src/swarmecho/visualize/replay.py)

The inspector's reference path is a geometric route derived from the map and endpoints. It provides context for the observed flight trajectory. Its cache depends on the source geometry and planner inputs, and unavailable routes are marked accordingly. Reward panels use recorded per-drone components to explain the episode's return. [Replay paths](../src/swarmecho/visualize/replay_path.py)

The local inspector groups artifacts by producing run, measurement event and map. Saved suites use stable IDs such as `static_eval_0000` and `random_eval_0000`, letting a selected heatmap point generate a replay command for the correct building. Manually named evaluations are identified by `eval_name`. Stars mark individual map/artifact choices. Timeline and playback controls navigate recorded frames, and building visibility controls isolate storeys, walls and roof. Reward debug plots start disabled and can be enabled under Playback.

The checkpoint-configuration panel reads training-map selection and agent counts from the producing run and actual artifact metadata. It describes the producing stage, and unavailable values appear as Unknown. `outputs/<run>/.cluster/inspector.json` can supply a local `display_name` alias while the artifact retains its underlying run identity. Categorical outcome CSVs and historical robust-rate CSVs remain readable, with XYZ coordinates required. [Inspector](../src/swarmecho/visualize/inspector.py) · [Metadata extraction](../src/swarmecho/visualize/inspector_metadata.py)

## Share an inspector artifact

The inspector's export dialog packages the selected replay or heatmap for browser-only viewing. The equivalent CLI accepts a replay JSON manifest or heatmap CSV in `export`, a lowercase hyphenated `name`, and optional title, section, tags, output location, replacement and reference-path settings. The [public inspector README](../public-inspector/README.md) covers the collection and publication workflow.

Exports contain viewer data, geometry and compressed payload chunks, plus optional reference-path and thumbnail assets. The exporter strips private manifest fields to its public field allowlist. Gallery metadata records section, tags and UTC publication/update times. Replacing an existing recognized export preserves its original publication time. The public viewer supports viewing the exported artifact and links back to the collection. Local run discovery and command generation are available through the local inspector. [Export implementation](../src/swarmecho/visualize/public_export.py)

### Export options and publication

| CLI option | Meaning |
| --- | --- |
| `export=...` | Replay JSON manifest or evaluation heatmap CSV to package |
| `name=...` | Required page name, lowercase letters/numbers with single hyphens, up to 80 characters |
| `title=...` | Display title, up to 160 characters |
| `reference_path=true` | Reuse or prepare a replay reference route during export, enabled by default |
| `overwrite=true` | Replace an existing recognized export with that name |
| `output=...` | Alternate collection root, with publication configured separately |
| `section=...` | Gallery section, created as needed |
| `tags=...` | Comma-separated tags, at most 12 tags of 32 characters each |
| `thumbnail=...` | Optional local PNG of at most 1 MiB |

The UI captures the current view for its optional thumbnail and prepares reference routes in a background worker. A failed route calculation still permits exporting the artifact with a visible unavailable-path status. Command-line export finishes after writing the selected bundle. Replacement preserves existing section/tags when omitted, while `tags=` clears tags. Omitting the thumbnail on replacement removes the old preview.

The collection contains `catalogue.json`, shared Plotly assets and a directory per replay or heatmap. Each page has an HTML viewer, a compressed-data index and chunks of at most 8 MiB, export metadata and optional thumbnail/reference path. Cards and sections are ordered by their latest relevant publication date. The viewer loads data over HTTP and uses browser WebGL and decompression support. Local preview is available through the inspector's exported-page link or an HTTP server for the collection directory.

The repository's `Publish public inspector` workflow uploads `public-inspector/` and deploys from the default branch. It is triggered by changes to that directory or the workflow and can be dispatched manually. Configure the repository's Pages publishing source to use this Actions workflow, then use the normal commit/push process to publish selected export files, the catalogue and updated shared assets. The deployment job reports its page URL. Existing exports are snapshots, so renderer changes require re-exporting affected pages. Removing a page also requires removing its catalogue entry, or rebuilding the catalogue through the next export. [Deployment workflow](../.github/workflows/public-inspector-pages.yml)

## Experimental records

Use the artifacts of an experiment for claims about its configuration and results. Local output folders may need to be supplied separately to another checkout.

The five-seed group `CLU_G03_baseline_val_50` corresponds to Determined experiments 57005–57009 and seeds 456789–456793. The local `outputs/CLU_G03_baseline_val_50_seed<seed>/cluster_run.json` receipts record seed-specific pools, training-seed map selection and validation every 50 updates. Numerical claims should use the logged per-seed curves and resolved configurations with their validation case records.

Historical experiment 55489, `CLU_G03_forward_aux_l0p3_eff25_seed456789_decom3000_branch_v3`, recorded 95.8% chain-holding validation success at 1.191 billion cumulative environment steps, including ancestor stages. It branched from `ckpt_002520`. Its archived configuration and map selection describe that historical experiment.

The original best checkpoint expects 141 inputs. Its separate `best_val_checkpoint_140obs` copy drops the always-zero vote input from the actor and critic encoder matrices and associated optimizer entries. The conversion report records CPU model/optimizer restoration and a 16-step recurrent inference comparison. The 95.8% score belongs to the original training record. [Historical resolved config](../outputs/000_best_run_so_far/config.yaml) · [Validation state](../outputs/000_best_run_so_far/checkpoints/best_val_checkpoint/validation_state.json) · [Step history](../outputs/000_best_run_so_far/checkpoints/best_val_checkpoint/step_history.json) · [Conversion report](../outputs/000_best_run_so_far/checkpoints/best_val_checkpoint_140obs/checkpoint_conversion.json)

The [ablation index](ablations/README.md) points to removed-mechanism experiments, including reverse target curricula and older observation regimes. Current feature definitions remain in the numbered references and live source.
