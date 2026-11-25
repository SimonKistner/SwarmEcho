# SwarmEcho

SwarmEcho is a **JAX-accelerated multi-agent reinforcement learning** project. It trains a drone swarm to search 3D buildings and form a communication relay between a base and a discovered target. The policy uses recurrent MAPPO with a centralized critic during training and TarMAC message passing between drones. The local quick start uses the authored-building presets G01 and G02.

The demos show SwarmEcho’s progression from the early 2D implementation to the current 3D simulator.

<p align="center"><strong>Earlier 2D demo</strong></p>
<p align="center">
  <img src="docs/assets/Preview_vid/demo_2x.gif" width="600" alt="Early SwarmEcho 2D drone swarm demo" />
</p>

<p align="center"><strong>Current 3D demo</strong></p>
<p align="center">
  <img src="docs/assets/Preview_vid/demo_3d.gif" width="600" alt="SwarmEcho drones exploring a 3D building" />
</p>

---

## Architecture

The `src/swarmecho/` package separates simulation, learning, evaluation, and map authoring.

```text
SwarmEcho/
├── docs/                      Numbered system references
├── cluster/                   Determined submission, configuration, and result retrieval
├── src/swarmecho/
│   ├── core/                  Level loading and configuration
│   ├── curriculum_config/     Levels, building maps, and the building editor
│   ├── env/                   Geometry, physics, observations, and rewards
│   ├── models/                Recurrent MAPPO actor, critic, and TarMAC
│   ├── training/              Rollouts, PPO, checkpoints, validation, and evaluation
│   ├── analysis/              Cross-run analysis
│   └── visualize/             Replays and the spatial inspector
└── tests/                     Unit and workflow tests
```

---

## Installation

Use `uv` on x86-64 Linux, including WSL2 on Windows. The project requires Python 3.11 or newer and uses the CUDA 12 build of JAX. Run the Python commands from the repository root.

```bash
git clone https://github.com/M4GI3R/SwarmEcho.git
cd SwarmEcho
uv sync
```

---

## Quick Start

A **map file** is a YAML file that describes one building, including its layout, base position, and where targets can spawn. A **level file** is a YAML file that tells SwarmEcho which maps to use or generate and how to train and evaluate the swarm.

- **G01** trains on one map, [g01_static_map.yaml](src/swarmecho/curriculum_config/maps/custom_maps/g01_static_map.yaml), in the custom maps folder.
- **G02** trains on five maps in the [same folder](src/swarmecho/curriculum_config/maps/custom_maps).
- **G03** selects 3,600 training maps and 400 separate validation maps from a pool of randomly generated buildings.

### 1. Start training

G01 validates on a fixed set of solvable targets on its training map. This command saves the best validation checkpoint for the evaluation example below.

```bash
uv run swarmecho-train level=G01_static logging.run_name=g01_example logging.use_timestamp_postfix=false logging.wandb_mode=offline validation.save_best_checkpoint=true
```

Use `level=G02_static` for G02, or pass the path to your own level YAML. See the [map source options](docs/03_marl_and_training.md#buildings-and-target-curriculum) for how to select maps.

Pass `section.field=value` to override a setting for the run. If you change `training.num_envs`, make it a multiple of `training.num_minibatches`. The small rollout check in [Tests](#tests) is available before starting a full run.

The [cluster workflow](cluster/README/README.md) covers Determined submissions, cluster presets, seed batches, and fetching results.

### 2. Run multiple seeds

This launcher runs jobs sequentially with the same training settings and a different training seed for each run. `base_seed` determines the set of seeds so you can repeat it.

```bash
uv run swarmecho-multi-train level=G01_static seeds=3 base_seed=42 logging.run_name=g01_baseline logging.wandb_mode=offline
```

### 3. Edit a 3D building

The building editor opens on port 8766. Save a building YAML and pass its path to the building validator. The second command uses an included example.

```bash
uv run swarmecho-maze-builder --port 8766
uv run swarmecho-validate-building "src/swarmecho/curriculum_config/maps/custom_maps/custom_building.yaml"
```

### 4. Evaluate and inspect a checkpoint

Once the example run has saved a best validation checkpoint, evaluate it on an authored map. Replace `g01_example` with your actual run directory name if needed. `eval_samples` sets the number of episodes for the map.

```bash
uv run swarmecho-evaluate checkpoint=outputs/g01_example/checkpoints/best_val_checkpoint eval_map=open_cuboid eval_samples=8
uv run swarmecho-inspect root=outputs
```

The inspector opens on port 8765 and displays replays and heatmaps. Add `replay_after=true` to the evaluation command to request replays. The inspector’s **Export public page** button packages an artifact for browser-only sharing. See [evaluation and inspection](docs/04_validation_evaluation_and_inspection.md#evaluate-or-inspect-a-checkpoint) and the [public inspector guide](public-inspector/README.md).

### 5. Explore training runs

The dashboard provides cross-run analysis.

```bash
uv run swarmecho-dashboard
```

### 6. Start curriculum training

To train on a list of maps in order, give each map its own level file and pass those files in order. When early exit is enabled, a stage ends once validation success meets the configured threshold for the required number of consecutive validation passes. A stage also ends at its time or step limit. The next stage starts from the previous stage’s final checkpoint.

Replace the paths with your level files.

```bash
uv run swarmecho-curriculum levels=path/to/first_level.yaml,path/to/second_level.yaml logging.run_name=curriculum logging.wandb_mode=offline
```

For example, the list could start with [the G01 map](src/swarmecho/curriculum_config/maps/custom_maps/g01_static_map.yaml) and continue with [the mirrored office](src/swarmecho/curriculum_config/maps/custom_maps/mirrored_office.yaml). The office needs its own level file and a checkpoint with a path prediction grid large enough for it. The standard G01 grid is too small for training on the office.

---

## Documentation Directory

1. [System overview](docs/01_system_overview.md) introduces the task, maps, levels, configuration and implementation structure.
2. [Environment and physics](docs/02_environment_and_physics.md) explains geometry, deployment, observations, communication, success and rewards.
3. [MARL and training](docs/03_marl_and_training.md) explains the actor and critic, PPO, sampling curricula, path supervision and checkpoints.
4. [Validation, evaluation and inspection](docs/04_validation_evaluation_and_inspection.md) explains measurement protocols, diagnostics and artifacts.

---

## Output Structure

Training runs write to their own directories under `outputs/` by default. This layout includes the validation artifacts and best checkpoint enabled by the G01 quick start. Standalone evaluation adds checkpoint-scoped artifacts under `artifacts/eval/`.

```text
outputs/
└── {run_name}/
    ├── config.yaml             Resolved run settings
    ├── checkpoints/
    │   ├── ckpt_<update>/      Saved training checkpoint
    │   └── best_val_checkpoint/
    ├── validation/
    │   ├── cases.npz           Fixed cases reused across validation passes
    │   └── results_u*.json     Validation metrics and per-map results
    └── artifacts/
        ├── train/              Inspection artifacts written during or after training
        └── eval/               Standalone checkpoint evaluation artifacts
```

Each curriculum stage gets its own run directory. `logging.use_timestamp_postfix=true` adds a timestamp to a configured run name.

---

## Tests

Run the pytest suite from the repository root on Linux or WSL2.

```bash
uv run pytest -v
```

The [tests](tests/) cover level configuration, building geometry and target placement, drone deployment and observations, relay communication and rewards, PPO updates, checkpoint handling, validation, and evaluation artifacts.

The [G03 workflow tests](tests/test_g03_workflows.py) build a tiny temporary map bank and run one real PPO update. They check that training saves a checkpoint with a finite loss, that the checkpoint can be restored and evaluated on a saved inspection map, and that validation uses separate maps, saves fixed cases, and writes a best checkpoint. To run those checks on their own, use `uv run pytest -v tests/test_g03_workflows.py`.

There is also a standalone smoke test, `uv run swarmecho-validate`. It runs a small rollout and one PPO update, then checks that the update produces finite statistics. It needs no existing checkpoint and lives in [the training package](src/swarmecho/training/validate_workflow.py), separate from the pytest suite.
