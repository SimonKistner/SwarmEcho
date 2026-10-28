# C relay curriculum

The C levels retain their historical reward blocks and are currently unsupported
by the signed individual reward system. Each YAML contains a TODO and a
commented default replacement immediately below its original reward block.
Migrate and retune that block before running a C level; see
[signed individual rewards](../01_reference/11_individual_rewards.md).

The sequence is C01 → C02 → C03 → C05a → C05b. C04 is intentionally skipped:
the station introduces vertical flight, then the existing cellar introduces
stairs. All stages start with one drone and can recruit up to their agent cap.
Policy, reward, drone radar, communication and optimizer settings follow B02cs1.
The historical C reward blocks include individual and shared useless-agent
costs in the former positive-magnitude convention. These settings are preserved
in the YAML for retuning, but do not describe the current runtime. The new
individual `useless_agent_at_success_reward` defaults to -3000. The separate
`useless_agent_at_success_all_agents_reward` defaults to -750 per useless drone
for each active drone, without dividing by population. Its absolute amount is
capped at 2000 per recipient by `useless_agent_at_success_all_agents_reward_cap`.
These current settings
are included in the commented replacement blocks.
Every C level uses `wall_sensing_layout: xyz`, `wall_sensing_rays: 26`, and
`wall_sensing_ray_max_distance: 8.0` m: six exact axis rays, twelve plane
diagonals and eight corner diagonals. XYZ also supports 14 (axes plus corners)
or 18 (axes plus plane diagonals). The global defaults are legacy Fibonacci
with 16 rays and 8 m range. Target visibility remains 4 m, and drone radar
remains 16 Fibonacci bins with three counting bands.

| Level | Map alternatives | Agent cap | Maximum steps | Evaluation episodes per map |
| --- | ---: | ---: | ---: | ---: |
| `C01_linear_hallway` | 6 | 5 | 300 | 64 |
| `C02_corner_hallway` | 24 | 5 | 400 | 32 |
| `C03_space_station` | 1 | 5 | 600 | 256 |
| `C05a_cellar` | 1 | 5 | 800 | 512 |
| `C05b_cellar` | 1 | 9 | 1000 | 512 |

## Geometry and target distances

C01 contains a sealed straight shaft in each signed X/Y/Z direction. Every
map has the same target band: some positions require two drones and some
require three. C02 contains every ordered pair of perpendicular signed
directions (6 × 4), with the same two-or-three-drone requirement. Names spell
out travel direction, such as `C01_x_positive` and
`C02_x_positive_then_z_negative`. Each C02 leg has four cells including the
shared corner: seven interior cells total. The base-to-bend centre distance is
15 m, equal to the outgoing leg's centre-to-centre length.

With the configured radii, an unobstructed n-drone chain can span at most
`8 + (n - 1) × 15 + 4` metres: **12 m with one**, **27 m with two**, and
**42 m with three**. The generator certifies entire target cells, including
their nearest and furthest corners, before the runtime applies wall buffers:

- C01 permits five cells, with full-cell base distances from 12.5 to about
  37.67 m. Every position exceeds the one-drone limit and lies within the
  three-drone limit. A straight visible chain provides the construction.
- C02 permits all three non-corner cells on the outgoing leg. The minimum
  Euclidean base distance is about 12.75 m. A first relay at the first leg's
  midpoint is 7.5 m from both base and bend. A second relay at the bend leaves
  19 m (one communication link plus vision) for the third drone to reach the
  target. The furthest target corner is about 17.85 m from the bend. This
  proves a three-drone construction and excludes a one-drone solution even
  with corner shortcuts. Near targets admit two drones; far targets need three.

`maps/C_curriculum/manifest.json` records these bounds and a 0.25 m safety
margin. Saved maps have tight bounds: C01 is 40 × 5 × 5 m rotated among the
axes; C02 is 20 × 20 × 5 m rotated among the planes. The training bank pads
coverage/geometry arrays internally for batching, while retaining each map's
physical bounds for collisions, radar and observation scaling. Evaluation and
inspector snapshots use the actual map dimensions. Existing replays keep their
saved geometry; rerun the preview command to inspect the revised maps.

C03 has a central cell, six spokes, face-centre crosses, and all cube edges.
Each spoke has three intervening hallway cells and reaches a face centre at
the fourth. Face centres branch in four directions, edge midpoints branch
in two, and the corners close the cube. The default seed creates 189 interior
cells and 71 passable walls. Each eligible internal X/Y wall has a 50% chance
of being inserted, then a 50/50 door/window choice. These choices happen once
at generation time and are reproducible. Vertical shafts remain open: the
map format supports door/window apertures only on X/Y walls.

C03 targets can use every interior cell, with runtime geodesic rejection
below **20 m = 15 m drone communication + 4 m visual + 1 m buffer**. The
multiplier is `20 / 15`, rather than 1.3. Evaluation and replays inherit the
same filter. C05 uses a geometry-preserving copy of
`B02c_static_map_cellar.yaml`, also with the 20 m filter, and samples both
floors. C05a and C05b share that geometry.

C01–C03 use the existing `minimal` roadmap with interior cell/portal nodes:
it represents vertical turns and uses only interior/portal anchors. Straight
shafts have no internal solid boxes because all walls lie on world bounds;
the roadmap still connects their eight cell-centre nodes. In C01 these
geodesic distances equal Euclidean distances. C05 retains B02cs1's full roadmap. The decommission sphere is 0.75 m
in C01–C03 so it fits above a centred base inside a 5 m shaft; C05 retains 1 m.

## Generate and train

The generator uses only the standard library and does not import the ML
runtime. Generated maps are already included. Regeneration overwrites the
same map names deterministically:

```bash
python src/swarmecho/curriculum_config/maps/scripts/C_curriculum_map_generator.py
```

The installed equivalent is `uv run swarmecho-generate-c-curriculum`.
`--seed` changes the station walls. Radius/cell-size options are available,
but changing them also requires corresponding level settings and updating
the C03/C05 separation multiplier. The certified minimum drone counts depend
on those radii.

Run the sequence in the existing WSL environment:

```bash
uv run swarmecho-curriculum levels=C01_linear_hallway,C02_corner_hallway,C03_space_station,C05a_cellar,C05b_cellar logging.run_name=C_curriculum
```

The current XYZ/26/8 m sensors require a compatible C checkpoint or fresh
training. To run a separate experiment initialized from the older B02cs1
weights, retain its original Fibonacci/16/4 m wall sensing by appending:

```bash
training.checkpoint_path=outputs/curr_cellarBuilding_B02cs1/checkpoints/ckpt_000540 training.ckpt_loading_mode=init env.wall_sensing_layout=fibonacci env.wall_sensing_rays=16 env.wall_sensing_ray_max_distance=4
```

Levels live under `levels/M/`, `levels/B/`, and `levels/C/`. Short CLI names
still work, as do grouped names (`level=C/C01_linear_hallway`) and old level
paths stored in previous runs. Dashboards and inspectors continue to use
saved run configs and geometry snapshots.

The map builder can open a generated stage folder through its existing option:

```bash
uv run swarmecho-maze-builder --map-dir src/swarmecho/curriculum_config/maps/C_curriculum/C01
```

## Progression and evaluation cost

C01/C02 use `evaluation.early_exit_source: training`. Each map maintains a
completed-episode history with as many samples as its assigned lanes. A gate
is checked only after every map has contributed a fresh window of samples.
Three consecutive gates must show **at least 95% success on every map**.
`early_exit_min_success_length_reduction: null` disables the completion-time
gate throughout C; successful episodes may use the full level horizon.
Repeated rollout updates cannot advance the hold using the same episodes.
Training action noise remains part of these measurements.

With 3,000 environments, C01 gets exactly 500 lanes per map; C02 gets exactly
125. There are no periodic metric evaluations in their configured training
budget (`eval_freq: 1000000`). Early exit or the last update still measures
all maps and writes their heatmaps: 384 episodes for C01 and 768 for C02.
Those final measurements are diagnostics, not an extra progression gate.
The 64 and 32 episodes per map are chosen budgets in
`evaluation.eval_parallel_envs`, not values derived from geometry. The smaller
C02 budget limits the cost of covering all 24 orientations. Scheduled replay
recording remains separate from metric evaluation and never advances the hold.

C03/C05 retain evaluation-based progression with two consecutive passing
evaluations: **95% success for C03/C05a**, **99% for C05b**, with no speed gate.
Periodic evaluations run every 50 updates (first at 51 with the default offset)
once rolling training success reaches `evaluation.eval_min_train_success: 0.7`.
This 70% gate is a cost-saving configuration choice, not the mastery threshold;
the final evaluation bypasses it. C03 measures 256 episodes, each C05 level 512.

Every C evaluation is a single deterministic pass without added action noise.
`training_robustness: false`, `final_robustness: false` and
`eval_action_noise_max: 0.0` disable periodic/final noisy ensembles, including
standalone parallel checkpoint evaluation with these level settings. Completion
reuses the ending metric pass for checkpoint heatmaps, without three extra runs.
Training action noise, if enabled independently, still affects training windows.

The existing curriculum runner advances when a stage exhausts its timestep
budget even if its early-exit threshold has not passed.

Every static suite map always contributes to metrics and heatmaps.
`evaluation.map_suite_replay_limit` only caps replay artifacts: C01 records
all six; C02 samples eight spread across the ordered suite. Null preserves
the previous all-map replay behavior. Replay cadence is 100 updates after
the same 70% training-success gate. Replays and heatmaps sample every valid
storey; no floor restriction is applied. When early exit ends a stage, its
stopping evaluation also writes the configured representative replay sample,
even if the periodic replay cadence has not arrived. This keeps the final
checkpoint inspectable for stages such as C02 that stop before update 101.

## Inspect the requested checkpoint on C maps

This command restores the policy once and creates short, inspectable replay
artifacts without parallel evaluation or CSV target selection:

```bash
uv run swarmecho-preview-c-curriculum --checkpoint outputs/curr_cellarBuilding_B02cs1/checkpoints/ckpt_000540 --wall-sensing-layout fibonacci --wall-sensing-rays 16 --wall-sensing-ray-max-distance 4
```

Those overrides match the old checkpoint and preview the C geometry with its
legacy sensors. Omit them when previewing a checkpoint trained with the current
XYZ/26/8 m C sensors. Checkpoint validation rejects incompatible ray layouts,
counts or ranges.

It writes six replays: two of six straight maps, two of 24 corner maps, the
station, and the nine-agent C05b cellar. C05a remains in the training sequence
and is omitted from the default preview. Targets are sampled from the level's
allowed volumes, using its geodesic filter where enabled. The default horizon
is 100 steps; frame zero already includes geometry, base and target positions.
Use `--all-maps` for all 32 default level/map cases, `--samples 4` for a larger sample,
or `--max-steps 1000` to run each level up to its configured horizon. A
comma-separated `--levels` limits the stages. Each invocation gets a fresh
`eval_C_geometry_*` folder beneath the checkpoint's `artifacts/eval/` tree.
All preview replays share one suite group, so the inspector shows one
`C_geometry` navigator entry with a separate map choice for each replay.
Each invocation creates its own group.

```bash
uv run swarmecho-inspect root=outputs/curr_cellarBuilding_B02cs1
```

The generator and its distance certificates were run on Windows. GPU
training, checkpoint restoration and replay execution need verification in
the project's WSL environment; run the preview command first to inspect them.
