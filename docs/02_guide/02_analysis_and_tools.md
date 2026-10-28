# Analysis and tools

## General run-analysis dashboard

```bash
uv run swarmecho-dashboard
```

The Streamlit dashboard in `analysis/dashboard.py` remains the general
cross-run configuration and telemetry browser, including runs. Its code
and primary launch behavior are preserved. Historical media browsing remains
available where a run contains media.

## Adaptive spawning dashboard

```bash
uv run swarmecho-spawn-dashboard root=outputs port=8767 open=false
# Or restrict discovery to one producing run:
uv run swarmecho-spawn-dashboard root=outputs/G01_adaptive_spawn
```

Open `http://127.0.0.1:8767` in a browser. This independent dashboard requires
neither W&B nor video generation. Its timeline, previous/next controls and Play
button step through training. Cells are sorted by Manhattan distance to the base
with consistent colours and horizontal alignment across probability and success
bar charts. A large plot shows thin per-cell probability histories through the
selected update; clicking a cell's legend highlights its line. Individual
probability plots follow in three columns with the same ordering. The dashed
reference in each detail plot is that cell's original volume-based probability.

Each enabled training run writes `adaptive_spawn/metadata.json` and an append-only
`adaptive_spawn/history.jsonl`. One small CPU-side record is saved after each PPO
update using outcomes already transferred for the training buffer; there are no
callbacks, disk writes or additional device-to-host transfers within the JAX
rollout. This is more frequent than evaluation and avoids repeatedly rewriting
the full history. Records include **smoothed probabilities applied during the
selected rollout** and **per-cell window success measured at its end**. The next
rollout's computed probabilities are also retained in the journal.

The page polls incremental records every two seconds, with auto refresh and
follow-latest controls. Browsing an earlier update remains stable during refresh.
Incomplete trailing writes remain invisible until completed; interrupted malformed
records are skipped with a visible count. Writes are closed per update for live
visibility, without forcing a costly disk `fsync` every time. Resume markers retain
the original journal but roll the displayed history back to the selected checkpoint
before adding the new session. Metadata and checkpoint state use atomic replacement.

## inspector

```bash
uv run swarmecho-inspect root=outputs
uv run swarmecho-inspect root=outputs/RUN port=8765 open=false
```

The inspector discovers replay manifests and evaluation heatmaps. It provides
building visibility controls, replay inspection, target selection, and a
command for generating a replay at the selected XYZ position. Existing spatial
artifacts and compatibility fallbacks remain supported.

The artifact picker initially loads the three most recent runs. **Load 10 more
runs and tests** adds the next ten unloaded run/test groups, ordered by their
newest artifact, or the remaining groups if fewer than ten are available.
Each group includes all its artifacts. The button remains available until the
inventory is exhausted; refreshing the list keeps the loaded group limit.
Loading more preserves the selected artifact, camera, and playback position.

The replay sidebar places **Episode** on the left, beside **Checkpoint
configuration** and **Building visibility** stacked on the right. The
configuration panel automatically shows the producing run's training map
type/count, training agent count, and the actual evaluation agent count;
unavailable values show **Unknown**. The public inspector uses the same panel.
**Reward debug plots** start disabled in both inspectors; enable the checkbox
under **Playback** when needed.
New artifacts start at 10° elevation with two 15% zoom-in steps applied to the
starting camera distance. Artifact switches reset the camera, while refreshing
the same selected artifact preserves its current pose.

For a multi-map static run such as `B02c_static`, the inspector groups each
evaluation event's heatmaps and replays by map and shows the configured map
names. The saved evaluation copies live in `static_eval_maps/` and use IDs such
as `static_eval_0000`. A selected heatmap target uses
`map_suite_map_id=static_eval_0000` in its replay command. Standalone parallel
checkpoint evaluation visits every frozen static map; selecting one ID evaluates
or replays only that map.

In the artifact picker, the star beside a map dropdown applies to the map
currently shown in that dropdown. Replays and heatmaps can be starred
independently for each random or static evaluation map. Starred maps show a
star in the dropdown and mark their run in the run list. The horizontal map row
also has a star beside each map, so maps can be starred without opening the
artifact picker; those stars apply to the selected heatmap or replay category.

Evaluations launched with an explicit `eval_name` are highlighted in purple as
manual evaluations. The run list shows a **MANUAL** tag when it contains one;
the artifact picker and horizontal step row show the evaluation name, such as
`9_agents`. Default evaluations without `eval_name` keep the regular styling.

After a parallel checkpoint map-suite evaluation finishes and saves its
artifacts, the terminal prints a completion summary with the number of maps,
episodes per map, and aggregate success rate.
Checkpoint evaluation artifacts use the accumulated step count recorded in the
checkpoint history, so branched checkpoints retain their full training-step
label in the inspector.

For a replay, the frame row beneath the artifact and map selectors contains
previous/next frame buttons, a timeline slider, and the current/last frame
count. Select that row with Up/Down, then use Left/Right to step through frames;
the playback controls retain Play, time, speed, and loop. The Episode summary
is above Building visibility in the left sidebar. **Base votes** appears above
**Episode**, with **Carried votes** immediately below **Episode**.

The separate **Vote visualization** box below **Building visibility** has a
**Show vote icons** checkbox. Active drones show a plain ➕ for YES and a plus
with a red prohibition overlay for NO. Names and icons use world-space anchors
relative to each drone. Physical clearance selects above, below or a side;
rotating the camera never changes that placement. Displayed walls, floors,
roofs and obstacles cover only the portions of a name or icon behind them,
including partial views at edges and through openings. Building transparency
also applies to this occlusion; hidden storeys do not block labels. Turning off
vote icons keeps the drone names. Names and icons have no hover popup. Hover a
drone dot to see its **Vote round ID** and any carried votes; empty or unavailable
carried-vote records are omitted from this popup.

The 3D camera keeps its current angle, pan, and zoom when playing or stepping
through replay frames, including the first mouse adjustment after loading.
Drag to orbit and scroll to zoom while playback continues; frame arrows also
work during a drag when the frame row is selected. Orbiting keeps the building
upright: dragging changes azimuth and elevation, including views from below,
without rolling the building onto its side or upside down. The Camera elevation
slider remains available during playback. **Auto rotate** uses its own animation clock,
with the speed set in degrees per real second independently of replay speed.
Dragging temporarily takes over from auto rotation; releasing resumes rotation
from the adjusted view.

The replay sidebar also has a **Reference roadmap path** switch. Turning it on
calculates the shortest visibility-roadmap route from the first recorded base
position to the first recorded target position. A spinner appears while a
separate worker process computes the route, so playback, frame controls, and
artifact selection remain usable. The route is drawn like a roadmap inspectable;
**Path waypoints** shows only vertices on that route. Switching the route off
does not discard the result. The inspector saves it next to the replay manifest
as `*.roadmap-path.json`, so later sessions load it from disk. Replacing the
replay invalidates that cache.

New replays record the planner settings used for the route. Older replays use
default settings for missing values and say so beside the route length. Replays
without a building snapshot use the current map file. If positions or obstacles
change during an episode, the reference stays tied to the first frame and the
sidebar labels it as such.

Reward debug plots are off by default and can be enabled in Playback. Leave them off to
give the 3D view the full centre area. The left plot shows cumulative reward for
every drone. The right plot defaults to **Reward breakdown**. Select one drone from those active at the displayed
frame to see its cumulative reward components as colored stacked areas. Positive
components stack above zero and negative components below it, with no net-reward
overlay. Components that stay zero for that drone throughout the entire episode
are omitted from both the plot and its legend. The legend below the chart identifies each color, and hovering
an area shows its reward name without permanent labels inside the plot. The drone
selection is retained while it remains active. **Relative advantage**
in the View dropdown restores the previous comparison: at every step it subtracts
the lowest drone reward before accumulating.
Both plots advance with the frame controls and playback speed. Their x axis
uses replay steps; the Episode reward remains the aggregate reward for the
selected frame. Replays without per-drone reward data show an unavailable
message in the plot area. The breakdown specifically shows **Data not available
(legacy replay)** for older recordings; the inspector never reconstructs components.

All policy and recorded-action replay collectors now capture the additive terms
returned by the reward function at each transition, including adaptive population
and reference rewards. The NPZ retains `reward_terms` as the historical per-drone
total and adds `reward_components` with shape `(frames, agents, components)`;
the JSON manifest's `reward_component_names` labels that final axis. The initial
frame has zero reward and components. Component amounts include the terminal
transition and are trimmed to the same frames as the recorded states. Event
counters and the positive/negative reference diagnostics are excluded because
they are not additional rewards. The writer checks that components sum to the
recorded totals before saving. Public exports retain this breakdown too.

For the `solo_B02c_a4c4_15x15_rewardChange` 180M `random_eval_0001`
comparison, run `uv run python tools/rescore_b02c_replay.py` from the repository
root. The command writes a separate replay under the run's
`artifacts/eval/replays` directory. It copies the recorded positions and all
other episode frames, then swaps the run's vested gap-change component for the
recurring gap reward in the current B02c level config. The original replay is
untouched. The saved `ckpt_000450` is the nearest checkpoint, but the command
uses the exact recorded trajectory, so it does not need to regenerate actions
from that checkpoint. The output appears in the inspector as
`legacy gap fixed trajectory`.

The two reward timelines sit side by side beneath the 3D view. Playback follows
elapsed time at the selected speed; if rendering takes longer than a frame
interval, it advances to the frame for the current time rather than slowing the
episode clock.

`swarmecho-eval-dashboard` and `swarmecho-artefacts` now launch the same
inspector. Use its `key=value` options, rather than old Streamlit or artefacts
server options. The old flat-map target picker, MP4 renderer, and evaluation
job queue are removed.

The **Export public page** button prepares the selected replay or heatmap as a
static page in `public-inspector/`, with optional precomputed reference paths.
Choose or create a section, add custom tags, and capture a thumbnail of the
current view. The public gallery sorts cards by publication date and sections
by their newest card, with horizontal dividers between section galleries.
The reduced viewer retains the 3D controls without discovery or evaluation
commands. Exports can be previewed locally and published with GitHub Pages;
see [the export and deployment guide](04_public_inspector.md).

Evaluation CSVs contain XYZ coordinates with either categorical stages or
cumulative robust outcome rates. Both formats remain supported. Replay JSON
manifests and their numerical data are under the run's artifact hierarchy;
checkpoint evaluation groups artifacts beneath its checkpoint directory.

## building editor

```bash
uv run swarmecho-maze-builder --host 127.0.0.1 --port 8766
```

The historical command name is retained. The server serves the existing spatial
editor and `/api/buildings` routes for listing, loading, authoring, validation,
and saving. Map-name validation retains the same filename rules.

Create and edit storeys, walls, tiles, exclusions, roof geometry, and spawn
settings.

In **Staircase** mode, click an interior cell to place stairs. Placing them on
the top storey automatically adds the storey above, copying the volume, walls,
and exclusions using the normal Add layer behavior. The upper stair cell and
its floor opening are created together. UP/DOWN markers show the connected
sections on both storeys; tile painting keeps that opening clear.

Click either section to select the staircase, then change **Stair ascent** or
use **Rotate 90°** / **R**. **Remove stairs** removes the staircase from both
storeys; the opening can then be filled with Tiles. New buildings and stair
templates use full cell width treads. The preview honors **Full cell width
stairs**, which applies to all stairs in the map; existing maps retain their
saved width setting, including legacy narrow stairs.

Finish the roof with **Add roof**, then validate the saved YAML:

```bash
uv run swarmecho-validate-building src/swarmecho/curriculum_config/maps/custom_building.yaml
```

The optional `--runtime-smoke` flag also constructs and resets the environment.
It requires the project's Python/JAX environment.

## Diagnostics

`swarmecho-validate` runs the small rollout/PPO validator.
`swarmecho-benchmark` benchmarks the environment.
The roadmap and radar diagnostic scripts remain under `tests/` and `tools/`.
# Geo experiment reports

The geo probe runner includes an observation-ambiguity audit. Open
[`geo_experiments/outputs/index.html`](../../geo_experiments/outputs/index.html)
for the report sidebar and the in-report comparison dropdown. Numeric table values
are matched by section/table/row/column meaning, units and version; figures are
unchanged. See the [geo runner guide](../../geo_experiments/README.md) for settings,
historical audit import and compact source-data conventions.

The sequential probe plan lives in `geo_experiments/probes.yaml`:

```bash
uv run python geo_experiments/suite.py plan=true
uv run python geo_experiments/suite.py
```

The current default runs only shared-coordinate target decoding on the latest
`G03_random_4k_v1_agents_4` checkpoint and all 400 verified frozen validation maps,
with 10k/1k/5k total samples and nine fits. Historical G02 plans still share
delivery-eligible collection across map-specific, map+cell and shared-coordinate
readouts; their authored-map behaviour is unchanged. Each has
its own report, required-agent target groups, chain-success distribution highlights
and an optional storey head. The cheap latent screen routes packet collisions to
saved-snapshot similarity reports; only conflicting finder GRUs can trigger
observation replay. With all inputs below threshold, a footer gives the rates.
`geo_experiments/auditing.py run=<suite collection>` can refresh these diagnostics
without a policy. Probe console output uses the training clock-prefix formatter. Historical G01
features/splits can be refitted with `geo_experiments/reproduce.py run=<folder>`.
See the [G03 protocol](../geo/G03_probes.md) and [scoped G02 protocol](../geo/G02_probes.md) for interpretation and limits.

