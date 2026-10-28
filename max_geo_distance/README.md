# Max geo distance

All scripts, documentation, and any future tests for this feature belong here.

## Running and archiving inspections

Run all three existing inspections in their original order:

```bash
uv run python max_geo_distance/run.py
```

This produces the same tables and console text as the previous command with
`--corrected-baseline --corner-patterns`, including preparation and inspection
timings (which naturally vary between runs). The roadmap and targets are prepared
once, and all selected inspections share the original paths.

Each inspection has its own file:

| File | Inspection |
| --- | --- |
| `path_baseline.py` | Original node-count histogram and range checks |
| `corrected_baseline.py` | Corrected placement-count baseline |
| `corner_patterns.py` | Original turn angles and consecutive corner patterns |
| `corner_simplification.py` | First two-node-to-one-drone rule and paired baseline comparison |
| `remaining_corner_geometry.py` | Locations and obstacle context of remaining short non-horizontal pairs |
| `doorway_simplification.py` | Shared-opening connector with a three-stage placement-count comparison |
| `lifted_corner_simplification.py` | XY intersection with an analytically feasible height and four-stage comparison |
| `chain_tightening.py` | Remove directly bypassable relays and repeat existing rules until stable |
| `general_corner_simplification.py` | Apply the same connector construction without short-gap or angle-pattern filters |
| `runtime_chain_hold.py` | Place exactly k drones and check actual float32 runtime chain-hold success |

`paths.py` holds shared sampling and shortest-path preparation. `reporting.py`
holds console formatting and transcript capture. `run.py` selects and chains
inspections; `inspect_paths.py` preserves the previously supplied commands.

Run only the inspection you need, either directly or through the runner:

```bash
uv run python max_geo_distance/corner_patterns.py
uv run python max_geo_distance/run.py --tests corrected_baseline corner_patterns
```

Every selected inspection writes a separate small UTF-8 text transcript under
`max_geo_distance/results/`. Filenames share a UTC timestamp per invocation:

```text
20261014_180000_123456Z_01_path_baseline.txt
20261014_180000_123456Z_02_corrected_baseline.txt
20261014_180000_123456Z_03_corner_patterns.txt
```

The timestamp above is an illustrative filename, not an actual run. Each file
contains the console's shared map/settings/seed header, that inspection's exact
console section, and the shared timing footer. The console prints shared text
only once. No tables are reformatted for the archive, and older results are not
overwritten. A fresh invocation runs only its selected inspections; it does not
replay old runs. The supplied historical console output is separately archived
with filenames beginning `20261014_user_supplied_`; its date comes from this
conversation and is not a recorded execution timestamp.

## Integrated feature and cumulative agent curriculum

[integration.md](integration.md) documents the default-off parameters, runtime
integration, adaptive spawning/size semantics, stage lifecycle and commands.
The reference GPU run certified all 11,995 produced placements; five targets
remain unresolved and are rejected. The new production JAX port has a separate
`validate_integration.py` command for comparing counts, real hold certification
and actual reset compatibility before launching training.

## Revised assignment checks before integration

[assignment_corrections.md](assignment_corrections.md) records the three issues
found by the numbered-map run and the revised experiment protocol. The current
scripts separate legal drone starts from station coordinates, round connector
placements to float32, enforce a small numerical margin, retain prior chains
when a simplification fails, and recover missing endpoint paths through visible
finder positions on existing reachable roadmap nodes/edges. Remaining misses
and invalid baselines are explicitly reported and remain failures in the
overall GPU denominator. Old transcripts and protocols are preserved.

Rerun the same two-phase command:

```bash
uv run python max_geo_distance/run_maps.py
```

These corrections preceded the production integration documented above. The historical sections
below describe the original experiments; revised output headers identify the
`finder-attachment-v2` protocol and can differ from their old path/count results.

## First inspection

From the repository root, run:

```bash
uv run python max_geo_distance/inspect_paths.py
```

This samples 4,000 continuous target positions uniformly by available spawn
volume on `B02c_static_map_noSpawn`, respecting its noSpawn cells, solid geometry,
and target wall buffer. The fixed base is the map's bottom-floor station.
Roadmap and range settings come from `G01_static`: full roadmap, density 0.5,
wall-end merging enabled, 15m agent communication, 15m base communication,
and 5m target visual range. Training settings and adaptive spawn weights are
not used. There is no minimum or maximum distance rejection.

The console shows a node-count histogram and how many path segments exceed
their respective ranges. Node counts exclude the base and target. Internal
node-node lines exclude both endpoint tips and count each occurrence across
the sampled paths. The base tip uses base communication range; the target tip
uses visual range. Direct paths have zero roadmap nodes and are reported
separately. Unreachable samples remain in the histogram's percentage denominator
and are listed separately rather than replaced.

The script reuses the existing roadmap builder and visibility calculation.
It computes one shortest-path tree from the base, then attaches each target
to its cheapest visible node (or uses a shorter direct segment). Paths minimize
raw Euclidean edge-length sums on this roadmap, without a corner penalty or
path simplification. Equal-length alternatives use the shortest-path solver's
chosen predecessor; node counts are not separately optimized. These statistics
do not yet estimate the minimum number of agents.

The authored station rests at z=0.125m, inside the floor's inflated drone
planning clearance. For path attachment, the script explicitly raises the
planning start just above that clearance at the same XY, still on the bottom
floor, and prints both coordinates. Base-tip distance checks use the actual
station coordinate. This endpoint convention is local to this inspection;
production code is unchanged.

Optional arguments: `--targets`, `--seed`, `--map`, and `--level`. Defaults are
4,000 targets and seed 456789. The script archives console text, creates no dashboard,
and runs no training or validation suite. The geometry-only inspections use
NumPy/SciPy on the CPU and do not initialize JAX; the separately selected runtime
check below does initialize JAX.

## Corrected baseline and corner inventory

```bash
uv run python max_geo_distance/inspect_paths.py --corrected-baseline --corner-patterns
```

This uses the same target sample and original shortest paths. It keeps the
original console tables and adds an agent-count histogram. The corrected chain
retains every roadmap node, inserts intermediate drones if a communication link
is too long, and extends the last segment until the final drone is within visual
range of the target. Direct paths get at least one finder drone. If the planning
attachment was raised above the authored station, it is included as an additional
drone. This is a constructed placement count on the selected path, not a minimum
agent estimate or a simulation of policy success. It assumes the existing roadmap
visibility and clearance geometry; production reset and communication code are
unchanged.

## Tightening the resulting chains

[chain_tightening_rule.md](chain_tightening_rule.md) records the latest lifted
results and the design for direct relay deletion and repeated local rules.

```bash
uv run python max_geo_distance/run.py --tests chain_tightening
```

Only this new report is printed and saved to
`results/<UTC timestamp>_08_chain_tightening.txt`. It compares all prior stages,
one direct-deletion scan and repeated rules, with separate savings and rejection
counts. Full coordinates for representative improved and remaining >7-agent
chains are included for future inspection. It creates no new placement-search
algorithm and makes no claim of a globally minimum count.

## Generalizing corner eligibility

[general_corner_rule.md](general_corner_rule.md) records the tightening result
and the recurring 2.325m corner excluded by the original experimental cutoff.

```bash
uv run python max_geo_distance/run.py --tests general_corner_simplification
```

This uses the existing XY intersection and safe-height construction on all
nondegenerate consecutive drone pairs, including horizontal cases and gaps
above 2m. It prints only its new comparison with the previous stable chain and
corrected baseline, and saves full example chains and outcomes to
`results/<UTC timestamp>_09_general_corner_simplification.txt`. Older inspections
and the default runner selection remain unchanged.

## Compare numbered B02c maps, then check every assignment on GPU

```bash
uv run python max_geo_distance/run_maps.py
```

This discovers the numbered `B02c_static_map_<digits>` maps (currently 02344,
02557 and 02713), samples 4,000 targets per map and applies the unchanged full
simplifier. It first prints per-map and pooled baseline/final tables, then checks
the exact retained assignments with the real GPU chain-hold protocol on every
map and prints an aggregate certification table. Each map uses its authored base,
`G01_static` settings and sampling seed 456789. No reference-map seven-agent
limit is imposed on other maps.

The identical console output is archived in separate timestamped
`results/*_11_maps_geometry.txt` and `results/*_12_maps_runtime_chain_hold.txt`
files. Placement failures are reported while the remaining maps continue;
a nonzero exit occurs after the final aggregate if any target is uncertified.
GPU availability/runtime errors stop execution with the partial report saved.
The design and interpretation are in [maps_comparison.md](maps_comparison.md).

## Check placed assignments in the real environment

```bash
uv run python max_geo_distance/run.py --tests runtime_chain_hold --runtime-device gpu
```

This uses exactly the proposed k agents and their simplified positions for each
target, then calls the real environment with zero actions only for its hold
window. It checks genuine connectivity, chain-hold success, unchanged positions
and physical validity in float32. It performs no navigation, policy rollout or
episode from spawn. The complete protocol and scope are recorded in
[runtime_chain_hold.md](runtime_chain_hold.md).

Group progress and per-agent-count pass/fail results are printed and archived
as `results/<UTC timestamp>_10_runtime_chain_hold.txt`. GPU is required by default;
CPU must be explicitly selected. First execution compiles per-k kernels.
`--runtime-batch-size` defaults to 128. Failures are saved before a nonzero exit.

The corner inventory measures the original path, before those insertions. Angles
are 3D direction changes: 0 degrees is straight and 180 degrees is a reversal.
The angle histogram rounds to the nearest 5 degrees. Consecutive turn patterns
describe `A -> B -> C -> D`, with turns at B and C, grouped by both angles,
horizontal/vertical/3D orientation, turn-normal alignment (same/opposite/skew),
and the length of B-C. Near-zero turn normals are labelled `straight`.
Separation buckets are `<0.5m`, `0.5-1m`, `1-2m`, `2-5m`, `5-15m`, and `>=15m`.
No turn is discarded for being small; no simplification rule is applied.

Counts are occurrences across all original target paths, including repeated
shared portions. Pattern percentages use all consecutive-pattern occurrences as
their denominator; angle percentages use all measured turns. Target percentages
still use all 4,000 samples. The most frequent 20 pattern groups are printed by
default, with the remaining occurrence count shown. Use `--pattern-limit 1000`
to display up to 1,000 groups. Only console text is saved; no additional maps,
target samples, dashboard, tests, or validation runs are created. The original
three inspections do not apply simplification.

## First corner simplification: scoped design

Run only this new step and its paired baseline comparison:

```bash
uv run python max_geo_distance/run.py --tests corner_simplification
```

The reference is the supplied corrected-baseline result in
`results/20261014_user_supplied_02_corrected_baseline.txt`: 1,527 of 4,000
targets required more than seven agents. The supplied corner inventory recorded
1,651 occurrences of horizontal, same-direction 45/45-degree turns separated
by 1-2m. These are observations from the previous run, not results of this new
rule. No execution of this new step has been performed by the coding agent.

This step reuses corrected-chain construction as its input, without running or
printing any of the earlier inspection reports. Before and after counts are
calculated for each identical sampled target, so comparisons also work with
different seeds or maps. Historical text is a reference, not a substituted
baseline measurement for the current invocation.

For four consecutive chain positions `A -> B -> C -> D`, B and C must both be
drones. The rule initially handles only horizontal turns whose measured angles
round to 45 degrees in the existing 5-degree bins, whose turn directions agree,
and whose B-C separation is at least 1m and less than 2m. It extends `A -> B`
beyond B and `D -> C` beyond C, solves their XY intersection once, and proposes
that one point P as a replacement for B and C. There is no candidate search,
alternative path search, or rollout simulation.

Acceptance requires a forward, coplanar intersection, a point within drone
world bounds and outside the inflated planning solids, and two unobstructed
links with the applicable ranges. Radio and visual visibility use the authored
solids, as in runtime; point clearance uses the existing planning margin. The
left link uses base communication range when A is the actual station. The right
link uses visual range when D is the target, otherwise agent communication
range. A planning-clear segment from P to at least one of the removed drone
positions is also required, preserving a local way to reach the replacement.

The scan proceeds from the base towards the target. After each accepted
replacement, affected neighboring windows are revisited. Every acceptance
removes one drone, so this process terminates. All replacements retain concrete
positions. This rule makes no claim of a globally minimum agent count or trained
policy success, and makes no changes to production spawning.

The console and a new `*_04_corner_simplification.txt` transcript show the paired
agent-count histogram, targets improved, drones saved, the before/after count
above seven agents, and eligible attempts accepted or rejected by reason. These
counts describe the sequential scan, including revisited windows, rather than
the original inventory's untouched path occurrences. The existing no-argument
`run.py` command continues to select only the original three inspections.

## Remaining corner geometry: scoped inspection

```bash
uv run python max_geo_distance/run.py --tests remaining_corner_geometry
```

This new inspection reconstructs each corrected chain and applies the existing
horizontal simplification as preprocessing. It prints only its own diagnostic
report, with no new replacement rule or production integration. The coding
agent does not execute it. The previous supplied simplification run reported
1,651 accepted replacements and 1,289 targets still requiring more than seven
agents; these are historical observations, not assumed results of this step.

The inspection considers consecutive drone pairs B/C in `A -> B -> C -> D`
with a positive B-C gap below 2m and non-horizontal legs. Vertical-plane pairs
are retained alongside general 3D pairs. It groups by the exact ordered B/C
coordinate tuples. Occurrences and distinct affected targets are counted
separately, including how many of those targets still require more than seven
agents. Groups are prioritized by the latter count, then occurrence count.
Rounded turn angles describe each group without determining group membership.

The summary shows up to `--pattern-limit` groups (default 20). Detailed examples
for the first `--geometry-details` groups (default 5) include actual A/B/C/D and
target coordinates, baseline/simplified counts, measured turn angles, the gap
between the two outer infinite lines, and direct A-D signal visibility. They
also list the nearest authored doors/windows/stair envelopes and four nearest
physical solid boxes with their bounds. Distances are measured from the B-C
midpoint to those envelopes/boxes; a nearby feature is context, not a confirmed
classification of the pair. Door/window envelopes use the authored opening
dimensions; stair envelopes cover the two-storey cell, not individual treads.

The identical console transcript is automatically saved as
`results/<UTC timestamp>_05_remaining_corner_geometry.txt`, including map, seed,
settings and timings. Unshown-group totals are included. To inspect more groups
and save their detailed examples, increase `--pattern-limit` and
`--geometry-details`. The existing default run still selects only the original
three inspections. No new simplification formula is introduced by this step.

## Doorway connector

The remaining geometry report identifies opposite doorway-face nodes as the
largest recurring short pair. [doorway_rule.md](doorway_rule.md) records the
saved evidence, the construction and its scope before execution.

```bash
uv run python max_geo_distance/run.py --tests doorway_simplification
```

This prints only the new doorway comparison and saves the same text as
`results/<UTC timestamp>_06_doorway_simplification.txt`. It constructs one point
in the common visible part of each matched opening, then applies exact point,
range, visibility and local-reachability checks. Output compares corrected
baseline, horizontal-only and combined counts on the same targets. The default
runner and previous inspections retain their existing selections and rules.

## Lifted corner connector

[lifted_corner_rule.md](lifted_corner_rule.md) records the doorway results and
the scoped next construction. `height_intervals.py` contains its reusable height
arithmetic; it does not run a separate inspection.

```bash
uv run python max_geo_distance/run.py --tests lifted_corner_simplification
```

This fixes one XY ray intersection and calculates the height intervals allowed
by all solids, endpoint ranges, and a connection to the removed nodes. It selects
one nearest feasible height and applies the exact placement/link checks before
replacement. The new report compares all four stages and archives the same text
as `results/<UTC timestamp>_07_lifted_corner_simplification.txt`, including
representative connector coordinates, feasible height intervals and rejection
reasons. The earlier reports are not printed, and default runner selection is
unchanged.
