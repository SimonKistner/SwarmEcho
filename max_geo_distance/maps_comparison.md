# Numbered static-map comparison

## Design recorded before execution

Apply the existing complete generalized simplification pipeline, unchanged,
to 4,000 independently sampled targets on each numbered B02c static map.
Default discovery matches only `B02c_static_map_<digits>.yaml` in the existing
map directory, sorted by numeric suffix. The currently authored set is
`B02c_static_map_02344`, `B02c_static_map_02557`, and `B02c_static_map_02713`.
`noSpawn`, `cellar`, and the smaller-map bank are outside this default set.

Use `G01_static`, seed 456789 independently per map, and each map's authored
fixed base. Sampling follows the existing available spawn-volume calculation,
including any authored noSpawn exclusions and target wall buffer. It does not
use adaptive training weights or minimum/maximum spawn-distance rejection.
Equal seeds do not imply identical world-space targets across different layouts.

For every target, reconstruct the selected shortest roadmap path and apply
the corrected baseline, horizontal connector, doorway connector, lifted
connector, relay deletion and generalized connector in their existing order,
including the existing repetition and shared-window cache. No extra candidates,
new margins, map-specific tuning or seven-agent cutoff are introduced.

Complete geometry processing on every map before printing the aggregate.
Report per-map corrected/final mean counts, final maximum, targets above seven,
unreachable targets and total drones saved. Print the final count distribution
for each map alongside pooled corrected/final counts and a console histogram.
Means exclude unreachable targets; distribution percentages include them.
The reference policy's seven-agent performance is not assumed for other maps.

Retain the exact targets and final chains in memory, then perform the existing
[real-environment static assignment check](runtime_chain_hold.md) on the GPU
for every map. Each environment has exactly its proposed k agents; the actual
hold requirement, zero-action protocol, float32 precision and strict pass
criteria remain unchanged. There is no resampling and no rerun of the CPU
simplifier during this phase. Kernels are compiled separately for each map/k;
JAX compilation caches are cleared between maps to release retained executables.

Placement failures do not stop subsequent maps. Print each map's existing
runtime report with failed sample indices, overlapping failure reasons and
representative coordinates; afterwards print pooled results by map and by k.
Pass rates by map and overall include targets with no assignment as failures.
Per-k rates use tested assignments. Report times include first compilation;
these are not measurements of a warmed production spawn filter.

Runtime errors such as unavailable GPU stop execution and leave partial
transcripts. There is no CPU fallback. A completed batch with any uncertified
target exits nonzero after all maps and aggregate results have been saved;
the previously accepted small reference-map failure rate does not alter
certification criteria or hide failures.

## Command and artifacts

```bash
uv run python max_geo_distance/run_maps.py
```

The command prints and saves identical console text in two timestamped files:

- `results/<UTC timestamp>_11_maps_geometry.txt`
- `results/<UTC timestamp>_12_maps_runtime_chain_hold.txt`

The second transcript names the first for provenance. Both identify the map
set, seed, sample count and source level. Geometry headers retain each authored
base, roadmap settings and vertex count. Runtime reports identify device,
precision, hold settings, batch size and per-map failure details. Existing
single-map commands and transcripts are unaffected.

Optional arguments: `--maps <names...>`, `--targets`, `--seed`, `--level`,
`--runtime-batch-size` and `--geometry-details`. GPU is always required for
this two-phase command.

## Evidence and unresolved questions

The prior reference-map runtime transcript is
[20261014_223008_092430Z_10_runtime_chain_hold.txt](results/20261014_223008_092430Z_10_runtime_chain_hold.txt):
3,995/4,000 assignments certified. The user accepted that observed approximation
for continuing exploration. This is evidence on that sample, not evidence
on these three layouts or a guarantee for all future target positions.

The first user execution is preserved in
[20261014_224038_869771Z_11_maps_geometry.txt](results/20261014_224038_869771Z_11_maps_geometry.txt)
and [20261014_224038_869771Z_12_maps_runtime_chain_hold.txt](results/20261014_224038_869771Z_12_maps_runtime_chain_hold.txt).
It constructed 11,939 assignments, achieved 11,901 real holds and certified
11,687 of 12,000 targets. Failures comprised 214 invalid one-agent placements,
38 connectivity failures and 61 missing routes on 02344. The invalid examples
place drones below the runtime floor bound; these are not precision noise.

[assignment_corrections.md](assignment_corrections.md) records the revised
protocol before the next user execution. The original protocol above and its
artifacts remain historical evidence. Current scripts identify the revised
protocol in headers and report recovery/guard outcomes. No revised run has been
performed by the coding agent; remaining failure causes must use its diagnostics.
