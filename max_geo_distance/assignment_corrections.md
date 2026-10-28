# Assignment corrections before production integration

## Evidence

The first numbered-map comparison is preserved in
[the geometry transcript](results/20261014_224038_869771Z_11_maps_geometry.txt)
and [the GPU transcript](results/20261014_224038_869771Z_12_maps_runtime_chain_hold.txt).
Of 12,000 original targets, 11,939 had assignments, 11,901 achieved runtime
chain hold, and 11,687 passed strict placement certification. The disjoint
reported problems were 214 invalid one-agent placements, 38 connectivity
failures and 61 targets without a roadmap route.

The displayed invalid placements use the station at z=0.125m or interpolate
below the runtime's minimum drone z=0.375m. Runtime pushes them upward. This is
a baseline-construction error, not rounding noise. The remaining connectivity
failures may include range and visibility boundaries, but their individual
causes have not all been established. Missing-route targets were not diagnosed
by the original transcript; they must not be labelled physically impossible.

## Revised protocol, recorded before the user rerun

All changes are confined to `max_geo_distance/`; production sampling, levels
and curricula are unchanged. Old transcripts remain intact. Revised headers
identify `legal-drone-start / float32-links / finder-attachment-v2`.

### Legal drone start

Keep the authored station coordinate unchanged for radio links. Begin flight
planning at its normal drone spawn height, inset 1mm from runtime bounds. If
inside inflated planning geometry, lift above the containing slab(s), checking
the resulting world bounds. Include this actual legal drone point in the
corrected baseline. Direct and short-target chains therefore cannot substitute
the ground station for a drone. Existing pruning can remove the initial drone
when a later drone has a valid direct station link.

This changes the baseline and can change selected paths because the tree now
starts at a valid drone location. Historical mean counts are not substituted
for this invocation's freshly computed baseline.

### Rounded placements and conservative acceptance

`placement_checks.py` mirrors the runtime's slab/range operations with NumPy
float32 operands and rounds witness positions before acceptance. It deliberately
does not claim CPU/GPU arithmetic is bitwise identical; the independent actual
GPU hold check remains authoritative.

Use a fixed 1mm numerical margin for physical signal-box inflation and link
acceptance. Construct range-limited points 2mm inside the nominal range, so
rounding cannot put an exactly constructed point just outside the acceptance
margin. For unusually tiny ranges, cap each margin at 1% of that range.
Height intervals and opening projections account for the new margins before
selecting their one connector. No extra XY candidate is searched.

Every connector family and relay bypass passes the shared rounded checks.
Failed proposals leave the previous positions intact. The generalized pipeline
also validates the entire baseline, previous stable chain and final chain.
A failed stable result rolls back to a previously validated chain. A baseline
that cannot pass is reported as an unresolved assignment with its reason and
example coordinates; it is not silently passed to training or called certified.
Final returned positions are already exactly representable as float32.

This may retain more agents than the old near-boundary arithmetic. Count
changes are reported rather than relaxed to reproduce previous histograms.

### Finder attachment and missing-route diagnosis

Preserve normal shortest-path endpoint attachment when it succeeds. Only on
an endpoint miss, consider physically visible finder positions within visual
range on the existing base-reachable roadmap:

- Existing reachable nodes and the legal drone start.
- The one analytic nearest point on each existing reachable edge.

Choose the cheapest valid attachment using the existing shortest-path tree.
The edge projection must be a legal rounded drone position and have a clear
planning connection to its selected endpoint; the finder-to-target sight line
uses physical signal geometry and the numerical margin. This does not require
a drone-clear flight segment all the way into the target coordinate. It adds
no roadmap vertices, changes no authored geometry, and runs no alternative
navigation search. It is a conservative recovery, not a completeness proof:
for example, a different point on an edge could see a target when its nearest
point cannot.

Record original endpoint misses, recovered witnesses, graph component count,
base-reachable node count, affected cells, planning-visible components, nearest
reachable-node distance, detailed representative targets and every unresolved
sample index. A remaining miss is explicitly unresolved, not proof that the
physical task cannot be solved. The rerun determines whether the original 61
misses are recovered or require a graph-coverage investigation.

### Reporting and GPU gate

Keep the same 4,000-target seed/sample budget on each map. Geometry reports
separate missing routes from invalid baseline assignments, print recovery and
guard counters, then print the aggregate histogram. GPU checks use the exact
retained targets and proposed k-agent placements with the unchanged real
50-tick G01 hold protocol. No navigation episodes are run.

Unresolved targets remain failures in the overall denominator. An improvement
in the tested-assignment pass rate cannot hide an increase in missing
assignments. Preserve first-tick connectivity, real hold success, no-collision,
unchanged-position and exact-population criteria. Save all maps before returning
nonzero if anything remains uncertified.

## Rerun and acceptance evidence

```bash
uv run python max_geo_distance/run_maps.py
```

This reruns the three numbered maps and archives new `*_11_maps_geometry.txt`
and `*_12_maps_runtime_chain_hold.txt` transcripts. An optional reference-map
check remains available through the existing single-map command or `--maps`.

The coding agent has not executed the revised experiment, compilation checks
or GPU code. Intended improvements must be checked against the new transcripts:
zero floor-placement/collision/drift failures; zero or explained connectivity
failures; and explicit recovered/unresolved counts for endpoint misses.
The subsequent user rerun is preserved in
[20261014_230730_049601Z_11_maps_geometry.txt](results/20261014_230730_049601Z_11_maps_geometry.txt)
and [its GPU transcript](results/20261014_230730_049601Z_12_maps_runtime_chain_hold.txt).
It has zero invalid baseline placements and zero runtime failures among 11,995
produced assignments. Finder attachment recovered 56 of 61 original misses;
five remain unresolved in one cell on 02344, whose graph is connected. The user
accepted continuing to integration. [integration.md](integration.md) records
that work and the separate verification required for the new production query.
