# General corner scan without the initial pattern filters

## Evidence

The [tightening result](results/20261014_220348_062269Z_08_chain_tightening.txt)
saved 148 drones by direct deletion; repeating the existing rules saved none.
The mean count was 4.267. There were 684 targets above seven agents: 681 at
eight and three at nine. The repeated cycle reached a zero-change result on
its first pass for every target with more than one agent.

The saved remaining chains repeatedly include the internal positions
`(5.4749999, 3.1500001, 8.1499996)` and
`(3.1500001, 3.1500001, 8.1499996)`, separated by 2.325m. One preceding point
is `(10.4750004, 4.6500001, 6.8249998)` and one following point is
`(3.1500001, 5.4749999, 8.1499996)`. This pair is excluded by the earlier
short-gap rules before their feasibility checks run. The 2m bound was an
experimental scope restriction, not a communication or collision limit.

The first horizontal rule also admits only a specific 45/45-degree pattern,
while the lifted rule excludes wholly horizontal windows. Further rule cycling
cannot bypass those eligibility restrictions. The saved examples motivate
removing them; they do not establish a new success count for all targets.

## Scoped next step

Reproduce the stable chain from all previous steps. Then scan every consecutive
pair of drones B/C in `A -> B -> C -> D`, with nonzero legs, regardless of B-C
length, turn-angle bins or whether the legs are horizontal. Use the existing
single forward XY-ray intersection and analytic safe-height construction. The
new step changes eligibility, not that construction or its final checks.

The always-applied triangle-inequality rejection `distance(A,D) > R_left +
R_right` eliminates pairs that cannot share any one connector. Ranges use the
base/agent/target roles as before. Other failures, including parallel or backward
rays and empty height sets, keep B/C unchanged. No fallback point is tried.

After accepted replacements, revisit neighboring windows. Run direct deletion
and the existing local families alongside the generalized scan until a complete
cycle has no reduction. Each accepted operation removes one drone. Old scripts
retain their original scope and default behavior, preserving historical results.

One invocation may memoize identical windows and endpoint roles for this new
construction against its fixed prepared geometry, avoiding repeated computation
on shared chain sections. This is an in-memory calculation cache, not a spatial
atlas, sampling change or persistent dataset. Count cached visits and unique
geometric evaluations separately; every proposed point has the same deterministic
result for its identical input. No memory cap or new production parameter is added.

This remains a greedy construction of feasible chains under the current model.
It does not prove globally minimal counts, feasibility of rejected pairs under
other placements, or trained-policy success. All accepted new positions still
pass the existing full point, link and local-reachability checks.

## Output and execution

```bash
uv run python max_geo_distance/run.py --tests general_corner_simplification
```

Only the new report is printed. Compare corrected baseline, previous stable
chain and the new result on identical sampled targets. Include accepted counts
by middle-gap category and horizontal/non-horizontal geometry, rejection reasons,
additional operations by old rule, cache counts and full before/after witnesses
for representative improved and any remaining >7-agent targets. Save the same
text as `results/<UTC timestamp>_09_general_corner_simplification.txt`.

No new experiment has been executed by the coding agent. There is no production
integration or additional validation suite in this step.
