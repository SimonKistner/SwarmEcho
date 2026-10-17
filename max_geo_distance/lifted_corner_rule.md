# Horizontal intersection with a calculated connector height

## Measured progress and remaining scope

The [saved doorway result](results/20261014_214955_450071Z_06_doorway_simplification.txt)
improved 3,608 targets, saved 4,489 additional drones, and accepted 4,489 of
4,827 eligible attempts. Its mean placement count was 4.700, compared with
5.822 after the horizontal-only rule and 6.235 for the corrected baseline.
Targets above seven agents fell from 1,289 to 1,035; the remaining distribution
was 621 targets at eight agents, 324 at nine, and 90 at ten.

There were 338 rejected doorway attempts: 229 empty common apertures, 49
endpoints too far from the opening plane, 35 empty necessary range bounds and
25 final range failures. Some of these occur on already short chains, so their
aggregate count alone does not identify the largest remaining >7-agent group.

The [earlier geometry report](results/20261014_213632_828118Z_05_remaining_corner_geometry.txt)
identified a diagonal B/C transition between two openings, affecting 1,227
targets then above seven agents. In its representative, B/C are on the same
storey but A arrives diagonally from below; the outer infinite lines are
0.812524m apart. The doorway rule handled its following wall-face pair, but
does not directly match this diagonal transition. The current saved sample 3
ends at eight agents after both previous rules. This is evidence for a focused
next rule, not a measurement of its eventual success or of its current total
frequency after doorway replacements.

## Scoped construction, recorded before execution

Consider `A -> B -> C -> D`, with B/C drones less than 2m apart and at least one
non-horizontal leg. Extend the XY projection of A-B beyond B, and the XY
projection of D-C beyond C. Solve their forward intersection once. Parallel,
degenerate or backward rays are rejected. This fixes P.x and P.y without
requiring the two 3D lines to intersect.

Derive a feasible set of P.z values analytically at that fixed XY:

1. Start with drone world bounds and the exact height intervals allowed by
   both endpoint range spheres.
2. Remove heights placing P in any inflated planning solid.
3. Remove heights obstructing A-P or D-P through any authored solid, including
   doorway jambs, floors and stair treads.
4. Intersect with heights having a planning-clear connection from P to at least
   one of the two removed positions B/C.

For an observer O and a solid box, the fixed XY segment determines a parameter
interval `[t0,t1]` where its horizontal projection crosses the box. Segment
height is `(1-t)*O.z + t*P.z`. Staying below the box gives an upper bound for
P.z; staying above gives a lower bound. Their complement is one forbidden
height interval. Its bounds come from evaluating the box's lower/upper z at
t0 and t1, without sampling heights. Intersect/subtract those intervals using
small numerical safety margins and the existing segment epsilon.

Project the B/C mean height onto the resulting feasible interval set. This is
one deterministic nearest-height calculation, with the lower height breaking
equal-distance ties. It is not a trial of multiple placements or an iterative
optimizer. Construct exactly one P, then independently check point clearance,
world bounds, full 3D endpoint ranges, exact authored-solid visibility and the
planning-clear connection to B/C. Empty interval sets and failed checks retain
the existing pair.

Apply corrected-chain construction, the existing horizontal rule and the
doorway rule as preprocessing. Scan the new rule from base to target and
revisit neighboring windows after acceptance. Each acceptance removes one
drone. Older rules and artifacts remain unchanged. Fixed XY, the short-gap
scope and numerical margins keep the result conservative; this does not claim
the globally best connector, minimal agent count or policy success.

## Command and saved output

```bash
uv run python max_geo_distance/run.py --tests lifted_corner_simplification
```

Only the new report is printed, comparing corrected, horizontal-only,
doorway-combined and lifted-corner counts on identical targets. It reports
incremental savings, >7-agent counts, rejection reasons and representative
accepted/rejected geometry, including allowed height intervals and proposed P.
The transcript is saved to
`results/<UTC timestamp>_07_lifted_corner_simplification.txt`.

The coding agent has not run this experiment or added a validation suite.
Actual outcomes remain unmeasured until the user runs the command.
