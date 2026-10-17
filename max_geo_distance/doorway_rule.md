# Doorway connector: evidence and scoped next step

## Evidence from the saved runs

The [first replacement result](results/20261014_212809_672783Z_04_corner_simplification.txt)
accepted 1,651 replacements, reducing the corrected baseline's 1,527 targets
above seven agents to 1,289. Its mean placement count fell from 6.235 to 5.822.

The [remaining geometry result](results/20261014_213632_828118Z_05_remaining_corner_geometry.txt)
measured 7,624 short non-horizontal occurrences at 834 ordered B/C locations.
They affected 3,802 targets, including every one of the 1,289 remaining targets
above seven agents. These are measured counts, not new simplification results.

The highest-priority pair appears on 3,139 targets and all 1,289 remaining
targets above seven agents. Its B/C coordinates have identical x/z and y values
10.475000381... and 9.524999618..., on opposite sides of the authored
`y-door(2, 2, 0)` at y=10m. The door spans x=11.5..13.5m and z=0..3.5m.
The 0.95m gap corresponds to the 0.25m wall thickness plus two 0.35m planning
clearances. Its first representative has a 0.587761m gap between the outer
infinite lines. This identifies a doorway crossing adjacent to a stairwell;
the nearest stair envelope alone had not established its type.

Several other frequent pairs likewise sit on opposite x-door faces. The
second-ranked pair, affecting 1,227 targets above seven agents, is a diagonal
transition between two openings, with a 0.812524m outer-line gap. The doorway
rule below is scoped to crossing one opening; that diagonal case remains a
separate follow-up. No future success count is inferred from these examples.

## Proposed deterministic construction

For `A -> B -> C -> D`, B/C must be drones less than 2m apart and lie outside
opposite wall faces of one authored door/window. Their connecting segment must
cross that aperture at both wall faces. Match the actual opening geometry,
without an angle-bin requirement or a nearest-feature classification.

Fix the proposed connector P on the opening's centre plane. Begin with the
aperture rectangle shrunk by drone radius plus planning clearance, intersected
with world bounds. For each observer A/D and every wall face between the
observer and that plane, project the aperture bounds back to P's plane. At a
face, the segment's tangential coordinates are

```text
q = (1 - t) * observer + t * P
t = (wall_face - observer_normal) / (portal_plane - observer_normal)
```

The aperture inequalities for q give coordinate intervals for P. Intersect
those intervals for both observers to obtain a common visible rectangle.
Necessary coordinate bounds from both endpoint ranges further restrict it.
Project the B/C midpoint into that rectangle once to obtain a single P.
There is no iterative optimization, alternative candidate, or rollout search.

The full endpoint distances, authored-solid visibility, planning-clear point,
world bounds, and a planning-clear connection to B or C are checked before
acceptance. Base communication range is used at the base; visual range is used
at the target. Other links use agent communication range. All map solids take
part in final checks, including adjacent floors and stair treads.

Coordinate range bounds are necessary conditions; their rectangular intersection
does not replace the final spherical range checks. A rejected single candidate
does not prove that no feasible connector exists elsewhere. Accepted replacements
retain a concrete geometric chain under the same model as previous inspections.
They do not prove a global minimum or trained-policy success.

Apply the existing horizontal rule first, then scan doorway pairs from base to
target, revisiting affected neighbors after acceptance. Replace two drones by
one for each acceptance. Preserve the prior inspection implementations and
historical artifacts.

## Execution and output

```bash
uv run python max_geo_distance/run.py --tests doorway_simplification
```

Only this new inspection report is printed. It compares corrected baseline,
horizontal-only and combined placement counts on identical targets, and reports
doorway attempts, acceptances/rejections, and per-opening effects. Representative
successful positions and rejected cases include A/B/C/D, proposed P where one
exists, opening bounds and rejection reasons. The console transcript is saved
as `results/<UTC timestamp>_06_doorway_simplification.txt`.

The coding agent has not executed this step. Outcomes remain unmeasured until
the user runs the command. No production integration or additional validation
suite is part of this step.
