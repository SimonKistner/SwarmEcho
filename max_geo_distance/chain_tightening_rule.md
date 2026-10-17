# Existing-relay removal and repeated local rules

## Saved evidence

The [lifted-corner result](results/20261014_215807_055727Z_07_lifted_corner_simplification.txt)
accepted 1,585 of 2,473 eligible attempts, improving 1,484 targets. Mean placement
count fell from 4.700 to 4.304. Targets above seven agents fell from 1,035 to
706: 678 at eight agents and 28 at nine. Previous horizontal/doorway replacement
counts were reproduced at 1,651 and 4,489.

The largest rejection category was right horizontal distance exceeding range
(540 attempts). Its recorded representative is a corner followed by an inserted
finder: moving the finder to the computed XY intersection would leave the target
outside visual range. A two-node replacement therefore cannot be assumed merely
because its middle link is short. Other categories were no forward intersection
(289), no height with left visibility (48), no point-clear height (10) and an
intersection outside bounds (1). These are scan attempts, not counts of distinct
unsolved targets.

Existing steps each apply one family of rules once, with local revisits inside
that family. They do not return to all earlier families after the lifted rule
moves connectors. The final chains may consequently expose additional earlier
replacements or existing drones whose neighbors can already connect. This is
a hypothesis to measure, not an inference that all 706 targets will be solved
by the proposed step.

## Scoped next construction

First reproduce all previous stages. Scan the resulting chain from base to
target. For `A -> B -> C`, remove the existing drone B only if the direct A-C
segment is within its applicable range and unobstructed through authored solids.
Use base communication range when A is the station, visual range when C is the
target, and agent communication range otherwise. Preserve at least one finder
drone. Retained drone positions remain unchanged, so their existing clearance
and deployment connections are preserved. After deletion, revisit the preceding
window to account for its new neighbor.

This deletion is particularly relevant after previous connector movements;
it does not assume that the initial shortest-path nodes were redundant. It
tests one deterministic local simplification and makes no new placement or
alternative-route search.

Then run the existing horizontal, doorway and lifted families followed by the
same deletion scan. Repeat that cycle until its agent count no longer decreases.
All these operations only remove drones, so at most the starting count minus
one successful removals can occur. A zero-change cycle terminates the process.
Stopping means these implemented rules cannot further reduce this chain in the
chosen order; it does not prove a global minimum or order-independent result.

## Output and execution

```bash
uv run python max_geo_distance/run.py --tests chain_tightening
```

Print only this new report. Compare corrected, horizontal, doorway, lifted,
direct-deletion and repeated-rule counts on identical targets. Report incremental
savings from direct deletion and subsequent cycles, attempts/rejections and
accepted removals by rule and endpoint role. Save full before/after coordinates
for representative improved targets and any remaining targets above seven agents,
so future analysis has actual final-chain witnesses rather than isolated attempts.

The identical transcript is archived as
`results/<UTC timestamp>_08_chain_tightening.txt`. Previous scripts/results and
the default runner selection are preserved. There is no production integration,
new simulation, random placement search or additional validation suite.
The coding agent has not executed this step; its outcomes remain unmeasured.
