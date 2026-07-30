# Future Features

## Information-ranked selective evaluation renderer

Add a renderer that selects evaluation episodes from the comprehensive
evaluation CSV according to useful spatial or outcome criteria instead of
the removed first-come success/failure buckets, corner special case, CSV target
loader, or cluster-representative replay path. Candidate selections include:

- nearest failure to the base;
- furthest successful target from the base;
- failures and successes near the decision boundary;
- spatial outliers or representatives of dense failure regions.

The evaluation CSV already provides the initial data needed for these
selections: target coordinates, success/failure outcome, and Euclidean
target-to-base distance. A future implementation should finish the full
parallel evaluation first, rank its rows, and only then collect videos for the
chosen targets. This keeps metric collection independent of rendering.

If the future renderer must reproduce the exact original episode rather than
re-evaluate a selected target position, the saved data must also include and
reuse the original reset PRNG key. Deterministic policy actions alone do not
reproduce randomized base and drone spawns. Alternatively, the CSV schema could
store the realized base and drone spawn positions needed for exact replay.
