# Removed geometry and feasibility variants

The maintained building compiler, roadmap and global relay estimator are explained in [environment](../02_environment_and_physics.md#collision-geometry-and-geodesics) and [training](../03_marl_and_training.md#relay-difficulty-and-target-eligibility).

| Removed alternative | Historical role |
| --- | --- |
| Simplifier relay estimator | Constructed and shortened a candidate relay route as an alternative feasibility estimate |
| Minimal roadmap | Used a smaller set of cell, portal and stair features instead of the full geometry-feature roadmap |
| Disabled feasibility gate/fallback | Allowed alternate reset behaviour when geometric relay-budget filtering was bypassed |
| Training relay-budget override | Used a target-feasibility drone budget different from the configured training fleet |
| Minimum geodesic separation | Rejected base–target pairs with routes shorter than a chosen distance |
| Roadmap corner allowance | Adjusted the historical separation rule to account for route corners |
| Inserted AABB obstacles | Added configured or randomized box obstacles through alternate runtime layout paths |

These alternatives affected target distributions and cached preparation. Historical results need their original geometry, clearance and feasibility settings. A common map filename alone cannot establish equal target difficulty.

Randomized bases, authored-map training, directly generated buildings and compiled-bank selection remain supported. Their constraints are described in the current references. In particular, removing minimum separation leaves the [randomized-base target-distribution issue](../02_environment_and_physics.md#target-placement-and-base-positions) to be handled when designing that experiment.
