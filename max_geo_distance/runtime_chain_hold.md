# Real-environment static assignment check

## Scope and reference

The [generalized result](results/20261014_221142_776313Z_09_general_corner_simplification.txt)
constructs chains of at most seven agents for all 4,000 sampled targets, with
mean count 3.464. Those CPU float64 geometry checks alone do not certify the
same placements in the runtime's float32 step function. Some sight lines and
finder placements have approximately micrometre-scale margins.

The user explicitly requested placing exactly the proposed agent count at
the simplified coordinates and checking the real chain-hold status, without
navigation or episodes from reset through deployment. The protocol below is
recorded before execution. The coding agent does not run it.

## Protocol

Reconstruct the identical generalized chains and targets, using the existing
sampling seed and simplifier. Group by proposed k, and construct real
`make_env_fns` environments with `num_agents=k`, preserving the selected level's
full other environment settings. No additional active or inactive drone slots
are present. Batch independent targets with equal k on the selected JAX device;
the final batch may duplicate an environment for padding, but padding results
are excluded and every environment still has exactly k drone slots.

Create an ordinary runtime reset state once per k as a field template, then
inject the base, target and proposed drone coordinates in float32. Set all k
agents active, velocities zero, chain counters zero, knowledge/success flags
false, and stationary counters zero. Set the clock to the first timestep at
which all k agents are due under normal spawn scheduling. This bypasses spawn
and travel without disabling runtime activation, deadline or idle termination.

Use the actual `step` function with zero-force actions for at most the configured
`hold_chain_for` ticks (50 in G01). Freeze a state once its real `done` is true,
so a premature termination cannot be stepped past to manufacture success. Do
not repin positions, reset hold counters or override success/termination during
the window. The first tick must compute connectivity from the injected poses;
the hold counter is not seeded to a successful value.

One unrelated state field, coverage, is marked fully covered before each tick
so the compiler can discard coverage-credit ray work. This is valid only for
the enforced `success_condition=chain_held`; that condition does not inspect
coverage. Reward calculation and policy observations are not needed. Geodesic
planning is disabled in the runtime template because physical connectivity and
motion still use all authored solids and do not require roadmap distances.
Levels with adaptive population changes, generated extra obstacles or delivery
vesting are rejected by this first static protocol rather than silently altered.

Certification requires all of the following for each unpadded target:

- Float32 poses are finite, within the runtime drone bounds and outside physical
  solids expanded by the actual drone radius.
- The first real tick reports `fully_connected`, and increments hold from 0 to 1.
- Every executed tick remains fully connected, with exactly k active agents.
- The final real `success` is true and `chain_held_steps >= hold_chain_for`.
- No collision occurs and positions remain bitwise unchanged after injection.

With G01 defaults, idle termination and chain success can occur on the same
50th tick. A true runtime success at that tick is accepted; idle termination
before success is a failure. No idle or time-limit setting is overridden.

## Command, output and interpretation

The current constructor includes the corrections in
[assignment_corrections.md](assignment_corrections.md). The actual runtime
hold/collision/drift certification criteria below are unchanged. Current
reports also show CPU guard outcomes and distinguish missing finder routes
from invalid baseline assignments, while keeping both in the failure denominator.

```bash
uv run python max_geo_distance/run.py --tests runtime_chain_hold --runtime-device gpu
```

GPU is the default and must be available; there is no silent CPU fallback.
For an explicitly selected CPU check, use `--runtime-device cpu`. Batch size
defaults to 128 environments and is controlled by `--runtime-batch-size`.
First execution compiles a kernel for each distinct k. Console progress reports
those groups; all progress and results are saved in the same transcript.

Output includes proposed counts, first-tick connectivity, genuine chain-hold
success and strict certification rates by k, failures by reason and zero-based
sample indices, and representative failed float64/float32/final poses plus
runtime connectivity flags. The transcript is saved as
`results/<UTC timestamp>_10_runtime_chain_hold.txt`. Any uncertified target makes
the command exit with a nonzero code after saving the results. Earlier reports
are not printed and their selection/outputs are preserved.

A complete pass confirms these sampled, placed assignments in the tested
runtime, precision and device with zero actions. It does not certify navigation,
learning success, tolerance to control errors, globally minimum agent counts,
or all continuous spawn positions and future maps.
