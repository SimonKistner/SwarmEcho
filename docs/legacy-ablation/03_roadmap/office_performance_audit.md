# Office performance: baseline before optimization

## Reset dependency audit

The dense reference `make_autoreset_fns.autoreset_step` computes the terminal transition,
rewards and diagnostic fields first, calls `reset` unconditionally, then selects
the reset state only when that lane is terminal.

For a nonterminal lane, every fresh reset array is discarded. `reset` uses
explicit JAX keys; it does not mutate a global RNG, update counters outside its
returned state, or invoke callbacks. The physical step preserves `state.key`.
Consequently skipping a discarded reset does not advance or perturb its RNG.
Factory-time geometry caches and initialization logs are outside `reset`.

For a terminal lane, reset remains mandatory. Preserve all of these contracts:

- Sample the new target and advance the episode key with the same split order.
- Keep the existing generated layout and stored roadmap; autoreset currently
  reuses them instead of generating new obstacles.
- Reset position, velocity, spawn activity, coverage and coverage credit,
  knowledge, success, idle and chain counters exactly as the existing reset does.
- Rewards and terminal diagnostics must describe the old episode.
- Return the same `done` signal: training independently uses it for recurrent
  resets, base message memory and advantage termination. Do not change these.

The reference implementation remains available alongside `.batched`. Added a mixed-lane
regression test (ongoing, timeout, idle, success; with and without generated
obstacles) that compares it with explicit reset-only-on-terminal behavior,
including keys and layout arrays. Tests have not been executed here.

Training now uses `.batched`: compute transitions first, compact terminal indices,
reset them in groups of up to 32, and scatter only valid lanes back. Padding lanes
use out-of-bounds drop indices so they cannot overwrite a real lane. No reset
executes when none are terminal. When more than half are terminal, use the dense
reference path instead. Conditions are outside `vmap`. Keys, rewards and terminal
information are preserved; invariant layouts/roadmaps are not scattered.

Added repeated-transition comparisons for 0, 1, 17, 18 and 35 terminal lanes out
of 35, covering the sparse/dense boundary, padding, generated layouts and both
reward modes. Float comparisons allow 1e-6 numerical tolerance, except the
chain-progress percentage diagnostic: its subtract-near-one formula amplifies
float32 rounding, so its absolute tolerance is two float32 epsilons times 100
(approximately 0.000024 percentage points). The route-excess diagnostic permits
4e-6 metres absolute error because it subtracts nearly equal route lengths;
frontier indices must still match exactly. State and reward tolerances remain
1e-6; integer keys and boolean fields require exact equality. Failures identify
the transition and field. Authored static geometry is also covered.
These tests must pass on the user's runtime before accepting the optimization.

## Measured baseline and implemented geodesic changes

The saved `outputs/B01_perf_before/timings.jsonl` shows about 161 seconds of
rollout/transfer and 11 seconds of optimizer work per steady update. Final
evaluation/artifact work took 239.5 seconds separately. Rollout is the priority.

Geodesic distances now use one [base, target, agents] matrix per training
transition. Endpoint attachments are calculated once and the matrix is shared
by rewards and diagnostics. Min-plus reduction handles eight pivots per loop
iteration (122 iterations for the office's 972 vertices, previously 972).
All vertices/edges, clearance checks and the existing finite-INF behavior remain.
Tests compare the blocked reduction against the original single-pivot reduction,
including an office-sized matrix, and check the original frontier selection.

No speedup is claimed yet. Run the after benchmark and compare phase timings.

## Remaining candidates, after measuring these changes

1. Cache base/target attachments across steps only while their
   positions, geometry and clearance remain unchanged. XLA may already remove
   some duplicate expressions; source repetition alone is not a timing result.
2. Tune exact min-plus block size using measured speed and peak memory. Do not
   prune paths or coarsen the graph.
3. Cache initial coverage/connectivity pieces independent of the target, keyed
   by layout and spawn activity. Target-dependent discovery cannot be cached as
   part of an invariant reset template.

Keep environment count, rollout length, network, reward coefficients, voxel
resolution and wall-intersection tolerances unchanged. Evaluate each change
separately against fixed seeds/actions and all state fields, then benchmark.
Approximate geodesics or radar are separate compromises requiring discussion.

## Baseline command (activated WSL environment)

```bash
uv run swarmecho-train level=B01_office reward.chain_reward_system=obstacle_geodesic training.total_timesteps=4000000 training.profile_timing=true evaluation.save_model=true evaluation.eval_video=false evaluation.early_exit=false logging.wandb_mode=disabled logging.run_name=B01_perf_before
```

This runs ten updates with the existing 4,000 environments and 100-step rollout.
It produces `outputs/B01_perf_before/timings.jsonl`. Training records separate
rollout+transfer, buffer/advantages and synchronized optimizer durations.
Evaluation/artifact time is a separate record. The first update includes
compilation; compare subsequent individual updates, including terminal-heavy
updates, instead of the cumulative SPS. The forced final evaluation provides
a fresh heatmap. Video is disabled to avoid an additional final-checkpoint
evaluation. These reporting overrides must be identical in the after run.

When optimization is ready, repeat the same command with only the run name
changed to `B01_perf_after`. Both start from the same configured seed and fresh
weights. Neither restarts from the other's checkpoint. A short fresh-policy
benchmark is not a guarantee of performance after convergence; later compare
both implementations from the same saved checkpoint as well.

The old `B01_chunked` run had saving disabled and contains no checkpoint. After
the baseline finishes, its new checkpoint can be evaluated with:

```bash
uv run swarmecho-evaluate level=B01_office checkpoint=outputs/B01_perf_before/checkpoints/ckpt_000010 mode=parallel replay_after=false eval_name=spawn_check evaluation.eval_robustness_runs=1
```

This checkpoint is from the short benchmark, not the lost nine-hour run. Its
target distribution can validate the removed base-distance rule; its outcomes
are not a meaningful estimate of a fully trained policy's performance.
