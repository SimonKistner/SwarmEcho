# Connectivity Ownership

Physics is the single communication-graph authority. This document records
the former split, the maintained ownership contract, and the deterministic
before/after verification.

## Maintained ownership

| Data or consumer | Maintained source |
|---|---|
| Direct drone/drone and drone/base edges | Physics range, activity, and DDA wall checks; stored as `EnvState.adj_matrix` in `[drones..., base]` order. |
| Direct drone/target visibility | The same physics pass; stored as the compact `EnvState.directly_sees_target` vector. |
| Multi-hop base/target flags | Temporary physics reachability; final vectors stored as `is_conn_base` and `is_conn_target`. |
| Target-knowledge propagation | Reuses the temporary full physics reachability result. |
| Finder-path memory propagation | Uses physics-owned direct drone edges while preserving the task-specific rule that the base is not a path-memory relay; its drone-only closure is temporary. |
| Actor observations and TarMAC | Consume stored flags and adjacency. |
| Reward delivery and contributor routing | Consume stored direct edges and target visibility; shortest routes therefore respect walls. |
| Evaluation renderer | Translates stored physics nodes into `[base, target, drones...]` order without CPU raycasting. |

Reset computes the same graph facts as every post-movement state. It does not
pre-propagate persistent target knowledge: `target_known` and
`base_target_known` still begin false, preserving the episode knowledge
contract.

Only one `(N+1) x (N+1)` direct adjacency matrix is retained in the current
state. The missing target endpoint costs one `(N,)` boolean vector.
Reachability results are temporary and are not stored across steps or added
to the rollout buffer.

## Why the duplication likely exists

The original consumers operated in different execution domains:

- training physics, observations, rewards, and TarMAC needed static-shape JAX
  arrays inside compiled GPU work;
- the fast renderer needed NumPy data in CPU worker processes;
- physics stored only the drone/base adjacency, while reward and renderer path
  displays also needed a target endpoint;
- reward calculation needed one deterministic route for credit assignment,
  not only reachability flags.

Those were real interface differences, but they did not justify competing
decisions. Evaluation already transferred `EnvState.adj_matrix` to the CPU
renderer, and only direct target visibility was missing. Retaining that compact
vector allowed all consumers to share the physics result.

No earlier diagnostic survived the obsolete-test cleanup. The maintained
before/after harness described below was added specifically for this ownership
change.

## Implemented consolidation

The implemented ownership change:

1. Uses one JAX graph calculation producing direct drone/base adjacency,
   direct drone-target visibility, reachability, and base/target-connected
   flags. Persist only the already-required adjacency, the two final
   per-drone flags, and—if consumers require it—the missing per-drone direct
   target-visibility vector. Reachability remains temporary; it must not become
   another stored `(N+1) x (N+1)` matrix.
2. Computes it in physics for reset and every post-movement state.
3. Has observations consume the stored flags instead of recomputing them.
4. Builds reward contributor paths from the stored wall-aware direct graph plus
   stored target-visibility edges, preserving the existing deterministic
   tie-break.
5. Has the CPU renderer consume the stored graph data without raycasting
   dynamic communication edges again merely to draw them.

This retains one GPU calculation and one CPU representation of its result—not
competing GPU and CPU decisions. It also makes the first reset
observation, TarMAC mask, rewards, and rendered explanation refer to the same
state.

## Before/after verification

The harness uses a deterministic episode from:

```text
outputs/tiny_maze_nCP_simpleCurr_v6_seed_1/checkpoints/ckpt_000250
```

That checkpoint uses the current 37-dimensional observation. Its Orbax weights
use 64/128 TarMAC signature/value widths, even though the run's saved YAML says
16/32. This is evidence that the historical constructor defaults, rather than
those YAML fields, shaped the saved model. The verifier derives weight-shaping
settings from Orbax metadata and prints any override it applies before restore.
Its training map was the file now retained as `M01_small_maze`. The harness
deliberately uses the maintained level for environment behavior instead of
loading the run's stale pre-cleanup configuration wholesale.

Record the before-change baseline:

```bash
uv run swarmecho-verify-connectivity baseline=record
```

After a connectivity refactor, compare the same checkpoint, seed, target spawn,
trajectory, and subsystem decisions:

```bash
uv run swarmecho-verify-connectivity baseline=compare strict=true
```

The command writes under the checkpoint run:

```text
artifacts/verification/connectivity/ckpt_000250_seed0/
├── baseline.json
├── current.json
└── vids/
```

The JSON contains every state position, physics adjacency and reachability
flags, observation flags, reward completion/contributor output, CPU reference
result, renderer-model result, and per-frame comparison. The video shows the
normal colored links plus the physics connection matrix. This one diagnostic
episode intentionally retains detailed frame data; neither training nor the
parallel benchmark accumulates those JSON graph copies over a long run.

The same command also runs a no-render benchmark by default. It measures a
JIT-compiled reset followed by batched observations, environment steps, and
rewards over 4,000 environments and 100 steps. It excludes policy inference,
PPO, host trajectory conversion, JSON generation, and rendering. Compilation
is reported separately, and the median of five synchronized timed repeats is
stored in the baseline. On `baseline=compare`, the verifier reports the
throughput percentage change but does not fail on performance alone.

Benchmark size and repetition count can be adjusted when necessary:

```bash
uv run swarmecho-verify-connectivity baseline=record \
  benchmark_envs=4000 benchmark_steps=100 \
  benchmark_warmup=2 benchmark_repeats=5
```

Use `benchmark=false` only when recording a behavior-only diagnostic. Baseline
and comparison benchmark dimensions must match.

The full report intentionally differs from the pre-change baseline because
reset state, reward routing, and renderer semantics were corrected. The
verifier separately compares deterministic positions, spawn state, and target
knowledge; this behavior projection must match. The command exits unsuccessfully
for a behavior difference, while approved connectivity-field differences are
reported without treating them as a failure. `strict=true` additionally
requires all current consumers to agree.

For a different compatible checkpoint, pass `checkpoint=<path>` and the
matching current `level=<name>`. Additional current config overrides are
accepted using normal dotted syntax. The harness checks checkpoint observation
and action widths before restoration, derives the remaining weight-shaping
architecture from Orbax metadata, and rejects incompatible environment-facing
dimensions clearly.
