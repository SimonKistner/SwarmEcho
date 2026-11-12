# Future Features

## Geo path-predictor directions (backlog, 2026-11-01)

- [ ] **Inspector agent-belief playback.** Select a replay agent and a probe from
  the same policy checkpoint; show per-timestep path probabilities in a second
  rotatable building synchronized with replay. This extends beyond the prototype's
  retained snapshot examples.
- [ ] **Recurrent probe and representation comparisons.** Explore additional
  belief memory, sequence outputs and multiple reference routes as separate
  experiments. These are outside the first path-predictor prototype.

Details and optional research diagnostics are recorded in the
[path-predictor backlog](../../geo_experiments/probes/path-predictor/BACKLOG.md).

## Random-building pool preparation (backlog, 2026-10-29)

- [ ] **Automatically size CPU preparation workers to the allocation.** Replace
  the fixed four-worker ceiling with detection of usable CPU capacity from CPU
  affinity and cgroup quotas, with a configurable override. Log the detected
  capacity and selected worker count. Account for nested numerical-library
  threads and memory pressure; compare preparation time before assuming the
  largest worker count is fastest.

- [ ] **Make map-pool identity reproducible from the experiment seed.** For
  unchanged generation settings and geometry code, the same seed should resolve
  to the same buildings and training/held-out split regardless of which run
  creates the pool first or which compatible pools already exist. Define and
  persist the relationship between experiment seed, generation seed and pool
  identity while preserving frozen selections on resume.

These features remain deferred. G03 refinement now supports an explicit pinned
PFSS pool and a map-selection seed independent of the training seed; this keeps
the existing dataset fixed across paired training seeds. It does not yet define
deterministic pool creation from the experiment seed. See the
[combined-wave submission guide](../../cluster/README.md#g03-refinement-one-combined-wave-two-training-seeds).

## Chain contribution visibility

- [x] Add `env.observe_chain_contributor` (default off; enabled in B01 office).
  The scalar is one only when the drone knows the target, a complete chain
  exists, and the reward's contributor selector includes that drone.
  It follows the existing minimum-hop contributor rule, not geometric roadmap
  ranking. Enabling it adds one observation input; older checkpoints require
  the option off. Inspector links highlight the same selected paths in green.

- [x] Optional redundancy and route-efficiency rewards, both default off and
  enabled for B01 office:
  - `reward.allow_redundancy_reward`: when fully connected, all members of at
    least one simple base-target path receive the gap reduction bonus. Dead-end
    branches remain non-contributors. Before full connection the selected
    frontier paths receive the bonus.
  - `reward.enable_chain_efficiency_reward`: add
    `chain_efficiency_bonus / num_agents * clip(roadmap_length / relay_length, 0, 1)`.
    The default weight is `0.5`. Lengths are physical metres, without the
    spawn-separation corner allowance. Efficiency does not modify gap shaping.
  - Efficiency alone rewards one deterministic shortest physical relay route;
    redundancy plus efficiency rewards each drone's best containing route,
    without summing rewards across routes. Both off retain single-route credit.
  - With redundancy, the contributor observation and green inspector links
    include all valid simple paths. Replays save the mode in their metadata;
    older artifacts retain the original single-selection display.
  - Subset dynamic programming records the minimum path length and number of
    permutations for each visited-drone subset. It forbids repeated drones and
    avoids factorial path enumeration in training; cost still grows with `2^N`.
  - Only `train/chain_efficiency` (best route efficiency, when enabled) and
    `train/number_of_valid_paths` are added to W&B. Each uses terminal values
    from successful episodes in a `num_envs`-sized window, reported only once
    full. No additional terminal stats are added.

## Information-ranked selective evaluation renderer

Add a renderer that selects evaluation episodes from the comprehensive
evaluation CSV according to useful spatial or outcome criteria instead of
the removed first-come success/failure buckets, corner special case, or CSV
target loader. Candidate selections include:

- nearest failure to the base;
- furthest successful target from the base;
- failures and successes near the decision boundary;
- spatial outliers or representatives of dense failure regions.

The evaluation CSV already provides the initial data needed for these
selections: target coordinates, one of four terminal outcome stages, and
Euclidean target-to-base distance. A future implementation should finish the full
parallel evaluation first, rank its rows, and only then collect videos for the
chosen targets. This keeps metric collection independent of rendering.

If the future renderer must reproduce the exact original episode rather than
re-evaluate a selected target position, the saved data must also include and
reuse the original reset PRNG key. Deterministic policy actions alone do not
reproduce randomized base and drone spawns. Alternatively, the CSV schema could
store the realized base and drone spawn positions needed for exact replay.
