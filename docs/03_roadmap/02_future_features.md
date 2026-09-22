# Future Features

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
    least one simple base-target path get zero gap penalty. Dead-end branches
    remain non-contributors. Before full connection the old shaping is retained.
  - `reward.enable_chain_efficiency_reward`: add
    `chain_efficiency_bonus / num_agents * clip(roadmap_length / relay_length, 0, 1)`.
    The default weight is `0.5`. Lengths are physical metres, without the
    spawn-separation corner allowance. Efficiency does not modify gap shaping.
  - Efficiency alone rewards one deterministic shortest physical relay route;
    redundancy plus efficiency rewards each drone's best containing route,
    without summing rewards across routes. Both off retain the original reward.
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
