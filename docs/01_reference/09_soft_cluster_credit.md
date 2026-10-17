# Progress-based soft cluster credit

`reward.progress_based_cluster_tiebreaker_soft` defaults to `false` and is
explicitly `true` in `G01_static`, directly below `chain_reward_system`.
It changes gap-reward allocation only. No leader labels, target positions,
vote information or other observations are added.

The disabled behavior still selects one shortest-hop relay path, resolving
equal-hop alternatives by slot order. The enabled behavior uses all equal-hop
alternatives between the same endpoints. Partial chains retain the existing
joint frontier-leader selection; complete chains use the base and target.
Longer detours and disconnected drones receive no gap credit.

## Candidate progress and local shares

At a fork in the shortest-hop graph, pretend each eligible next drone is the
frontier leader. Compare its remaining gap to the opposite frontier (or the
target for a complete chain). In obstacle-geodesic mode this uses the existing
free-space distance matrix from actual leader selection; Euclidean mode uses
straight-line distances. No additional roadmap query is performed.

For successors `j` of the same fork:

```text
score_j = -remaining_gap_j / max(comm_radius, 1e-6)
share_j = exp(score_j) / sum_k(exp(score_k))
```

Subtracting the maximum score before exponentiation keeps this numerically
stable. The common mission distance cancels, so smaller remaining gap means
greater hypothetical closed-route progress. One communication radius is the
fixed softness scale; there is no additional tuning parameter. Equal progress
gives equal shares, and better progress gives a smoothly larger share.

One unit of credit enters each selected endpoint-to-frontier route. Credit
splits at forks and recombines at merges. Consequently:

- Two equivalent parallel intermediates receive half credit each; three receive
  one third each.
- Drones traversed by every shortest route keep full credit.
- Separate clusters split independently; a second cluster does not divide a
  mandatory link's reward again.
- Progress weighting changes allocation, not the total gap reward of a fixed
  shortest-hop formation. Turning redundant alternatives into essential serial
  links can increase the number of full-credit positions.
- Relabelling drone slots does not change soft allocation, except that exact
  ties in the existing frontier-leader selection retain its previous semantics.

Recurring rewards multiply the usual closed-metres amount by each drone's
fractional credit. Gap-change rewards instead difference those weighted credited
metres. Delivery vesting records the same weighted credit at delivery, so its
initial snapshot and subsequent changes use matching units.

The toggle requires a target mission and `allow_redundancy_reward=false`.
Terminal success allocation, unused-agent penalties, the chain-contributor
observation and the `useless_agents_at_success` diagnostic continue to use the
existing single selected complete chain. A redundant drone can therefore receive
some gap reward while still counting as unused at success.

The feature changes rewards without changing model shapes. Checkpoint contracts
record it as a behavioral setting; loading older weights is supported with a
behavior-change warning when enabling it. Existing checkpoints without this
contract entry are interpreted as disabled.

## Verification status

The implementation has been inspected without running Python compilation checks,
training, GPU code or the WSL virtual environment. Runtime verification remains
to be performed in the project's WSL environment. Useful cases are symmetric
diamonds, unequal-progress alternatives, independent two- and three-way clusters,
slot permutations, partial chains on either side, complete chains and delivery
vesting with fractional credit.
