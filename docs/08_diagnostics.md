# Diagnostics Guide

## IC3 communication diagnostics

These metrics are emitted only when `network.memory_comm_frequency_control: ic3` is active. IC3 gates agent-agent TarMAC sender tokens inside the normal `memory_comm_every_k_steps` communication slots; physical/wall-aware range still comes from the environment adjacency matrix, and base replay also follows `memory_comm_every_k_steps`.

- `comm/ic3_rate_mean`: Mean completed-episode communication rate in the training sliding window. For each completed episode, this is `active communicate gates / active IC3 communication opportunities`, where opportunities exist only on `memory_comm_every_k_steps` communication slots and inactive agents are excluded.
- `comm/ic3_rate_std`: Standard deviation of the completed-episode rates in the same window.
- `comm/ic3_always_true_episode_pct`: Fraction of completed episodes whose IC3 rate is at least `network.ic3_comm_always_threshold`. This catches collapse to always speaking.
- `comm/ic3_always_false_episode_pct`: Fraction of completed episodes whose IC3 rate is at most `1 - network.ic3_comm_always_threshold`. With the default threshold `0.95`, this means rates `<= 0.05`. This catches collapse to silence.
- `comm/ic3_rate_when_target_known`: Mean completed-episode gate rate restricted to IC3 communication opportunities where at least one agent already knew the target. Episodes with no such opportunity use a safe zero denominator contribution.
- `comm/ic3_rate_success`: Mean IC3 rate among successful completed episodes in the training window. Use as correlation-only context, not causal attribution.
- `comm/ic3_rate_failure`: Mean IC3 rate among failed completed episodes in the training window. Use as correlation-only context, not causal attribution.
- `comm/ic3_gate_prob_mean`: Mean learned probability of the communicate action over completed episodes. This can diverge from sampled rate early in training.
- `comm/ic3_gate_entropy_mean`: Mean entropy of the IC3 gate distribution over completed episodes. Low values indicate confident gate decisions.
- `eval/comm/ic3_rate_mean`: Deterministic-evaluation communication rate using the same opportunity normalization as training.

## Adaptive spawn diagnostics

Adaptive spawn diagnostics are grouped to make category-by-category control easier to scan.

For each path-length category, `adaptive_spawn_control` reports values in category-major order:

- `adaptive_spawn_control/spawn_success_rate/category_XXX`: Recent success rate for the category.
- `adaptive_spawn_control/spawn_configured_rate/category_XXX`: Controller-configured sampling probability for the category.
- `adaptive_spawn_control/spawn_actual_pct/category_XXX`: Actual fraction of sampled completed episodes in the category.

Raw counts are separated under:

- `actual count/spawn_actual_count/category_XXX`: Number of completed episodes assigned to the category in the diagnostic window.
