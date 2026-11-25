# Removed policy and optimization variants

The [MARL reference](../03_marl_and_training.md) defines the recurrent actor, observation-based centralized critic and PPO path used now.

| Removed alternative | Historical change to the experiment |
| --- | --- |
| Feed-forward actor or critic | Removed recurrent history from one learning component |
| Global mean critic | Pooled team features into a shared scalar value instead of producing agent-specific estimates |
| Privileged graph-masked critic | Used an earlier critic input/masking design with simulator-derived information |
| Communication disabled | Removed learned peer messages from policy execution |
| Alternate communication cadence | Sent or consumed messages at a configurable multi-step interval |
| Self-attention in peer communication | Allowed the sender's own token into its received TarMAC context |
| Stored base-memory replay | Reused recorded base-cache values as detached inputs during PPO replay |
| Legacy entropy estimate | Used the earlier stored-action correction instead of the current sampled squashed-distribution entropy path |
| Raw value targets | Optimized values without the current running target normalization |
| Disabled value clipping | Bypassed PPO's clipped critic objective |
| Training action noise | Added an explicit exploration-noise path outside the policy distribution |

Stored base memory changes the gradient path through a cached packet. Differentiable replay uses recorded writer identities to reconstruct writes from recurrent policy outputs within the rollout. This distinction matters for interpreting an experiment's message-learning mechanism.

Earlier architecture discussions also proposed flat perfect-state or graph neural critics. A proposal alone establishes no implemented ablation or measured outcome. The current critic's shared token processing and per-agent attention preserve parameter sharing and permutation-equivariant value prediction, as explained in the maintained reference.

Configurable auxiliary enable/disable selectors and explicit output-grid settings were retired. The current auxiliary training mechanism and automatic shape derivation are documented under [path prediction](../03_marl_and_training.md#auxiliary-geodesic-path-prediction). A detached head remains obtainable with `lambda_target=0`, so it is a current experimental control.
