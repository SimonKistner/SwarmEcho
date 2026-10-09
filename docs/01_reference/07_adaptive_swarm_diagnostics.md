# Adaptive swarm diagnostics

These metrics are emitted to **Weights & Biases only** when adaptive size is enabled. `adapt_size/` metrics use the last `training.num_envs` **completed episodes** and first appear once that window is full. Ratios and success/failure subgroup means are omitted when their denominator or subgroup is absent; an absent series is not zero. `total_` values sum this sliding window, not the entire run. Compare them with `train/success_rate` and `train/ep_length`, which use the same completed-episode window.

| W&B key | Exact meaning | Healthy development / warning sign |
| --- | --- | --- |
| `adapt_size/calls_mean` | Successful reinforcement deployments per episode | Above zero on missions needing relays as success improves. High calls with flat success suggests growth without effective positioning. |
| `adapt_size/decommissions_mean` | Decommissions per episode | Can stay near zero if removal rarely helps. High values with repeated calls suggest churn. |
| `adapt_size/active_agents_at_EP_end_mean/all` | Mean active population **at episode end**, all episodes | Above the initial cohort on hard missions. This is not a time average; inspect success/failure splits. |
| `adapt_size/active_agents_at_EP_end_mean/success` | Same, successful episodes only | Enough drones to bridge the route without always ending at nine. Nine on easy targets suggests overcalling. |
| `adapt_size/active_agents_at_EP_end_mean/fail` | Same, failed episodes only | Low value plus low success suggests undercalling or early decommissioning; nine plus low success suggests poor positioning. |
| `adapt_size/called_decommissioned_mean` | Per-episode mean of `min(calls, decommissions)` | Rough churn count; high values suggest additions soon become unnecessary. Identities need not match. |
| `adapt_size/called_decommissioned_fraction` | Mean `min(calls, decommissions)/calls` over episodes with calls | High can mean overcalling and correction, or useful temporary search capacity. |
| `adapt_size/called_decommissioned_fraction_pooled` | Sum of `min(calls, decommissions)` divided by sum of calls | Weights high-call episodes more. A large difference from the preceding metric shows a few episodes dominate. |
| `adapt_size/total_calls` | Successful deployments in the window | Compare with total decommissions and success; compare only equal window sizes. |
| `adapt_size/total_decommissions` | Decommissions in the window | Rising toward or above calls implies churn or shrinking from the initial cohort. |
| `adapt_size/total_calls_to_decommissions_ratio` | Window calls divided by window decommissions | Above one means more additions than removals. Omitted when nobody decommissioned. |
| `adapt_size/decommission_termination_rate` | Fraction ending with zero active drones and no success | Should approach zero. A rise suggests reckless retirement or a sphere that is too easy to occupy inadvertently. |
| `adapt_size/rejected_calls_mean` | Capacity-rejected majority attempts per episode | Should remain low. Persistent growth means drones keep voting for reinforcements at full capacity. Each event requires fresh credits. |
| `adapt_size/useless_agents_at_success` | Active drones outside the selected contributing chain, averaged over successes | Ideally falls while success remains high. Near zero alone can mean the policy avoids calling and solves fewer targets. |

Reward series use the existing `rewards/` calculation: mean **episode total** in the same window, including zeros when no event occurred. Costs are negative and magnitudes depend on penalty settings and event counts. These series appear alongside other rewards rather than under `adapt_size/`.

| W&B key | Meaning and comparison |
| --- | --- |
| `rewards/reinforcement_cost` | Team cost for successful calls. B02c sets this to 50 per voted deployment; scheduled initial drones are exempt. Compare with `calls_mean`. |
| `rewards/decommission_cost` | Removed drones' decommission costs. B02c starts at zero. Compare with `decommissions_mean`. |
| `rewards/rejected_call_penalty` | Penalties paid by qualified YES voters on rejected majority batches. A more negative value with higher `rejected_calls_mean` diagnoses overcalling at capacity. |
| `rewards/all_decommissioned_penalty` | Team penalty when every drone is removed. Should tend to zero alongside `decommission_termination_rate`. |
| `rewards/useless_agetnst_at_success_penalty` | Team penalty per active noncontributor at success. The requested spelling `agetnst` is deliberate in the public W&B key; the config field is `success_unused_agent_penalty`. Compare with `useless_agents_at_success` and success rate. |

The useful overall pattern is **higher success across target distances**, shorter episodes, fewer rejected batches and fewer unused drones at success. A low unused-drone count with falling success can simply mean the policy avoids calling. A low call count with high success only on easy targets can recreate the original pass-through shortcut. Inspect hard-target evaluation and replay chains to see whether extra drones bridge real gaps.
