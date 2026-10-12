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

Every `rewards/*` component uses the same aggregation: sum the actual configured
reward across all drone slots and all steps of each episode, then average those
totals over the last `training.num_envs` completed episodes. These are raw,
undiscounted reward units after reward weights and sharing have been applied;
there is no division by episode length, active population, number of callers,
or event count in the chart calculation. Failed episodes and episodes without
the relevant event remain in the average. The actual reward components sum to
`train/ep_return`, up to floating-point rounding. Exclude the positive/negative
reference diagnostics from this sum: they break down the existing net reference
component and do not add another payout.

| Other reported reward keys | Payout included in the episode total |
| --- | --- |
| `rewards/coverage` | Exploration credit before the drone knows the target |
| `rewards/collision` | Collision penalties across drones and steps |
| `rewards/finder` | Individual finder/delivery bonuses |
| `rewards/target_found` | Shared target-found/delivery bonuses |
| `rewards/time_penalty` | Accumulated team time cost |
| `rewards/gap_reduction` | Configured recurring gap reward or signed changes in gap credit |
| `rewards/peer_informing` | Bonuses for newly informing peers |
| `rewards/success` | Success bonus actually paid, whether terminal or incremental hold records |
| `rewards/no_movement_termination` | Configured idle-termination reward or penalty |
| `rewards/efficiency` | Chain-efficiency payouts; emitted only when enabled |

All these components and the adaptive components below are directly comparable
at the same training point in the same run. For example, success `+800` and
reference `+3` mean the average net reference payout is `3 / 800 = 0.375%` of
the success payout. A penalty of `-400` has half the magnitude of that success
payout and offsets it in the return. For a comparison across every component,
use `abs(component mean) / sum(abs(all actual component means))`; exclude the
reference breakdowns to avoid double counting. Using the net return as the
denominator becomes misleading when rewards and penalties cancel.

A small net mean does not establish that the reward has little influence on
learning: positive and negative payments can cancel across steps, drones, and
episodes. Reference credit can also be absent in no-call episodes or expire
before success. These charts measure aggregate contribution to return, not
per-caller payments or policy-gradient importance. In the current C levels,
terminal success pays 12000 per successful episode, so a success chart near
800 corresponds to roughly 6.67% successes in that completed-episode window.

| W&B key | Meaning and comparison |
| --- | --- |
| `rewards/reinforcement_reward` | Signed rewards summed over qualified deployment voters. Each receives the full configured value; B02c uses -50 per voter. Initial spawns and rejected calls pay nothing from this term. |
| `rewards/decommission_reward` | Signed rewards summed over removed drones, each receiving the full configured amount on removal. B02c starts at zero. Compare with `decommissions_mean`. |
| `rewards/rejected_call_penalty` | Penalties paid by qualified YES voters on rejected majority batches. A more negative value with higher `rejected_calls_mean` diagnoses overcalling at capacity. |
| `rewards/all_decommissioned_penalty` | Team penalty when every drone is removed. Should tend to zero alongside `decommission_termination_rate`. |
| `rewards/useless_agetnst_at_success_penalty` | Team penalty per active noncontributor at success. The requested spelling `agetnst` is deliberate in the public W&B key; the config field is `success_unused_agent_penalty`. Compare with `useless_agents_at_success` and success rate. |
| `rewards/success_noncontributor_penalty` | Additional individual penalties at terminal chain success, summed across penalized drones per episode. C levels charge each active noncontributor `-4000`; contributors and inactive drones receive zero from this term. |
| `rewards/reward_by_reference` | Sum of the signed, linearly decaying reference shares paid independently to qualified callers. Includes the recruits' normal shared penalties and terminal rewards, but excludes their reference bonuses. Zero when disabled; negative values mean recruited drones accumulated net penalties while their references were active. |
| `rewards/reward_by_reference_positive` | Mean episode total of positive caller-recruit payments only. Negative payments contribute zero. Same units and completed-episode window as other reward charts; a diagnostic breakdown of the net reference reward. |
| `rewards/reward_by_reference_negative` | Mean episode total of negative caller-recruit payments only. Positive payments contribute zero; the value stays negative. Same units and window as the positive chart; a diagnostic breakdown, not another penalty. |

For both reference diagnostics, split each caller-recruit payment by sign on
each step, sum it per caller over the episode, sum across callers, then average
across the same completed-episode window. Split before combining payments from
different recruits, so a caller receiving `+10` from one recruit and `-10` from
another contributes `+10` and `-10` to the respective diagnostics rather than
zero to both. A payment's sign is based on the recruit's full normal reward for
that step, after its individual and shared reward terms have been combined.

`positive + negative = reward_by_reference`, up to floating-point rounding.
For example, `+300` and `-297` explain a net reference chart of `+3`: there are
substantial payments in both directions despite the small net value. Compare
the positive chart directly with success/exploration payout magnitudes and the
absolute negative chart with penalty magnitudes. These are reward outcomes
within the discounted reference window, including shared rewards and costs;
they do not isolate a recruit's causal effect on mission success.

The useful overall pattern is **higher success across target distances**, shorter episodes, fewer rejected batches and fewer unused drones at success. A low unused-drone count with falling success can simply mean the policy avoids calling. A low call count with high success only on easy targets can recreate the original pass-through shortcut. Inspect hard-target evaluation and replay chains to see whether extra drones bridge real gaps.
