# Adaptive swarm diagnostics

Reward components use the signed individual system described in [signed individual rewards](11_individual_rewards.md). Historical run series retain their original names and values.

These metrics are emitted to **Weights & Biases only** when adaptive size is enabled. `adapt_size/` metrics use the last `training.num_envs` **completed episodes** and first appear once that window is full. Ratios and success/failure subgroup means are omitted when their denominator or subgroup is absent; an absent series is not zero. `total_` values sum this sliding window, not the entire run. Compare them with `train/success_rate` and `train/ep_length`, which use the same completed-episode window.

The legacy `train/no_movement_termination_rate` has different units: it is
already a **percentage** (`100 * mean(idle_flag)`), so a logged value of `0.18`
means 0.18%, not 18%. It records the idle threshold flag without excluding
simultaneous success. It must not be interpreted as a pure idle-failure rate.

| W&B key | Exact meaning | Healthy development / warning sign |
| --- | --- | --- |
| `adapt_size/calls_mean` | Successful reinforcement deployments per episode | Above zero on missions needing relays as success improves. High calls with flat success suggests growth without effective positioning. |
| `adapt_size/max_agents_called_rate` | Fraction of window episodes with at least `k - adapt_size.initial_agents` successful calls, where `k` is `env.relay_agent_budget` (or `env.num_agents` when unset) | Shows how often the policy called enough reinforcements for the stage's target budget. |
| `adapt_size/max_agents_plus_one_called_rate` | Fraction of window episodes with at least `k + 1 - adapt_size.initial_agents` successful calls when the deployable cap exceeds `k` | Shows how often the policy called one agent beyond the target budget. Omitted when the cap equals `k`. |
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
| `adapt_size/rejected_calls_mean` | Capacity-rejected quorum attempts per episode | Should remain low. Persistent growth means drones keep voting for reinforcements at full capacity. Credits mode requires another delivered batch; latest-decision mode closes the rejected round and requires new-round approvals. |
| `adapt_size/useless_agents_at_success` | Active drones outside the selected contributing chain, averaged over successes | Ideally falls while success remains high. Near zero alone can mean the policy avoids calling and solves fewer targets. |

Both rates range from 0 to 1 and use successful call totals, including replacement calls after decommissioning. They do not assert that `k` or `k+1` drones were active simultaneously. In G02, the cap is `k+1` until the final stage, where it equals `k`.

With dynamic relay-budget filtering and adaptive size enabled, W&B logs each
target's **exact accepted estimator count** in numeric category order. Each
`req_agent_K` group emits these metrics in this order:

| W&B key | Meaning |
| --- | --- |
| `req_agent_K/total_eps_in_eps_window` | Fraction of the ordinary `training.num_envs` completed-episode window whose target has estimator count `K`. |
| `req_agent_K/active_agents_at_EP_end_mean` | Mean active, deployed population at episode end in category `K`'s separate smaller window; includes successes, failures and the initial drone. |
| `req_agent_K/success_rate` | Successful-episode fraction in that same smaller category window. |

`K` is the estimator result for that episode's actual base/target/map, not the
current curriculum stage, population cap, or a proven physical minimum. For
example, an episode with estimated count five and six active drones at its end
adds six to the `req_agent_5` population window. Categories run from one through
the current geometric budget, including counts below the stage. The spare `k+1` slot
does not change category membership and the measured population is never clipped
to `K`.

`logging.required_agents_metrics_window_episodes` sets the smaller category
window size (default 500), independently of `training.num_envs` and the cell
controllers. Population and success means report once the category has any
completed episodes, using up to its last 500. An unobserved category emits no
population or success mean. Category proportions wait for the ordinary episode
window to fill, then include zero for categories absent from that window and
sum to one. Thus 4000 training environments use the last 4000 completions;
after OOM reduction to 3000, proportions use the last 3000. Both proportions and
success rates are fractions in [0, 1], not percentages. No per-category
`episodes_in_window` metric is emitted.
Only terminal transitions are counted, once each, including episodes crossing
rollout boundaries. Decommissioned drones are excluded from the end population;
replacements are not counted a second time as lifetime deployment totals.
Windows start empty on each training invocation, including resume, matching the
other host-only adaptive diagnostics rather than restoring controller state.

The accepted estimator count is cached at reset, carried with the episode and
read before autoreset. These diagnostics add no estimator queries, observations,
reward changes or population-control changes. Coverage and unfiltered episodes
have no category and do not emit these diagnostics. Existing saved runs are not
backfilled.

Reward series use the existing `rewards/` calculation: mean **episode total** in the same window, including zeros when no event occurred. Costs are negative and magnitudes depend on penalty settings and event counts. These series appear alongside other rewards rather than under `adapt_size/`.

Every `rewards/*` component uses the same aggregation: sum the actual configured
reward across all drone slots and all steps of each episode, then average those
totals over the last `training.num_envs` completed episodes. These are raw,
undiscounted reward units after reward signed coefficients and local attribution have been applied;
there is no division by episode length, active population, number of callers,
or event count in the chart calculation. Failed episodes and episodes without
the relevant event remain in the average. The actual reward components sum to
`train/ep_return`, up to floating-point rounding. Exclude the positive/negative
reference diagnostics from this sum: they break down the existing net reference
component and do not add another payout.

| Other reported reward keys | Payout included in the episode total |
| --- | --- |
| `rewards/exploration_reward` | Exploration credit before the drone knows the target |
| `rewards/collision_reward` | Individual collision rewards across drones and steps |
| `rewards/finder_reward` | Individual finder/delivery rewards |
| `rewards/target_found_reward` | Full configured target-found/delivery reward per active drone; the event's team total is `target_found_reward * active_agents` |
| `rewards/time_presure_reward` | Individual time pressure summed across acting drones and steps |
| `rewards/gap_reward` | Configured recurring gap reward or signed changes in gap credit |
| `rewards/peer_informing_reward` | Locally attributed rewards for newly informing peers |
| `rewards/success_reward` | Individual success rewards actually paid, whether terminal or incremental hold records |
| `rewards/no_movement_termination_penalty` | Configured idle-termination reward or penalty |
| `rewards/chain_efficiency_reward` | Chain-efficiency payouts; emitted only when enabled |

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
per-caller payments or policy-gradient importance. In G02, gross terminal success pays 3000 times the active population. A success-reward chart therefore cannot be converted into success rate without accounting for the populations of successful episodes. Useless-agent adjustments appear separately.

| W&B key | Meaning and comparison |
| --- | --- |
| `rewards/reinforcement_reward` | Signed rewards summed over qualified deployment voters. G02 uses +10 per voter on necessary grants. Enabled vote shaping suppresses this term on unnecessary grants. Initial spawns and rejected calls pay nothing from this term. |
| `rewards/withheld_reinforcement_vote_reward` | Eligible informed, connected NO decisions, charged only on communication ticks while recruitment is needed. G02 pays -1 per source per tick. Emitted only with reinforcement vote shaping enabled. |
| `rewards/unnecessary_reinforcement_call_reward` | Responsible qualified YES sources on unnecessary granted calls. G02 pays -200 per source, using decision-time provenance to exempt uninformed or previously necessary approvals. Emitted only with reinforcement vote shaping enabled; capacity rejections do not stack this term. |
| `rewards/decommission_reward` | Signed rewards summed over removed drones, each receiving the full configured amount on removal. G02 uses zero. Compare with `decommissions_mean`. |
| `rewards/rejected_call_reward` | Penalties paid by qualified YES voters on rejected majority batches. A more negative value with higher `rejected_calls_mean` diagnoses overcalling at capacity. |
| `rewards/all_decommissioned_penalty` | Full signed amount to every last acting drone when all are removed. Should tend to zero alongside `decommission_termination_rate`. |
| `rewards/useless_agent_at_success_reward` | Individual terminal chain-success adjustment, summed across recipients. G02 pays -3000 to each useless active drone; contributors and inactive drones receive zero from this term. |
| `rewards/useless_agent_at_success_all_agents_reward` | All-agents terminal chain-success adjustment, summed across recipients. Each active drone gets `-min(750 * useless_count, 2000)` in G02, so the episode total is active count times that capped individual amount. Inactive drones receive zero. Included in normal reward and recorded replay components; the cap is not a separate reward component. |
| `rewards/reward_by_reference` | Sum of the signed, linearly decaying reference shares paid independently to qualified callers. Includes the recruits' normal signed rewards, including terminal adjustments, but excludes their reference bonuses. Zero when disabled; negative values mean recruited drones accumulated net penalties while their references were active. |
| `rewards/reward_by_reference_positive` | Mean episode total of positive caller-recruit payments only. Negative payments contribute zero. Same units and completed-episode window as other reward charts; a diagnostic breakdown of the net reference reward. |
| `rewards/reward_by_reference_negative` | Mean episode total of negative caller-recruit payments only. Positive payments contribute zero; the value stays negative. Same units and window as the positive chart; a diagnostic breakdown, not another penalty. |

For both reference diagnostics, split each caller-recruit payment by sign on
each step, sum it per caller over the episode, sum across callers, then average
across the same completed-episode window. Split before combining payments from
different recruits, so a caller receiving `+10` from one recruit and `-10` from
another contributes `+10` and `-10` to the respective diagnostics rather than
zero to both. A payment's sign is based on the recruit's full normal reward for
that step, after its individual reward terms have been combined.

`positive + negative = reward_by_reference`, up to floating-point rounding.
For example, `+300` and `-297` explain a net reference chart of `+3`: there are
substantial payments in both directions despite the small net value. Compare
the positive chart directly with success/exploration payout magnitudes and the
absolute negative chart with penalty magnitudes. These are reward outcomes
within the discounted reference window, including locally attributed rewards and costs;
they do not isolate a recruit's causal effect on mission success.

The useful overall pattern is **higher success across target distances**, shorter episodes, fewer rejected batches and fewer useless drones at success. A low useless-drone count with falling success can simply mean the policy avoids calling. A low call count with high success only on easy targets can recreate the original pass-through shortcut. Inspect hard-target evaluation and replay chains to see whether extra drones bridge real gaps.
