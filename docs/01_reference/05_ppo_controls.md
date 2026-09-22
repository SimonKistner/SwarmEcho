# PPO controls and diagnostics

The runtime uses `TrainingConfig` in `core/config.py` and
`training/ppo.py`. The generic MAPPO trainer is retained for the workflow validator.

## Controls

```yaml
training:
  actor_clip_eps: 0.2
  value_clip_eps: 0.2 # null disables value clipping; zero does not
  entropy_mode: legacy # legacy | squashed
  value_normalization: none # none | running
  diagnostics_every: 1 # rollout updates between diagnostic passes
env:
  observe_current_timestep: false # step / max_steps
```

`B01_office_find_only.yaml` enables `squashed`, `running`, and the timestep
observation. Both clipping thresholds remain 0.2. Old `clip_eps` overrides
must be replaced with the two explicit parameters.

* Actor clipping applies to the probability ratio and the actor clip-fraction
  diagnostic. Gradient clipping remains independently controlled by
  `max_grad_norm`; actor and critic still share that gradient norm and Adam.
* Value clipping uses raw reward units with `none`, and normalized units with
  `running`. The critic objective remains half the maximum of ordinary and
  clipped squared errors. `null` selects ordinary half-MSE.
* `legacy` retains the existing Gaussian entropy plus the stored-action tanh
  correction. `squashed` uses fresh reparameterized current-policy samples,
  including the gradient through the tanh correction. Sampling uses independent
  keys and never advances the rollout RNG.
* `running` fits normalized targets. Pooled active-agent target moments are
  merged once per rollout and frozen for its optimizer epochs. These are
  cumulative running moments, not per-minibatch statistics. Rollout predictions
  and GAE targets remain in raw reward units. The variance floor is 1e-4.
  This is ordinary value normalization, not PopArt: changing statistics can
  change the raw interpretation of an unchanged output head.
* Rewards and episode returns are not normalized. Advantage normalization uses
  only active samples, independently of the value-normalization setting.

## Always-applied corrections

Communication uses each environment's episode step, including across rollout
boundaries, auto-resets, evaluation, and replay. `memory_comm_every_k_steps`
retains its existing meaning; step zero is a communication opportunity.

Inactive agents have zero observations, recurrent state, messages, actions,
and value outputs. They are excluded from critic attention, actor/critic losses,
advantage/value statistics, entropy, and PPO diagnostics. Shared reward terms
are divided among currently active agents, preserving team reward magnitude;
inactive reward entries are zero. A newly spawned agent first acts using its
next observation, rather than executing an action chosen while inactive.

The reward defaults live in `RewardConfig`, including
`no_movement_termination_penalty=-1000`. Level overrides still win.
The no-movement penalty is included in `rewards/no_movement_termination`.

The task's `max_steps` deadline remains terminal (zero bootstrap), as requested.
Adding timestep observation changes observation width and checkpoint compatibility.

The raw-value/legacy-entropy settings preserve those objective definitions.
They do not undo the explicitly requested timing, inactive-agent, and reward-default
corrections. The timestep feature remains optional.

## Small diagnostic set

| W&B metric | Interpretation |
| --- | --- |
| `critic/rmse_before`, `critic/rmse_after` | Ordinary prediction RMSE in raw reward units, on the same fixed rollout targets before and after optimization. Normalization statistics are already updated and frozen for both passes. |
| `critic/relative_mse_after` | Ordinary MSE / target variance; NaN when variance is too small to interpret. Includes systematic prediction bias. |
| `critic/value_clip_blocked_fraction` | Active samples where the clipped branch dominates strictly outside the value-change threshold, flattening that sample's critic gradient; averaged over optimization. |
| `policy/gaussian_entropy` | Current unsquashed Gaussian differential entropy. |
| `policy/squashed_entropy` | Fresh-sample current bounded-action differential entropy; negative values are valid. |
| `policy/action_saturation_x/y/z` | Fraction of active stored rollout action components with absolute tanh output > 0.99; these describe the behavior policy, not post-update resampling. |

Diagnostic sampling is separate from training sampling and does not mutate
model state. Existing `ppo/*` metrics remain update averages, weighted by active
sample count. `ppo/value_loss` describes the objective in its configured units,
not the raw RMSE. Compare the new RMSE metrics across normalization modes.

## Optional privileged critic

Set `network.critic_type: privileged` (or CLI `network.critic_type=privileged`)
to enable the compact state input. The default `observation` keeps the existing
critic network, inputs, initialization, and PPO calculation; no privileged
features are collected or stored. Level defaults are not changed.

Before the critic encoder, GRU and all-active-agent attention, each agent receives
its existing observation plus normalized position, six adjacent coverage cells,
the true relative target vector, a relative base vector unless already observed,
chain-holding progress, base target knowledge, and episode progress unless already
observed. Coordinates use the longest map dimension as their scale, consistent
with actor base/target vectors. Coverage lookups clamp at grid boundaries.
There is no inactivity counter, actor memory/message input, new raycast, path
search, or full coverage-map copy. Actor observations and evaluation actions
receive no privileged information. The removed graph-masked privileged classes
are not used by this path.

Rollouts store nine floats per agent and eight shared floats per environment
(nine when episode progress is absent from actor observations). Shared data is
broadcast only during critic computation; existing observations are not copied
into a separate privileged buffer. Collection uses pre-action state, with the
same feature assembly for PPO replay and final-state bootstrapping. Inactive
agent tokens are zeroed and excluded from attention and losses as before.

Privileged checkpoints record their input layout. Switching critic types requires
a compatible checkpoint or a fresh run; older observation checkpoints still load
under the original observation settings. No throughput guarantee is made without
a matched benchmark.

## Checkpoint persistence

Running mean, variance, and count are saved as non-parameter model state.
`training_contract.json` records critic units, architecture, observation semantics,
and relevant behavioral settings. Incompatible units/input representations fail
before restore; changes such as entropy mode or clipping thresholds warn.
Older checkpoints without metadata warn in raw mode and are rejected in running
mode to avoid silently reinterpreting their critic output. No optimizer-state
checkpointing or automatic architecture/unit conversion is added.

Focused regression checks are in `tests/test_ppo_controls.py`; they were added
without executing training or tests during implementation.
