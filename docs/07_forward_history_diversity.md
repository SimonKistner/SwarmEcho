# Forward-History Diversity for MAPPO

This document describes the optional diversity extension inspired by *Celebrating Diversity in Shared Multi-Agent Reinforcement Learning*.

The feature is disabled by default. With `diversity.enabled=false`, SwarmEcho builds and trains the same shared MAPPO actor-critic path as before. With `diversity.enabled=true`, the actor gets a small role-specific residual head and training receives an intrinsic reward from a shared-vs-role forward prediction objective.

## Goal

The paper maximizes mutual information between an agent identity and its trajectory:

```text
I(trajectory; identity)
```

The practical meaning is: if an observer watches what an agent does over time, the observer should be able to infer which identity or role produced that behavior. This encourages agents to specialize, but the paper also regularizes the non-shared part so agents remain mostly shared unless specialization helps.

SwarmEcho has a different base RL stack:

```text
paper:     discrete value-based QPLEX/QMIX-style learning
SwarmEcho: continuous-action MAPPO with a Gaussian actor and centralized critic
```

Because of that mismatch, this implementation does not port QPLEX. It adapts the core idea to the existing actor-critic architecture.

## Architecture

The existing actor remains the shared policy trunk:

```text
obs_i -> shared actor trunk -> shared Gaussian mean/log_std
```

When diversity is enabled, the actor adds a zero-initialized role residual to the Gaussian mean:

```text
mu_i = shared_mu(obs_i) + role_residual(role_i, shared_features_i)
```

The role residual starts at exactly zero, so enabling diversity does not initially change the behavior. The `diversity.l1_coef` loss then pressures this residual back toward zero during training. This is the MAPPO analogue of the paper's L1 regularization on the non-shared Q-functions.

Roles are assigned by index:

```text
role_i = agent_i % num_roles
```

If `diversity.num_roles=0`, the system uses one role per agent, which is closest to the paper's fixed identity setup. If `diversity.num_roles=3`, then agents are assigned to three reusable specialist groups. This allows training with 9 agents and later scaling the same role pattern to larger swarms, such as 90 agents with 30 agents per role.

## Forward-History Objective

The paper has an observation-aware diversity term that asks whether identity helps explain the next observation. This implementation uses the same idea with fixed-length local history.

At rollout time, each agent maintains a history window:

```text
history_i[t] = last H pairs of (obs_i, squashed_action_i)
```

The action stored in this history is the normalized tanh-squashed action before `max_force` scaling, because that is the action that determines the physics step up to a constant scale.

Two auxiliary predictors are trained:

```text
shared predictor:
  q_shared(next_obs | history)

role-aware predictor:
  q_role(next_obs | history, role)
```

Both are implemented as MLPs in `src/models/diversity.py`. They use fixed-variance Gaussian likelihoods, so the intrinsic signal is equivalent to the prediction-error advantage:

```text
r_intrinsic = 0.5 * (mse_shared - mse_role)
```

If the role-aware predictor predicts the next observation better than the shared predictor, the transition contains role-specific information and receives positive intrinsic reward. If the shared predictor is already as good, there is little or no reward for specialization.

Episode-boundary transitions are masked out so an auto-reset observation is not treated as a meaningful next observation.

## Training Data Flow

With diversity disabled:

```text
obs -> shared actor -> env step -> env reward -> buffer -> GAE -> PPO
```

With diversity enabled:

```text
obs -> shared actor + role residual -> env step -> next_obs
    -> update local history
    -> shared-vs-role prediction advantage
    -> env reward + beta * intrinsic reward -> buffer -> GAE -> PPO
    -> auxiliary predictor loss + role residual L1
```

Environment episode metrics still track the environment reward, not the intrinsic training reward, so success/coverage/chain metrics remain comparable to baseline MAPPO.

## Control Knobs

The feature is configured under `diversity` in `src/core/config.py`:

```yaml
diversity:
  enabled: false
  method: forward_history
  num_roles: 0
  history_len: 4
  beta: 0.05
  aux_coef: 1.0
  l1_coef: 0.001
  reward_clip: 1.0
  normalize_intrinsic: true
  adapter_scale: 1.0
  predictor_hidden_dim: 128
  predictor_num_layers: 2
```

Specialization is controlled by several independent levers:

- `num_roles`: how many distinct reusable specialists can exist.
- `adapter_scale`: how strongly role residuals can affect the actor mean.
- `l1_coef`: how strongly specialization is pushed back toward the shared policy.
- `beta`: how much intrinsic reward enters GAE and PPO.
- `reward_clip`: limits the immediate intrinsic reward magnitude.
- `normalize_intrinsic`: normalizes the intrinsic signal over valid agents each step.
- `history_len`: how much recent trajectory context the predictors can use.

Smaller `num_roles`, smaller `adapter_scale`, larger `l1_coef`, and smaller `beta` keep behavior closer to the shared baseline. Larger values allow stronger specialization but increase the risk of non-cooperative diversity.

## Relationship to the Paper

The paper trains on a fixed agent count with exact agent IDs and deploys the same identities afterward. This implementation supports that by setting:

```text
diversity.enabled=true
diversity.num_roles=0
```

For SwarmEcho, specialist groups are often a better fit than exact IDs because the intended use case is swarm-like and may need to scale across different numbers of drones. Reusable roles preserve the paper's central insight, "shared when possible, diverse when necessary", while avoiding brittle dependence on a specific agent count.

This is therefore best understood as:

```text
CDS-inspired forward-history MAPPO
```

not a reproduction of the paper's QPLEX-based CDS algorithm.

## Implementation Points

The main code changes are:

- `src/core/config.py`: adds the `DiversityConfig` block.
- `src/models/actor.py`: adds zero-initialized role residual adapters.
- `src/models/diversity.py`: adds shared and role-aware forward predictors.
- `src/models/mappo.py`: wires the optional diversity module into the model wrapper.
- `src/training/mappo_buffer.py`: stores `next_obs`, histories, role IDs, and masks only when enabled.
- `src/training/mappo_trainer.py`: adds auxiliary predictor loss and role-adapter L1 loss.
- `src/training/runner.py`: maintains rolling histories, computes intrinsic reward, and keeps baseline metrics on environment reward.

## Caveats

This extension should improve role differentiation if the task benefits from specialist behavior, but it does not guarantee the same gains as the paper. The paper's result depends on a full value-factorization stack. Here, the method is adapted to continuous MAPPO.

The likely outcomes are:

- More role-differentiated behavior: likely.
- Better exploration or relay-chain specialization on hard maps: plausible.
- Same performance gains reported by the paper: not guaranteed.
- Stable improvement on every level: uncertain.

For experiments, compare at least:

```text
baseline MAPPO
diversity enabled with num_roles=0
diversity enabled with num_roles=3
diversity enabled with beta=0.0
diversity enabled with l1_coef=0.0
```

The last two isolate the contribution of intrinsic reward and L1-controlled specialization.
