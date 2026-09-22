# Defense of the Current MAPPO + Agent-Centric Critic Architecture

> Design rationale for the observation-based critic. The current runtime also
> supports an optional compact privileged critic; see
> [PPO controls](05_ppo_controls.md). This rationale does not restrict that option.

## Position

SwarmEcho should keep its current MAPPO architecture with a decentralized shared actor and an Agent-Centric Centralized Critic (ACC) over the joint observation stack.

This is the right architecture for this project because the core research problem is not simply to solve a simulator with privileged information. The goal is to learn scalable, interchangeable swarm behavior from the same local information structure that would exist in deployment. The current ACC gives the critic centralized training context without breaking the agent-number-invariant, observation-grounded nature of the task.

In short:

```text
Actor: local observation only
Critic: all agents' local observations, processed as an unordered set of agent tokens
Output: one value estimate per agent
```

That is exactly the compromise SwarmEcho needs: centralized training, decentralized execution, permutation-equivariant credit assignment, and no dependence on hand-crafted simulator oracle state.

## Why This Is Unusual

Many MAPPO implementations use a privileged global state for the critic. That is often reasonable in games or benchmarks where a clean global state is part of the environment definition. In SwarmEcho, however, a hand-crafted perfect state would be a design intervention: it would decide which simulator facts matter, how they are encoded, and which hidden facts the critic is allowed to use.

That is not neutral. It changes the training problem.

SwarmEcho's current ACC is unusual because it asks the critic to learn value estimates from the agents' joint perceptual evidence rather than from an externally curated world-state vector. This is intentional. The architecture tests whether the swarm can learn relay behavior from information that is observable, communicable, and compatible with deployment.

## Why Joint Observations Are Already Highly Informative

The critic does not see a single agent's observation. It sees the full stack:

```text
(o_1, o_2, ..., o_N)
```

Each observation already contains the main state variables required for the task:

- the agent's own motion state;
- relative base geometry;
- base-connectivity and target-connectivity indicators;
- target-known state;
- target-relative information once known;
- local coverage samples;
- radar information about obstacles, agents, and chain-relevant drones.

Because all agents' observations are available to the critic, phase transitions are also visible. If any agent discovers or knows the target, the critic can attend to that token. If a chain is forming, the critic sees the base-connected and target-connected structure through the observation stack. If success is near, the relevant connectivity pattern is already present.

The privileged state would therefore not add a fundamentally different task description in most cases. It would mostly provide cleaner or more exact encodings of quantities that the ACC can already infer.

## Why ACC Is Better Than a Flat Privileged World State

A flat perfect-state critic would typically concatenate agent positions and other simulator variables:

```text
[x_1, y_1, v_1, x_2, y_2, v_2, ..., x_N, y_N, v_N, target, base, ...]
```

This is a poor inductive bias for SwarmEcho.

- It makes ordering matter unless carefully corrected.
- It creates accidental agent identities.
- It does not naturally scale from small swarms to large swarms.
- It forces an MLP to rediscover set and graph structure from a flat vector.
- It conflicts with the design goal that agents are interchangeable.

The current ACC instead treats agents as tokens. The same encoder is applied to every agent, attention computes interactions between tokens, and the shared value head outputs one value per agent. This preserves the required symmetry:

```text
renaming agents renames value outputs, but does not change the learned rule
```

That is the correct structure for a swarm.

## Why Attention Is the Right Bias

Relay-chain formation is not a simple global average problem. The value of an agent depends on relationships:

- distance to base and target-side information;
- distance to nearby agents;
- whether other agents bridge a communication gap;
- whether the agent is redundant, isolated, or useful;
- how local neighborhoods compose into a global chain.

Simple summation or mean pooling can capture coarse team statistics, but it is too blunt for this topology-sensitive task. The critic must be able to focus on the agents that matter for a particular value estimate.

Attention gives the ACC that ability while remaining permutation-equivariant. For agent `i`, the critic can condition on `i` and selectively attend to the other agents most relevant to its current value. This is especially well matched to chain formation, where the important neighbors change over time and depend on geometry.

Graph neural networks would also be a defensible alternative. A graph-attention or message-passing critic over communication edges is conceptually close to the current design. But a flat handcrafted world state is not.

## Why a Privileged Target Position Is Not Obviously Helpful

Giving the critic the hidden target position from the beginning sounds powerful, but it is not necessarily the right training signal.

Before discovery, the actor cannot condition on that target position. If target locations are randomized, the critic may explain returns using information the policy cannot act on. At best, that may reduce value error slightly. At worst, it creates a critic that is clever in ways the actor cannot use.

The useful learning signal before discovery is not "the target is exactly here." It is "explore efficiently under uncertainty." The current observation-based ACC is better aligned with that requirement.

Once the target is discovered or communicated, the joint observation stack already exposes the phase change through target-known and connectivity information. The critic can attend to the informed agents. A privileged phase flag would be a cleaner scalar, not a qualitatively new capability.

## What a Hand-Crafted Critic Could Add

The genuinely missing information in the current ACC is limited:

- exact global coverage history rather than local coverage samples;
- exact continuous pairwise geometry rather than radar-binned perception;
- full map or obstacle structure outside current sensing range;
- perhaps remaining-horizon information if not otherwise represented.

These could improve sample efficiency. But they are not strong reasons to replace the ACC with a conventional perfect-state critic. They are better treated as targeted ablations or auxiliary signals.

The maintained recurrent actor and critic already weaken the coverage-history
argument: agents and the critic can build a belief over explored regions from
experience rather than receiving a global oracle.

## Why Not Feed Reward Terms Directly

The critic should not receive reward decomposition values as observations.

Feeding values such as `r_chain_gap`, `r_success`, or `r_collision` directly into the critic would blur the line between state representation and target construction. It may reduce value loss, but it does so by giving the critic precomputed answers rather than forcing it to learn from state.

A cleaner critic may use state facts from which rewards are computed, such as positions, connectivity, or coverage summaries. But if those facts are already present or inferable from the joint observation stack, the gain is mainly optimization convenience, not a principled architectural improvement.

## Alternatives Considered

### Privileged flat world-state critic

Rejected as the default. It is not agent-number-invariant by construction, scales poorly, risks accidental agent identity, and weakens the observation-grounded nature of the experiment.

### Global mean critic

Useful as an ablation, but too coarse for per-agent credit assignment. SwarmEcho rewards include local exploration, collision, finder, and chain-contribution effects. A single pooled value loses important agent-specific structure.

### Hand-crafted topology critic

Potentially useful for diagnostics, but not ideal as the main architecture. Exact symbolic topology can make the critic depend on engineered abstractions rather than learning chain quality from relative positioning and observable connectivity. If used, it should remain permutation-equivariant and should be evaluated as an ablation.

### Recurrent actor or critic

The maintained defaults use recurrent actor and critic memory. Actor memory lets
each decentralized agent condition actions on episode history:

```text
obs_i -> actor encoder -> GRU_i -> policy head
```

Critic memory keeps the ACC philosophy intact by adding per-agent memory before attention:

```text
obs_i -> critic encoder -> GRU_i -> cross-agent attention -> V_i
```

This improves partial observability and coverage-history estimation while preserving the observation-grounded, permutation-equivariant nature of the architecture. It complements the ACC rather than replacing it.

### Graph neural critic

A strong alternative. It shares the same philosophy as the ACC: agents are interchangeable nodes, interactions are relational, and the architecture scales with swarm size. It may be worth testing later, especially with communication edges. It is not an argument against the current ACC; it is the closest alternative in the same design family.

## Final Defense

The current MAPPO + ACC architecture is the best default for SwarmEcho because it matches the structure of the problem:

- swarms are sets of interchangeable agents;
- relay behavior is relational and topology-sensitive;
- execution must remain decentralized;
- training should not depend on hidden simulator oracle state;
- the critic needs per-agent credit assignment, not only a team scalar;
- attention provides scalable, permutation-equivariant focus over relevant teammates.

A privileged hand-crafted world state may improve some training curves, but it would not be a cleaner or more faithful solution. In this project, the main challenge is learning robust chain-forming behavior from distributed observations. The ACC directly encodes that commitment.

Therefore, the current architecture is not an accidental deviation from standard MAPPO practice. It is the appropriate architectural choice for SwarmEcho's scientific goal.
