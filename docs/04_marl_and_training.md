# Multi-Agent RL & Training

SwarmEcho uses a Multi-Agent Proximal Policy Optimization (MAPPO) framework. The architecture resides in `src/models/` and the training orchestrator in `src/training/`.

## Neural Architecture
*(Found in `src/models/mappo.py`, `src/models/actor.py`, and `src/models/critic.py`)*

The model framework is built using Flax (`nnx` API).

### 1. Decentralized Actor
- Located in `src/models/actor.py`.
- **Local Policy**: The actor is represented by a shared MLP (typically 2-3 layers with LayerNorm and Tanh activations).
- **Execution**: Each drone passes its own local observation (e.g., 57-dimensional vector) through the shared actor to sample a continuous force vector.
- **Continuous Action Sampling**: Uses a diagonal Gaussian distribution to sample action vectors. These samples are squashed using a Tanh activation to lie cleanly within $(-1, 1)$ boundaries. The buffer stores the pre-squash sample $u_i$ directly, and standard Gaussian log-probabilities are computed over $u_i$. This avoids hard clipping gradient anomalies, and does not require explicit Tanh Jacobian correction because the correction terms cancel out in the PPO ratio.

### 2. Centralized Critic Variants
- Located in `src/models/critic.py`.
- During training, the critic views the concatenated observations of all $N$ agents to estimate state values.
- We support two centralized critic types:
  - **Agent-Centric Critic (`AgentCentricCritic`) [Default & Recommended]**: Estimates a unique value $V_i$ for each agent. It uses an MLP encoder followed by a masked multi-head cross-agent attention block (where agent $i$ attends to all other agents but not itself) and an MLP head to output a tensor of shape `(..., N)`.
  - **Global Mean Critic (`GlobalMeanCritic`) [Kept for Ablations]**: Encodes all agent observations, applies standard multi-head self-attention, and aggregates them via a Global Mean Pooling layer to produce a single team-wide scalar value $V$ of shape `(...,)`.

## Cooperative Team Reward
*(Found in `src/env/rewards.py`)*

A single scalar reward $r_t$ is computed and shared across all agents to encourage cooperative task-solving behaviors:

$$r_{total} = R_{coverage} + R_{target\_found} + R_{chain\_gap} + R_{proximity} + R_{collision} + R_{success} + R_{hub\_proximity}$$

- **$R_{coverage}$ (Local, Dense)**: Proportional to the number of *newly discovered* grid cells visited by the swarm in the current step. Once an individual agent persistently knows the target's location, its coverage reward is gated off.
- **$R_{target\_found}$ (Shared + Local, Sparse)**: Shared team bonus when the target is first discovered, plus a localized "finder" bonus awarded only to the specific agent(s) that deliver the target info to the base station.
- **$R_{chain\_gap}$ (Local, Dense)**: A topological penalty proportional to the gap distance between the base-connected sub-network and target-connected sub-network tips. Only "contributing" drones (those on the shortest topological path or in the connected components) receive this dynamic penalty, while non-contributing drones receive a maximum penalty.
- **$R_{proximity}$ (Local, Dense)**: Penalty for agents that are too close to each other, to discourage clustering and promote collision avoidance.
- **$R_{collision}$ (Local, Dense)**: Penalty for each drone currently colliding with boundaries or obstacles.
- **$R_{success}$ (Shared, Sparse)**: A large terminal bonus divided among the team, granted when a continuous multi-hop communication link is successfully established and held for the required time.
- **$R_{hub\_proximity}$ (Shared, Dense)**: Encourages staying near points of interest (base or target). If target is known, drones are rewarded for proximity to either base or target; otherwise, only proximity to base is rewarded.

