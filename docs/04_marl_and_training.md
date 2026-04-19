# Multi-Agent RL & Training

SwarmEcho uses a Multi-Agent Proximal Policy Optimization (MAPPO) arrangement. The relevant logic is distributed across `src/models/` for architecture and `src/training/` for execution loops.

## Neural Architecture
*(Found in `src/models/mappo.py` & `src/models/actor_critic.py`)*

The framework uses Flax (`nnx`). 
- **Actor (Local Phase):** A standard MLP (3 layers, LayerNorm + Tanh). It takes the 65-dimensional local observation and executes the continuous action limits.
- **Centralized Critic & Self-Attention:** The Critic is tasked with judging the state. To do so without breaking permutation invariance, it passes all aggregated individual agent features into a **Multi-Head Self-Attention** block, effectively allowing entities to cross-reference their isolated topological roles. After attention mapping, a Global Mean Pooling layer synthesizes the entire N-agent board state down to a singular representation used to infer the value $V$.

## Cooperative Team Reward 
*(Found in `src/env/rewards.py`)*

A single scalar reward $r_t$ is distributed universally to all executing members to ensure altruistic topological behaviors.

$$r_{total} = R_{coverage} + R_{chain\_gap} + R_{collision} + R_{target\_found} + R_{success}$$

- **$R_{coverage}$ (Dense):** Proportional to the area of *newly discovered* geographical grid cells this time step.
- **$R_{chain\_gap}$ (Dense):** The primary topological penalty. The system dynamically measures the distance spanning between the sub-topology connected to the Base and the one connected to the Target. Drones actively pull away to minimize this negative coefficient.
- **$R_{collision}$ (Dense):** Penalty per drone currently impacting the physical map walls.
- **$R_{target\_found}$ (Sparse):** Positive one-off spike when the globe-map visualizes the target goal point for the first time.
- **$R_{success}$ (Sparse):** Very large terminal bonus granted strictly when a pure hop-by-hop relay line stabilizes between Base and Target.
