# Multi-Agent Reinforcement Learning (MARL) Experimental Setup & System Architecture

This specification sheet documents the system design, neural architecture, and optimization framework for **SwarmEcho**, focusing on the Multi-Agent Proximal Policy Optimization (MAPPO) setup. This document acts as an independent specification to evaluate the eligibility of reinforcement learning techniques for this multi-robot coordination problem.

---

## 1. Usecase & Mission Scenario

The primary objective of SwarmEcho is to establish a dynamic, ad-hoc, delay-tolerant multi-hop communication relay chain between a stationary **Base Station** and a mobile or remote **Target** using a homogeneous swarm of $N$ robotic agents (drones). 

### Physical & Communication Constraints
* **Environment Geometry**: An open $112.0\,\text{m} \times 112.0\,\text{m}$ two-dimensional bounded workspace with outer boundaries representing hard walls.
* **Base Station Location**: Fixed at the map center coordinates $[56.0, 56.0]$.
* **Target Spawning**: Dynamically spawned at the start of each episode outside a $35.0\,\text{m}$ radius from the base station to ensure a multi-hop distance requirement.
* **Drone Spawning**: All $N = 9$ drones spawn stacked at the base station position. To break spatial symmetry and facilitate stable rollout behavior, drones are activated with a **staggered spawn delay** of $5$ environment steps between consecutive drone activations.
* **Communication & Perception Ranges**:
  * **Drone-to-Drone Communication Range ($d_{\text{comm}}$)**: $10.0\,\text{m}$.
  * **Base-to-Drone First-Hop Range ($d_{\text{comm, base}}$)**: $30.0\,\text{m}$.
  * **Visual/Sensing Radius ($r_{\text{vis}}$)**: $5.0\,\text{m}$ (used for target line-of-sight detection and wall sensing).
  * **Exploration Sampling Radius**: $6.0\,\text{m}$.

### Execution Paradigm
The system operates under the **Centralized Training, Decentralized Execution (CTDE)** paradigm. During deployment, each drone has access *only* to its local egocentric observations and executes its policy independently. During training, a centralized critic utilizes the joint observation space of all agents to guide optimization.

---

## 2. Decentralized Actor Network

The Actor policy is shared across all agents (parameter sharing) and executed decentralized via vectorization (`vmap`).

### 2.1 Ego-centric Observation Space
Each agent $i$ receives a local observation vector of dimension $D = 9 + 16 + 4B = 57$ (for $B = 8$ angular radar bins), structured as follows:

```
┌────────────────────────────────── Observation Vector (57-dim) ──────────────────────────────────┐
│                                                                                                 │
│  Ego-State (9-dim)                                                                              │
│  ├─ Normalized Velocity: [v_x, v_y] / v_max                                               (2)   │
│  ├─ Base Station Odometry: (pos_base - pos_i) / max_dim                                   (2)   │
│  ├─ Connectivity Flags: [is_connected_to_base, is_connected_to_target]                    (2)   │
│  ├─ Target Known Flag: 1.0 if target discovered, 0.0 otherwise                            (1)   │
│  └─ Target Odometry: (pos_target - pos_i) / max_dim (gated to [0, 0] if target unknown)   (2)   │
│                                                                                                 │
│  Local Coverage Probes (16-dim)                                                                 │
│  └─ 16 radial boolean probes (0.0/1.0) spaced at 22.5° intervals at 6.0m sampling radius        (16)  │
│                                                                                                 │
│  Ego-Radar Bins (32-dim, B=8)                                                                   │
│  └─ 8 angular slices (45° width) containing 4 spatial features:                                 │
│       1. Inverse distance to closest wall obstacle: max(0, 1 - d_wall / r_vis)           (8)   │
│       2. Inverse distance to any teammate drone: max(0, 1 - d_drone / d_comm)             (8)   │
│       3. Inverse distance to target-chain connected drones: max(0, 1 - d_tgt / d_comm)   (8)   │
│       4. Inverse distance to base-chain connected drones: max(0, 1 - d_base / d_comm)     (8)   │
│                                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

The graph connectivity flags (`is_connected_to_base` and `is_connected_to_target`) are computed using JAX-parallelized matrix squaring of the adjacency matrix. The target position is masked until the target has been physically discovered by at least one connected drone; once discovered, this knowledge propagates along the communication chain and is persistently memorized by individual agents.

### 2.2 Neural Architecture
The decentralized actor model is implemented in [actor.py](file:///q:/_0_Projects/000_SwarmEcho/SwarmEcho/src/models/actor.py) as `DecentralizedActor`.
* **Trunk**: A Multi-Layer Perceptron (MLP) consisting of hidden layers of width $256$ with **Layer Normalization** and **Tanh activations** applied to each layer.
* **Heads**: Two separate linear heads outputting:
  1. Mean vector: $\mu_i \in \mathbb{R}^2$
  2. Log standard deviation: $\log \sigma_i \in \mathbb{R}^2$, clipped to the range $[-5.0, 2.0]$ for numerical stability.

### 2.3 Action Space & Tanh Squashing
The physical action space consists of a 2D continuous force vector $f_i \in [-F_{\text{max}}, F_{\text{max}}]^2$, where $F_{\text{max}} = 50.0\,\text{N}$. 

To avoid training instabilities and gradient clipping artifacts at the boundaries, actions are sampled using a **Tanh-squashed Diagonal Gaussian policy**:
1. A pre-squash latent action is sampled:
   $$u_i \sim \mathcal{N}(\mu_i, \sigma_i^2)$$
2. The normalized policy action $a_i \in (-1, 1)^2$ is computed as:
   $$a_i = \tanh(u_i)$$
3. The physical force vector is scaled by the environment runner:
   $$f_i = a_i \cdot F_{\text{max}}$$

To ensure consistent policy evaluation during the training update step, the experience buffer stores the **pre-squash sample $u_i$** instead of the squashed action $a_i$. The policy calculations and PPO updates are conducted directly in the pre-squash space (using variables $u_i$ and standard Gaussian log-probabilities $\log \mathcal{N}(u_i | \mu_i, \sigma_i)$). This avoids the need for explicit Tanh Jacobian corrections during PPO optimization, as the Jacobian correction terms cancel out in the PPO ratio $\frac{\pi_\theta(a_i | o_i)}{\pi_{\theta_{\text{old}}}(a_i | o_i)} = \frac{\pi_\theta(u_i | o_i)}{\pi_{\theta_{\text{old}}}(u_i | o_i)}$, and keeps optimization stable.

---

## 3. Agent-Centric Centralized Critic (AAC)

To resolve multi-agent credit assignment and eliminate self-loop bias, SwarmEcho uses an **Agent-Centric Centralized Critic** (`AgentCentricCritic` in [critic.py](file:///q:/_0_Projects/000_SwarmEcho/SwarmEcho/src/models/critic.py)).

### 3.1 Network Architecture
The critic processes the joint observation stack of all active agents $(o_1, \dots, o_N) \in \mathbb{R}^{N \times D}$ and outputs an individual value estimate $V_i \in \mathbb{R}$ for each agent $i$:

```
        Joint Observations: [o_1, o_2, ..., o_N]  (N, D)
                             │    │          │
                     ┌───────┴────┼──────────┴──────┐
                     │ Shared MLP Encoder (H=256)   │
                     └────────────┬─────────────────┘
                                  ▼
                        Latents: [e_1, e_2, ..., e_N]  (N, H)
                             │    │          │
                     ┌───────┴────┼──────────┴──────┐
                     │ Masked Cross-Agent Attention │   <── Diagonal masked (i != j)
                     └────────────┬─────────────────┘
                                  ▼
                    Contexts: [x_1, x_2, ..., x_N]  (N, H)
                             │    │          │
                     ┌───────┴────┼──────────┴──────┐
                     │ Concatenate: [e_i || x_i]    │   (N, 2H)
                     └────────────┬─────────────────┘
                                  ▼
                     ┌────────────┴─────────────────┐
                     │ Shared Value MLP Head        │
                     └────────────┬─────────────────┘
                                  ▼
                Value Estimates: [V_1, V_2, ..., V_N]  (N,)
```

1. **Shared Encoder**: A single-layer MLP maps each individual observation $o_i$ to a latent encoding $e_i \in \mathbb{R}^H$ (where $H = 256$).
2. **Masked Multi-Head Cross-Attention**: The queries, keys, and values are generated from the latents. To construct the context for agent $i$, a cross-attention block query $q_i = e_i$ attends to keys and values of all teammates $e_j$ ($j \neq i$). An off-diagonal mask is applied:
   $$\text{Mask}_{i, j} = \begin{cases} 
   \text{True} & \text{if } i \neq j \\
   \text{False} & \text{if } i = j
   \end{cases}$$
   This results in an attention context vector $x_i \in \mathbb{R}^H$ representing the state of the swarm from the perspective of agent $i$ without self-influence.
3. **Concat & Value Output**: The local encoding $e_i$ and the cross-agent context $x_i$ are concatenated to form $[e_i \mathbin{\Vert} x_i] \in \mathbb{R}^{2H}$. A value head (a 3-layer LayerNorm + Tanh MLP) processes this vector to estimate the agent-centric state value:
   $$V_i(o_1, \dots, o_N) = \text{CriticHead}([e_i \mathbin{\Vert} x_i])$$

### 3.2 Rationale
* **Permutation Invariance**: The joint critic is permutation-invariant with respect to teammate ordering because the encoder and head weights are tied across all agents, and attention operations are symmetric.
* **Credit Assignment**: By withholding agent $i$'s own features from its attention context, the value baseline focuses purely on evaluating how the cooperative context surrounds agent $i$. This prevents self-loop bias and stabilizes variance in multi-agent advantage calculations.

---

## 4. Cooperative Shortest-Path Reward Structure

The reward function ([rewards.py](file:///q:/_0_Projects/000_SwarmEcho/SwarmEcho/src/env/rewards.py)) generates a reward vector $R = [r_1, \dots, r_N]$ allocating specific dense and sparse signals.

### 4.1 Reward Components

#### Local Rewards (Not Scaled by Swarm Size $N$)
* **Exploration Bonus ($r_{\text{coverage}, i}$)**: Dense reward for visiting uncovered grid cells.
  $$r_{\text{coverage}, i} = \Delta \text{Cells}_i \times 1.0\,\text{m}^2 \times 0.01$$
  *Gated off individually once agent $i$ persistent-knows the target location, focusing its behavior on the relay task.*
* **Collision Penalty ($r_{\text{collision}, i}$)**: Dense penalty applied if agent $i$ collides with a boundary.
  $$r_{\text{collision}, i} = -0.5 \times \mathbb{I}(\text{collision}_i)$$
* **Finder Bonus ($r_{\text{finder}, i}$)**: Sparse bonus awarded *only* to the specific agent(s) inside the communication link that deliver the discovered target information to the base station.
  $$r_{\text{finder}, i} = 125.0 \times \mathbb{I}(\text{first\_delivery}_i)$$

#### Global Team Rewards (Scaled/Divided by Swarm Size $N$)
* **Target Found Bonus ($r_{\text{found}}$)**: One-off team bonus when the target is first discovered.
  $$r_{\text{found}, i} = \frac{250.0}{N}$$
* **Success Bonus ($r_{\text{success}}$)**: Terminal reward given when a continuous communication chain links base station and target.
  $$r_{\text{success}, i} = \frac{500.0}{N} \times \mathbb{I}(\text{fully\_connected} \land \text{episode\_end})$$
---

### 4.2 Shortest Path Chain Gap Penalty

The core topological driver in SwarmEcho is the **Chain Gap Penalty** ($r_{\text{chain\_gap}}$).

#### Tip Selection & Chain Gap
The system dynamically computes the connected components from the base station and the target:
1. **Base-Connected Sub-network**: Drones connected to the base station via multi-hop links. The drone in this component closest to the target is the **Base-Chain Tip** ($p_b$).
2. **Target-Connected Sub-network**: Drones connected to the target via multi-hop links. The drone in this component closest to the base station is the **Target-Chain Tip** ($p_t$).
3. The Euclidean distance between these tips defines the **Chain Gap**:
   $$g = \lVert p_b - p_t \rVert_2$$
   If the chain is fully connected, $g = 0$.

#### Shortest-path credit assignment
To focus optimization and avoid rewarding redundant or idle drones, the chain gap penalty uses **Min-Plus matrix multiplication** to identify one deterministic shortest route from each connected front to its selected tip.

* Drones are determined to be **contributing** if they lie on the shortest topological path between the base station and target tips.
* **Contributing Drones**: Receive the dynamic, distance-scaled gap penalty:
  $$r_{\text{chain\_gap}, i} = \frac{-g \cdot w_{\text{gap}}}{N}$$
  where $w_{\text{gap}} = \frac{5.0}{\lVert \text{pos}_{\text{target}} - \text{pos}_{\text{base}} \rVert_2}$ is the normalized gap weight.
* **Non-Contributing Drones**: Receive the maximum penalty:
  $$r_{\text{chain\_gap}, i} = \frac{-5.0}{N}$$

This formulation guarantees that agents not actively participating in the communication backbone are penalized, forcing them to either join the chain or minimize interference.

---

## 5. PPO Setup & Training Orchestration

The training loop (coordinated by [runner.py](file:///q:/_0_Projects/000_SwarmEcho/SwarmEcho/src/training/runner.py)) is JAX-native, scaling across parallelized environments.

### 5.1 Rollout & Buffer Storage
During the rollout phase:
1. The policy interacts with $E = 4096$ parallel environments for a horizon of $T = 64$ steps.
2. At step $t$, the actor maps current observations $O^t \in \mathbb{R}^{E \times N \times D}$ to pre-squash actions $U^t \in \mathbb{R}^{E \times N \times A}$ and log-probabilities $\log \pi(A^t | O^t) \in \mathbb{R}^{E \times N}$.
3. The centralized critic computes the values $V^t \in \mathbb{R}^{E \times N}$ based on joint observations.
4. Experience transitions are collected in a NumPy-backed rollout buffer ([mappo_buffer.py](file:///q:/_0_Projects/000_SwarmEcho/SwarmEcho/src/training/mappo_buffer.py)).

### 5.2 Generalized Advantage Estimation (GAE)
At the end of a rollout, GAE is computed per-agent on the CPU. The TD-error $\delta_{i, e}^t$ and advantage $A_{i, e}^t$ for agent $i$ in environment $e$ at step $t$ are calculated as:
$$\delta_{i, e}^t = r_{i, e}^t + \gamma V_{i, e}^{t+1} (1 - d_e^{t+1}) - V_{i, e}^t$$
$$A_{i, e}^t = \delta_{i, e}^t + \gamma \lambda (1 - d_e^{t+1}) A_{i, e}^{t+1}$$
where $d_e^t$ is the environment termination flag. The value targets (returns) are computed as:
$$Y_{i, e}^t = A_{i, e}^t + V_{i, e}^t$$

**Advantage Normalization**: To stabilize SGD, the advantages are flattened and normalized across the entire batch (dimension $T \times E \times N = 64 \times 4096 \times 9 = 2,359,296$ transitions):
$$\bar{A}_{i, e}^t = \frac{A_{i, e}^t - \mu_A}{\sigma_A + 10^{-8}}$$

---

### 5.3 Optimization Objective & Loss Functions

The training updates are run JIT-compiled for $10$ epochs over $16$ minibatches (minibatch size $= 16,384$ team steps, yielding $147,456$ individual transitions per update step).

#### 1. Policy Surrogate Loss (PPO Clip)
For each agent transition in the minibatch, the importance sampling ratio is:
$$r_i(\theta) = \exp\left(\log \pi_\theta(a_i | o_i) - \log \pi_{\theta_{\text{old}}}(a_i | o_i)\right)$$
The clipped policy objective is:
$$L_{\text{CLIP}}(\theta) = -\mathbb{E}_{\text{MB}}\left[ \frac{1}{N} \sum_{i=1}^N \min\left( r_i(\theta) \bar{A}_i, \text{clip}(r_i(\theta), 1 - \epsilon, 1 + \epsilon) \bar{A}_i \right) \right]$$
where clipping parameter $\epsilon = 0.2$.

#### 2. Value Function Loss (MSE)
The critic is updated to minimize the mean squared error between the estimated values and the GAE value targets:
$$L_{\text{VF}}(\phi) = \mathbb{E}_{\text{MB}}\left[ \frac{1}{N} \sum_{i=1}^N \left( V_i(o_1, \dots, o_N; \phi) - Y_i \right)^2 \right]$$

#### 3. Entropy Regularization
To encourage exploration and prevent premature policy convergence, an entropy bonus is maximized:
$$L_{\text{ENT}}(\theta) = \mathbb{E}_{\text{MB}}\left[ \frac{1}{N} \sum_{i=1}^N \mathcal{H}\left( \pi_\theta(\cdot | o_i) \right) \right]$$
where $\mathcal{H}$ represents the entropy of the Tanh-squashed diagonal Gaussian policy.

#### 4. Joint Loss Function
The total loss minimized by the trainer ([mappo_trainer.py](file:///q:/_0_Projects/000_SwarmEcho/SwarmEcho/src/training/mappo_trainer.py)) is:
$$L_{\text{total}}(\theta, \phi) = L_{\text{CLIP}}(\theta) + c_1 L_{\text{VF}}(\phi) - c_2 L_{\text{ENT}}(\theta)$$
where $c_1 = 0.5$ (value loss coefficient) and $c_2 = 0.01$ (entropy coefficient).

---

## 6. Hyperparameter Reference Table

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Total Timesteps** | $100,000,000$ | Total environment interaction steps for training |
| **Parallel Envs ($E$)** | $4,096$ | Number of vector-parallelized environments |
| **Rollout Horizon ($T$)**| $64$ | Steps per agent collected before optimization |
| **PPO Epochs** | $10$ | Number of passes over the buffer per update cycle |
| **Minibatches** | $16$ | Division of buffer for stochastic gradient steps |
| **Learning Rate ($\alpha$)**| $3 \times 10^{-4}$ | Adam learning rate |
| **Optimizer** | Adam | Standard first-order stochastic optimizer |
| **Grad Clip Norm** | $0.5$ | Maximum global norm limit for gradient updates |
| **Discount Factor ($\gamma$)**| $0.99$ | Temporal discount factor for GAE |
| **GAE Lambda ($\lambda$)** | $0.95$ | GAE interpolation parameter |
| **Clip Epsilon ($\epsilon$)**| $0.2$ | PPO policy ratio clipping threshold |
| **Value Coeff ($c_1$)** | $0.5$ | Weight coefficient of the Critic loss |
| **Entropy Coeff ($c_2$)** | $0.01$ | Weight coefficient of the policy entropy bonus |
| **Actor Layers** | $3$ (depth) | Shared MLP trunk layers in the Actor |
| **Critic Layers** | $3$ (depth) | Post-attention MLP value head layers in the Critic |
| **Hidden Dimension** | $256$ | Hidden layer width (units) for Actor and Critic MLPs |
| **Critic Type** | `agent_centric` | Agent-centric masked attention Centralized Critic |
| **Early-exit Threshold** | $0.999$ | Optional parallel-evaluation success rate that saves a handoff checkpoint and returns from training |
