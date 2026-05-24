# Environment & Physics

The simulated geometric world of SwarmEcho relies on native JAX tensor operations optimized for rapid batching. Code relating to physical constraints resides in `src/env/`.

## Action Space
*(Found in `src/env/physics.py`)*

Agents (drones) operate in a continuous 2D plane:
$$a_t \in [-\text{max\_force}, \text{max\_force}]^2$$
- Valid outputs are X and Y continuous force vectors.
- Neural networks output a mean ($\mu$) and standard deviation ($\sigma$) to sample forces. 
- Actions are strictly clipped before Euler integration (`Velocity = Velocity * Drag + Force * Timestep`).
- The environment halts forward momentum completely (inelastic "bounce") if a drone cell hits an occupancy matrix wall.

## Permutation-Invariant Observation Space (57 Dimensions)
*(Found in `src/env/observations.py`)*

Each drone receives an ego-centric overview combining geometric awareness with spatial radar mapping, preventing ordering bias.

### A. Self-Intelligence Core (9 Dims)
1. **Velocity (2D)**.
2. **Relative Vector to Base (2D)**.
3. **Connection Flags (2D)**: Am I touching the Base chain? Am I touching the Target chain?
4. **Target Known Flag (1D)**: A persistent boolean stating if the target's location is unlocked via direct sight or multi-hop topological link (Gossip Protocol).
5. **Relative Vector to Target (2D)**: Masked out (`[0,0]`) until the target is known.

### B. Coverage Probes (16 Dims)
Queries 16 radial points in a circle at `exploration_sampling_radius` (default 6.0 meters). If the coordinate in the occupancy grid has been "mapped" by the swarm historically, it returns `1.0` (else `0.0`). Acts as a navigational push toward undiscovered grid cells.

### C. Unified Radar (32 Dims)
*(Powered by `src/env/raycast.py`)*
The continuous 360° vision is binned into 8 angular slices. For each slice, the closest signals degrade linearly via $\max(0, 1 - d / \text{radius})$ for 4 active channels (each normalized by its respective range limit):
1. **Physical Walls**: Inverse distance to closest wall obstacle (normalized by `visual_radius`).
2. **Nearest Teammate**: Inverse distance to any active teammate drone (normalized by `comm_radius`).
3. **Teammate tethered implicitly to Target**: Inverse distance to target-chain connected drones (normalized by `comm_radius`).
4. **Teammate tethered implicitly to Base**: Inverse distance to base-chain connected drones (normalized by `comm_radius`).

*(Note: Pure Base proximity and Pure Target proximity channels are defined in the config but currently inactive/commented out in the implementation to reduce dimension complexity.)*
