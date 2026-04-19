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

## Permutation-Invariant Observation Space (65 Dimensions)
*(Found in `src/env/observations.py`)*

Each drone receives an ego-centric overview combining geometric awareness with spatial radar mapping, preventing ordering bias.

### A. Self-Intelligence Core (9 Dims)
1. **Velocity (2D)**.
2. **Relative Vector to Base (2D)**.
3. **Target Known Flag (1D)**: A persistent boolean stating if the target's location is unlocked via direct sight or multi-hop topological link (Gossip Protocol).
4. **Relative Vector to Target (2D)**: Masked out (`[0,0]`) untill the target is known.
5. **Connection Flags**: Am I touching the Base chain? Am I touching the Target chain?

### B. Coverage Probes (8 Dims)
Queries 8 radial points (North, East, etc.) set roughly 20 meters away. If the coordinate in the occupancy grid has been "mapped" by the swarm historically, it returns `1.0`. Acts as a long-range navigational push toward undiscovered grid cells.

### C. Unified Radar (48 Dims)
*(Powered by `src/env/raycast.py`)*
The continuous 360-vision is binned into 8 pie slices. For each slice, the closest signals degrade linearly via $1 - (d / \text{radius})$ for 6 distinct channels:
1. Physical Walls.
2. Nearest Teammate.
3. Teammate tethered implicitly to Target.
4. Teammate tethered implicitly to Base.
5. Pure Base proximity.
6. Pure Target proximity.
