# Environment & Physics

The simulated geometric world of SwarmEcho relies on native JAX tensor operations optimized for rapid batching. Environment code resides in `src/swarmecho/env/`.

## Action Space
*(Found in `src/swarmecho/env/physics.py`)*

Agents (drones) operate in a continuous 2D plane:
$$a_t \in [-\text{max\_force}, \text{max\_force}]^2$$
- Valid outputs are X and Y continuous force vectors.
- Neural networks output a mean ($\mu$) and standard deviation ($\sigma$) to sample forces. 
- Actions are strictly clipped before Euler integration (`Velocity = Velocity * Drag + Force * Timestep`).
- The environment reflects velocity with the configured wall restitution if a drone path hits an occupancy-matrix wall.

## Permutation-Invariant Observation Space
*(Found in `src/swarmecho/env/observations.py`)*

Each drone receives an ego-centric overview combining geometric awareness with
spatial radar mapping, preventing ordering bias. The maintained M-series levels
disable the three transitional odometry/coverage aids, so their default actor
observation is **37 dimensions**. Enabling all three aids produces the older
57-dimensional layout; that layout is retained only as a transitional option
for early 3D work and is not the maintained 2D default.

### A. Self/knowledge block (9 dimensions maximum; 5 in the maintained default)
1. **Velocity (2D)**.
2. **Relative Vector to Base (2D, transitional and disabled by maintained levels)**.
3. **Connection Flags (2D)**: Am I touching the Base chain? Am I touching the Target chain?
4. **Target Known Flag (1D)**: A persistent boolean stating if the target's location is unlocked via direct sight or multi-hop topological link (Gossip Protocol).
5. **Relative Vector to Target (2D, transitional and disabled by maintained levels)**: Masked out (`[0,0]`) until the target is known.

### B. Coverage Probes (16 Dims, transitional)
Queries 16 radial points in a circle at `visual_radius + 1.0` (meter offset dynamically calculated from the visual range). If the coordinate in the occupancy grid has been "mapped" by the swarm historically, it returns `1.0` (else `0.0`). Acts as a navigational push toward undiscovered grid cells.

### C. Unified Radar (32 Dims)
*(Powered by `src/swarmecho/env/raycast.py`)*
The continuous 360° vision is binned into 8 angular slices. For each slice, the closest signals degrade linearly via $\max(0, 1 - d / \text{radius})$ for 4 active channels (each normalized by its respective range limit):
1. **Physical Walls**: Inverse distance to closest wall obstacle (normalized by `visual_radius`).
2. **Nearest Teammate**: Inverse distance to any active teammate drone (normalized by `comm_radius`).
3. **Teammate tethered implicitly to Target**: Inverse distance to target-chain connected drones (normalized by `comm_radius`).
4. **Teammate tethered implicitly to Base**: Inverse distance to base-chain connected drones (normalized by `comm_radius`).

The current default layout is therefore:

```text
5 self/knowledge dimensions + 32 radar dimensions = 37 dimensions
```

The coverage grid is still used internally for exploration reward and
evaluation heatmaps even when the local coverage-probe observation block is
disabled. Base-vector, target-vector, and coverage-probe observations are
documented as transitional aids in `assumptions.md`, not as required final
features.

## Environment state ownership

The runtime state is nested by responsibility rather than kept as one flat
record. `PhysicsState` owns positions, velocities, activation, dimensions, and
the episode clock. `CommunicationState` owns direct adjacency, visibility,
connectivity summaries, and persistent target knowledge. `ExplorationState`
owns the 2D coverage grid. `RelayTaskState` owns chain progress and the
discrete 2D finder-path arrays. Collision and coverage deltas are retained as
per-step diagnostic signals because rewards consume them. See
`docs/assumptions.md` for the architectural contract.
