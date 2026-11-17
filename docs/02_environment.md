# 02 Environment

## Buildings, geometry and target placement

The simulator uses building YAMLs for the geometry of walls, doors, openings and floors. These solids determine where drones can fly, which pairs have line of sight and where the free-space roadmap can route around obstacles. Targets are sampled in continuous space within eligible spawn volumes.

CLU G03 is one example of the supported geometry. Its 15 × 15 m footprint and three storeys give a 3 × 3 × 3 grid of 5 m cells. For generated buildings, `random_buildings.length_m` and `width_m` set the footprint in metres. Both must be whole multiples of `cell_size_m` and at least two cells across. `stories` determines the vertical cell count, using the same cell edge length. Room bounds control how many rooms the generator partitions each floor into, while door and window probabilities control extra openings. Wall and tile thickness must be smaller than a cell. Larger or differently shaped grids use the same generator and simulator. For example, a 20 × 15 m footprint with four storeys and 5 m cells defines a 4 × 3 × 4 grid. The layout still needs enough clear space for stairs, doors, the base and legal target positions. An authored building declares its own dimensions and geometry. Maps prepared together in a training bank must share the cell size and wall and tile thickness.

`env.coverage_voxel_size` is a separate grid for exploration credit. It must divide every world dimension. A smaller voxel changes which areas count as newly covered and can increase the exploration reward available for the same physical route. `env.drone_radius` provides collision clearance. The planning graph adds `env.obstacle_planning_clearance_m` to that radius, so changing either can change which passages the planner considers traversable. `env.roadmap_node_density` changes the spacing of sampled waypoints used to measure collision-free route lengths.

`env.target_wall_buffer_fraction` shrinks target spawn boxes near walls by a clearance measured as a fraction of the building cell edge. For generated buildings, `random_buildings.advanced_exclustion_base_LOS` excludes the base cell, legal cells with an in-range line of sight from the base, and up to one nearby cell around a corner. The generator uses a representative point per cell and the sum of the base radio and target sight ranges. The default generated and authored tasks use the base position stored in the map. `training.randomize_base` can sample a new base where the setup permits it. Generated exclusions are attached to the original map cells, so check which nearby targets are legal if you move the base. Validation, adaptive target spawning and precomputed target eligibility require fixed bases.

[Geometry and target generation](../src/swarmecho/env/random_buildings.py) · [Building and spawn boxes](../src/swarmecho/env/buildings.py) · [Roadmap](../src/swarmecho/env/roadmap_cpu.py) · [Building bank](../src/swarmecho/env/building_bank.py)

## Physics and deployment

An action supplies three force components after tanh squashing. `max_force` scales those components, `drag` scales the previous velocity before the force update, and `max_speed` caps the resulting speed. The environment checks the proposed movement against bounds and solids and clears velocity on collision. `dt` is the simulation time per step and `max_steps` caps an episode. G03's 700 steps at 0.1 s allow up to 70 s of simulated time.

`env.num_agents` sets the number of drone slots. With G03's automatic deployment, one drone starts at reset, then `spawn_delay` sets the minimum interval between subsequent launches. `adapt_size.decommission_radius_m` sets a sphere above the base, and `decommission_hold_steps` sets how many consecutive steps an active drone must stay within the valid zone to leave the episode. Leaving the zone resets the dwell count. Setting `adapt_size.enabled=false` retains staggered deployment and disables decommissioning. A task that requires surplus cleanup must keep decommissioning enabled. Launches reset recurrent memory. Inactive slots have zero observations and actions and are masked during PPO. An episode ends if no active or scheduled drones remain.

[Preset](../src/swarmecho/curriculum_config/levels/G/G03_CLU_random_buildings.yaml) · [Physics and episodes](../src/swarmecho/env/environment.py) · [Deployment](../src/swarmecho/env/adapt_size.py)

## Actions, observations and communication

The actor samples three Gaussian movement commands. The environment receives their tanh-squashed values, which it scales to force. With the G03 preset, each drone has **140 observation values**.

| Input | Values | Meaning |
| --- | ---: | --- |
| Velocity and connectivity | 6 | Velocity divided by maximum speed, plus base connectivity, target connectivity and target knowledge |
| Task and population signals | 3 | Decommission-zone membership, active fleet size divided by 10 for base-connected drones, and selected chain membership gated by target knowledge |
| Target displacement | 3 | Relative target position divided by visual radius while the drone directly sees the target, zero otherwise |
| Wall and drone radar | 128 | One wall proximity signal and three drone proximity signals per direction, interleaved across 32 directions |

Three distance settings serve different links. `comm_radius` determines peer radio links and the drone radar's range. `comm_radius_base` determines links to the base. `visual_radius` determines direct target sight and normalizes visible target displacement. Walls can block all three. Wall sensing uses its own ray count and range. Increasing `wall_sensing_ray_max_distance` extends wall sensing without changing target visibility.

For drone radar, `legacy` reports the strongest proximity in each direction separately for visible peers, target-connected peers and base-connected peers. `band_sep_counting` divides that range into `distance_bands` and returns a count for each direction, band and peer group, capped and normalized by `radar_count_cap`. `drone_radar_bins` controls angular resolution. With matching wall and drone bin counts, the legacy layout interleaves the wall and drone readings per direction. Changing the mode or bin counts changes the observation layout.

The optional observation toggles control which inputs are present. `observe_base_population` lets drones connected to the base see the fleet size, normalized by a fixed reference of 10. Disconnected drones see zero for that input. With adaptive deployment enabled, turning it off removes one input per drone. The population and decommission-zone inputs are available with adaptive deployment. `observe_decommission_zone` supplies a local flag for the valid decommission sphere, while `observe_chain_contributor` indicates membership in the selected chain when the drone knows the target. `observe_visually_gated_target_displacement` supplies the direction and distance to a directly seen target as a normalized three-vector. Changing any of these changes the input expected by the model. Use a fresh model for such an ablation and keep the same observation layout when running an existing checkpoint.

Target knowledge spreads one hop per step using the pre-action radio graph. Directly connected informed drones can report to the base. The base retains the report and can inform newly connected drones. Learned TarMAC messages also use the line-of-sight radio graph. Each drone attends to peers' previous signatures and values. An uninformed drone in contact with the base can receive the base's cached message.

[Observation and radio code](../src/swarmecho/env/environment.py) · [TarMAC actor](../src/swarmecho/models/actor.py) · [Base memory](../src/swarmecho/training/train.py)

## Success and reward

An active drone connected through the communication graph to both the base and the target indicates that a chain exists. `hold_chain_for` sets the number of consecutive connected steps required. A gap resets the hold counter. With `success_condition=chain_held`, reaching the hold threshold succeeds. With `chain_held_and_cleand_up`, the chain must also be free of active surplus drones. That is the G03 training criterion. The G03 validation and inspection criteria use `chain_held`.

The chain hold counter keeps advancing while surplus drones return to the base. The selected contributor chain identifies which drones are surplus. Relay gap shaping can share credit over eligible tied shortest-hop branches even when the selected chain chooses one path. The reward settings are signed coefficients applied at particular events or on particular steps. A zero coefficient disables that reward contribution.

| Reward | G03 behavior |
| --- | --- |
| Exploration and costs | +2 per newly covered voxel, split among simultaneous viewers. Exploration credit stops for a drone that knows the target. Collision costs 0.5 and each acting step costs 0.75. |
| First delivery and first success | On delivery, each qualifying direct reporter earns +100 and every active drone earns +100. First success pays +3,000 per active drone. |
| Relay shaping | Once base knows the target, +0.2 per credited metre of closed geodesic gap each step. Soft credit shares across tied shortest-hop relay branches. The shortest complete relay path pays its members up to +25 each, scaled by the ratio of the direct free-space geodesic to path length. |
| Surplus cleanup | While the chain survives, eligible surplus drones earn +3 per metre toward the decommission zone and incur −3 per metre away, measured geodesically. A qualifying surplus drone earns +3,000 on removal. G03’s generic decommission reward is zero. |
| Terminal penalties | The last acting drones incur −1,000 if the fleet is exhausted. On terminal chain success, each active drone outside the selected chain incurs −3,000, and each active drone incurs an additional −750 per surplus drone, capped at −2,000 per recipient. |

[Reward implementation](../src/swarmecho/env/environment.py) · [G03 coefficients](../src/swarmecho/curriculum_config/levels/G/G03_CLU_random_buildings.yaml)
