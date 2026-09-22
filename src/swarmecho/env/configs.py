"""Dependency-free environment and reward settings."""
from dataclasses import dataclass

@dataclass(frozen=True)
class EnvConfig:
    num_agents: int = 5
    dt: float = 0.1
    max_force: float = 15.0
    max_speed: float = 5.0
    drag: float = 0.85
    drone_radius: float = 0.25
    comm_radius_base: float = 6.0
    comm_radius: float = 5.0
    visual_radius: float = 4.0
    target_spawn_buffer: float = 0.5  # Legacy config compatibility; no base-distance restriction.
    target_wall_buffer_fraction: float = 0.1
    radar_bins: int = 8
    spawn_delay: int = 0
    hold_chain_for: int = 50
    max_steps: int = 700
    no_movement_termination_steps: int = 50
    success_when_target_found_or_delivered: bool = False
    movement_epsilon: float = 1e-3
    observe_target_vector: bool = False
    observe_base_vector: bool = False
    observe_coverage_probe: bool = False
    observe_chain_contributor: bool = False
    observe_current_timestep: bool = False  # Episode step / max_steps, in [0, 1].
    coverage_voxel_size: float | None = None
    num_obstacles: int = 0
    obstacle_size_min_m: float = 2.0
    obstacle_size_max_m: float = 4.0
    obstacle_spawn_layer_min: int = 2
    obstacle_spawn_layer_max: int = 5
    obstacle_boundary_buffer_m: float = 0.5
    obstacle_target_buffer_m: float = 0.5
    obstacle_planning_clearance_m: float = 0.1
    roadmap_merge_walls: bool = True
    roadmap_corner_bonus_m: float = 8.0
    obstacle_layout_version: str = "three_aabb_v1"


@dataclass(frozen=True)
class RewardConfig:
    target_found_requires_delivery: bool = True
    chain_reward_system: str = "euclidean"
    exploration_bonus: float = 0.25
    collision_penalty: float = 0.5
    finder_bonus: float = 50.0
    max_gap_penalty: float = 5.0
    target_found_bonus: float = 100.0
    success_bonus: float = 500.0
    no_movement_termination_penalty: float = -1000.0
    # Credit every drone on a simple base-target path, including alternate routes.
    allow_redundancy_reward: bool = False
    enable_chain_efficiency_reward: bool = False
    chain_efficiency_bonus: float = 0.5


