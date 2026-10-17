# SwarmEcho documentation

The [repository README](../README.md) is the entry point for the current runtime.

## Current reference

- [System overview](01_reference/01_system_overview.md)
- [Environment and physics](01_reference/02_environment_and_physics.md)
- [MARL and training](01_reference/03_marl_and_training.md)
- [Agent-centric critic design rationale](01_reference/04_mappo_acc_architecture_defense.md)
- [Maintained assumptions](01_reference/05_assumptions.md)
- [PPO controls](01_reference/05_ppo_controls.md)
- [Adaptive swarm size](01_reference/06_adaptive_swarm_size.md)
- [Adaptive swarm diagnostics and interpretation](01_reference/07_adaptive_swarm_diagnostics.md)
- [Adaptive target spawning: rules, formula and dashboard](01_reference/08_adaptive_target_spawning.md)
- [Progress-based soft cluster credit](01_reference/09_soft_cluster_credit.md)
- [Adaptive map sampling](01_reference/10_adaptive_map_sampling.md)

## Operational guides

- [Configuration](02_guide/01_configuration_guide.md)
- [Analysis and tools](02_guide/02_analysis_and_tools.md)
- [Commands](02_guide/03_commands.md)
- [Public inspector exports and GitHub Pages](02_guide/04_public_inspector.md)
- [C relay curriculum and geometry previews](02_guide/05_C_curriculum.md)
- [G curriculum: five static maps and independent adaptive spawning](02_guide/06_G_curriculum.md)

The general run-analysis dashboard remains available for runs.
The separate inspector handles replay and evaluation visualization.

## Status, proposals, and history

- [Geo Phase 0 completed: frozen probes and observation-ambiguity results](geo/Phase0.md)
- [Geo Phase 1: visual baseline, own-cell diagnostics and proposed auxiliary loss](geo/Phase1.md)
- [Geo Phase 2: unscoped long-term spatial research ideas](geo/Phase2.md)
- [2D removal audit](03_roadmap/05_2d_removal_audit.md)
- [Future features](03_roadmap/02_future_features.md)
- [Road to 4k maps: scaling concerns and proposed work](road-to-4k.md)
- [Office performance audit](03_roadmap/office_performance_audit.md)
- [Historical cleanup plan](03_roadmap/01_cleanup_plan.md)
- [Historical transition discovery](03_roadmap/03_environment_transition_discovery.md)
- [Building-editor design proposal](03_roadmap/04_map_builder_implementation_plan.md)
- [Archived editor plan](04_archive/03_building_editor_plan.md)
- [Idea archive](04_archive/01_idea_archive.md)
- [Experimental setup archive](04_archive/02_rl_experimental_setup.md)

Historical documents preserve context and are not operational instructions.
Current code, selected level YAML, and current guides take precedence.
