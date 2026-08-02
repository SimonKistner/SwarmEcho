# SwarmEcho documentation

The repository root [README](../README.md) is the entry point. This file is the
documentation map and intentionally sorts before the numbered topic folders.

## Where to find things

### `01_reference/` — what the current system is

- [System overview](01_reference/01_system_overview.md)
- [Environment and physics](01_reference/02_environment_and_physics.md)
- [MARL and training](01_reference/03_marl_and_training.md)
- [MAPPO and agent-centric critic architecture](01_reference/04_mappo_acc_architecture_defense.md)
- [Project assumptions](01_reference/05_assumptions.md)

### `02_guide/` — how to use the repository

- [Configuration and curriculum](02_guide/01_configuration_guide.md) — configuration
  domains, level overrides, and maps.
- [Analysis and tools](02_guide/02_analysis_and_tools.md) — dashboards,
  renderers, and evaluation artifacts.
- [Commands](02_guide/03_commands.md) — supported command-line workflows.

### `03_roadmap/` — planned work

- [Cleanup plan](03_roadmap/01_cleanup_plan.md)
- [Future features](03_roadmap/02_future_features.md)
- [3D transition discovery brief](03_roadmap/03_3d_transition_discovery.md) —
  coupling audit, architecture recommendation, first executable slice, and
  owner decision gate.

### `04_archive/` — historical and experimental context

- [Idea archive](04_archive/01_idea_archive.md)
- [Experimental setup archive](04_archive/02_rl_experimental_setup.md)

Archived documents preserve context and are not operational instructions.

### `05_assets/` — documentation media

- Preview media is stored in `05_assets/Preview_vid/`.

## Source of truth

The current code is authoritative. Use `src/swarmecho/core/config.py`, the
selected level YAML, and the relevant module under `src/swarmecho/` to resolve
any disagreement in a document.
