# System overview

SwarmEcho is a JAX-based multi-agent reinforcement-learning system for exploring
environments and establishing a communication relay from a base to a target.
The maintained training runtime uses recurrent MAPPO and optionally TarMAC.

## Runtime components

| Component | Responsibility |
| --- | --- |
| `env/environment.py` | State, reset/step, observations, coverage, relay rewards, and episode completion |
| `env/buildings.py` | Validate and compile authored maps |
| `env/obstacles.py` | Obstacle generation, visibility, and geodesic roadmaps |
| `env/critic.py` | Compact privileged critic features |
| `core/config.py` | Strict level loading and CLI overrides |
| `training/train.py` and `ppo.py` | Rollout collection, recurrent PPO, evaluation, and training lifecycle |
| `training/evaluate.py` | Parallel evaluation and selective replay generation |
| `visualize/replay.py` and `inspector.py` | Serialized replays and interactive inspection |
| `analysis/dashboard.py` | General cross-run analysis, including runs |

The general dashboard and inspector are separate applications. The building
editor is served by `maze_builder_server.py`; its historical filename and CLI
name remain for compatibility, but it serves only the building API.

Models, rollout buffers, checkpoints, artifact naming, run lifecycle, and
orchestration are shared infrastructure. Unsuffixed filenames do not imply
a separate environment implementation.

See [physics](02_environment_and_physics.md),
[training](03_marl_and_training.md), and [tools](../02_guide/02_analysis_and_tools.md).
