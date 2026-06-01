# Inactive & Unused Features Analysis

This document identifies codebase features in SwarmEcho that are currently inactive and unused (i.e. never executed or enabled in any default or level configuration).

---

## 1. Global Mean Critic (`GlobalMeanCritic`)
* **Location**: `src/models/critic.py` (Lines 67-99), `src/models/mappo.py`
* **Status**: **Inactive & Unused**. All default configurations and curriculum level configurations (A-series and B-series) use the `AgentCentricCritic` (which estimates individual agent values $V_i$). The `GlobalMeanCritic` is never enabled, and its code paths are never executed.
* **Refactoring Value**: **High**
  * **Rationale**: Eliminating this class reduces complexity in the Flax model definitions and removes the necessity of handling dual shapes (`(...,)` vs `(..., N)`) in the PPO trainer (`mappo_trainer.py`) and GAE value calculation loops (`runner.py`).
  * **Impacted Files**:
    - `src/models/critic.py` (Remove `GlobalMeanCritic` class definition)
    - `src/models/mappo.py` (Simplify wrapper to only instantiate `AgentCentricCritic`)
    - `src/core/config.py` (Remove `critic_type` option validation)
    - YAML configs (Remove comments referencing `global_mean`)

---

## 2. MEM_T8 Diagnostic Target Scaffolding
* **Location**: `src/curriculum_config/levels/MEM_T8_memory.yaml`, `src/curriculum_config/maps/memory_t_maze_8.yaml`, `src/tests/test_memory/`, plus guarded compatibility hooks in `src/env/`, `src/training/runner.py`, and `src/visualize/`.
* **Status**: **Diagnostic-Only**. These paths are not part of the normal one-target SwarmEcho relay-chain task. They exist only so the memory T-maze can run eight isolated cue/choice tasks inside one environment instance.
* **Included Hooks**:
  * `target_pos.shape == (N, 2)` support for MEM_T8 per-agent target slots.
  * `spawn_points.anti_target` and `anti_target_known` for paired wrong-branch decoys.
  * one-shot `-finder_bonus` anti-target penalty.
  * `target_found_fraction` reporting so MEM_T8 can log `0/8` through `8/8` instead of the normal binary target-found metric.
  * renderer support for drawing MEM_T8 anti-target markers.
  * `env.mem_test_mask_nonlocal_obs`, which keeps the normal observation shape but zeros leak-prone non-local channels for the memory diagnostic.
* **Refactoring Value**: **Low while memory experiments are active**
  * **Rationale**: The scaffolding is deliberately isolated and disabled for normal single-target levels. It should remain clearly marked as MEM_T8-only and should not be referenced from the main system docs.
  * **Impacted Files if removed**:
    - `src/curriculum_config/levels/MEM_T8_memory.yaml`
    - `src/curriculum_config/maps/memory_t_maze_8.yaml`
    - `src/tests/test_memory/`
    - MEM_T8-only branches in `src/env/maps.py`, `src/env/physics.py`, `src/env/rewards.py`, `src/env/state.py`
    - MEM_T8-only observation masking in `src/env/observations.py`
    - MEM_T8-only metric fallback in `src/training/runner.py`
    - MEM_T8-only marker rendering in `src/visualize/renderer_cv2.py` and `src/visualize/renderer_mpl.py`
