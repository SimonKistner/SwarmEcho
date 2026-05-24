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
