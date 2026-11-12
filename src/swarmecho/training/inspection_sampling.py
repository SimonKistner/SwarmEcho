"""Resolve independent, adaptive, or uniform current-budget inspection targets."""

import numpy as np
import json
from pathlib import Path


def inspection_sampling_kwargs(level, spawn_sampler=None, *, checkpoint=None):
    """Use live state during training or the saved controller for checkpoint inspection.

    Arrays and the budget are copied before evaluation so curriculum advancement
    cannot change the distribution used by robustness passes or their artifacts.
    Curriculum inspection copies only the budget. Independent inspection and
    validation never receive these reset arguments.
    """
    mode = getattr(level.evaluation, "target_sampling", "independent")
    if mode == "independent":
        return {}
    if mode == "curriculum":
        budget = getattr(spawn_sampler, "curriculum_budget", None)
        if budget is None and checkpoint is not None:
            path = Path(checkpoint) / "adaptive_spawn_state.json"
            if not path.is_file():
                raise ValueError("Curriculum inspection requires checkpointed adaptive spawn state.")
            state = json.loads(path.read_text(encoding="utf-8"))
            budget = state.get("spawn_curriculum", {}).get("budget")
        minimum = level.training.adaptive_target_spawn.curriculum_start_agents or 1
        if type(budget) not in (int, np.int32, np.int64) or not minimum <= budget <= level.env.num_agents:
            raise ValueError("Curriculum inspection requires a valid live or checkpointed target budget.")
        # Only the feasibility limit is shared. Map and cell success weights stay in training.
        return {"target_agent_budget": np.int32(budget)}
    if mode != "training":
        raise ValueError("Unknown inspection target sampling mode: " + str(mode))
    if spawn_sampler is None and checkpoint is not None:
        from swarmecho.training.adaptive_spawn import AdaptiveTargetSpawnController

        spawn_sampler = AdaptiveTargetSpawnController(
            level.building, level.env, level.training.adaptive_target_spawn,
            feasibility_budget=(level.env.geometric_agent_budget
                                if level.training.dynamic_max_geo_distance.enabled else None),
            relay_options=level.training.dynamic_max_geo_distance,
            progression_config=level.evaluation,
        )
        if not spawn_sampler.restore(checkpoint):
            raise ValueError("Training target sampling requires adaptive_spawn_state.json in the checkpoint. "
                             "Use inspection_evaluation.target_sampling=independent for independent targets.")
    if spawn_sampler is None:
        raise ValueError("Training target sampling requires the live or checkpointed adaptive spawn controller.")
    budget = getattr(spawn_sampler, "curriculum_budget", None) or level.env.geometric_agent_budget
    return {"target_spawn_weights": np.array(spawn_sampler.box_weights, dtype=np.float32, copy=True),
            "target_agent_budget": np.int32(budget)}
