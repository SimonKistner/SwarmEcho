"""Validation-gated target groups, independent of adaptive spawn weights."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from swarmecho.training.adaptive_spawn import atomic_json


class UniformTargetCurriculum:
    curriculum_source = "validation"

    def __init__(self, config, maximum, *, skip_empty_forward_groups=False):
        self.config = config
        self.maximum = maximum
        self.group = config.start_agents if config.mode == "forward" else maximum
        self.skip_empty_forward_groups = skip_empty_forward_groups

    @property
    def curriculum_budget(self):
        return self.group if self.config.mode == "forward" else self.maximum

    @property
    def minimum(self):
        return 1 if self.config.mode == "forward" else self.group

    @property
    def complete(self):
        return self.group == (self.maximum if self.config.mode == "forward" else self.config.start_agents)

    def reset_kwargs(self):
        return {"target_agent_budget": np.int32(self.curriculum_budget),
                "target_min_agents": np.int32(self.minimum)}

    def observe_curriculum(self, source, required_agents, successes, lengths, max_steps):
        if source != "validation":
            return {}
        required, success = np.asarray(required_agents), np.asarray(successes)
        if required.shape != success.shape or np.any((required < 1) | (required > self.maximum)):
            raise ValueError("Target curriculum needs certified per-case requirements and successes.")
        group = self.group
        selected = required == group
        count = int(selected.sum())
        rate = float(success[selected].mean()) if count else 0.0
        if not self.complete:
            if self.config.mode == "forward" and self.skip_empty_forward_groups:
                # Fixed same-map cases may contain no examples of a group.
                # Gate on the hardest available group within the current budget.
                current = required[required <= group]
                if not count and current.size:
                    selected = required == current.max()
                    count = int(selected.sum())
                    rate = float(success[selected].mean())
                if count and rate >= self.config.success_threshold:
                    higher = required[required > group]
                    self.group = int(higher.min()) if higher.size else self.maximum
            elif count and rate >= self.config.success_threshold:
                self.group += 1 if self.config.mode == "forward" else -1
        return {"group": group, "next_group": self.group, "success_rate": rate,
                "eligible_episodes": count, "target_budget": self.curriculum_budget}

    def save(self, checkpoint):
        atomic_json(Path(checkpoint) / "target_curriculum_state.json", {
            "format": "swarmecho-target-curriculum/v1", "config": asdict(self.config),
            "maximum": self.maximum, "group": self.group})

    def restore(self, checkpoint, *, strict=True):
        """Restore matching stages; branches may start fresh on another curriculum."""
        path = Path(checkpoint) / "target_curriculum_state.json"
        if not path.is_file():
            if not strict:
                return False
            raise ValueError("Resume requires target_curriculum_state.json.")
        saved = json.loads(path.read_text(encoding="utf-8"))
        saved_config = saved.get("config")
        if isinstance(saved_config, dict):
            saved_config = {"start_agents": 2, **saved_config}
        if (saved.get("format") != "swarmecho-target-curriculum/v1"
                or saved_config != asdict(self.config) or saved.get("maximum") != self.maximum
                or type(saved.get("group")) is not int
                or not self.config.start_agents <= saved["group"] <= self.maximum):
            if not strict:
                return False
            raise ValueError("Target curriculum checkpoint settings changed; start a new run.")
        self.group = saved["group"]
        return True
