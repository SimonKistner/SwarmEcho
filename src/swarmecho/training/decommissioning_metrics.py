"""Completed-episode adaptive diagnostics; the caller publishes only to W&B."""
from collections import deque
import numpy as np

REWARD_NAMES = ("all_decommissioned_penalty",)
NON_CONTRIBUTOR_REWARD_NAMES = (
    "non_contributor_return_reward", "non_contributor_departure_reward",
    "non_contributor_decommission_reward",
)
INFO_NAMES = (
    "adapt_decommissions", "adapt_end_agents",
    "adapt_all_decommissioned", "adapt_useless_agents", "adapt_zone_valid",
)


def validate_zones(valid):
    if not np.all(np.asarray(valid)):
        raise ValueError("No reachable decommission sphere fits above a base. "
                         "Reduce decommissioning.decommission_radius_m or adjust the base layout.")


class AdaptiveWindow:
    def __init__(self, size):
        self.episodes = deque(maxlen=size)

    def append(self, info, *, success):
        validate_zones(info["adapt_zone_valid"])
        self.episodes.append({**{k: float(info[k]) for k in INFO_NAMES}, "success": bool(success)})

    def metrics(self):
        if len(self.episodes) != self.episodes.maxlen:
            return {}
        rows = list(self.episodes)
        values = lambda name: np.asarray([r[name] for r in rows])
        decomm = values("adapt_decommissions")
        success = values("success").astype(bool)
        end = values("adapt_end_agents")
        result = {
            "decommissions_mean": decomm.mean(),
            "active_agents_at_EP_end_mean/all": end.mean(),
            "total_decommissions": decomm.sum(),
            "decommission_termination_rate": values("adapt_all_decommissioned").mean(),
        }
        for label, subset in (("success", success), ("fail", ~success)):
            if subset.any():
                result[f"active_agents_at_EP_end_mean/{label}"] = end[subset].mean()
        if success.any():
            result["useless_agents_at_success"] = values("adapt_useless_agents")[success].mean()
        return {f"decommissioning/{k}": float(v) for k, v in result.items()}


class RequiredAgentsDeploymentWindow:
    """Category exposure in the ordinary episode window, plus local outcomes.

    Population and success use independent smaller windows for each exact
    accepted estimator count. All completed episodes enter both window types.
    """

    def __init__(self, size, *, episode_window_size, max_required_agents):
        for value in (size, episode_window_size, max_required_agents):
            if type(value) is not int or value < 1:
                raise ValueError("Required-agent metric windows and category count must be positive integers.")
        self.size = size
        self.max_required_agents = max_required_agents
        self.windows = {count: deque(maxlen=size) for count in range(1, max_required_agents + 1)}
        self.episode_categories = deque(maxlen=episode_window_size)
        self.category_counts = np.zeros(max_required_agents + 1, dtype=np.int64)

    def append(self, required_agents, active_agents_at_end, *, success):
        required_agents = int(required_agents)
        if not 1 <= required_agents <= self.max_required_agents:
            raise ValueError("Completed target estimator count is outside the current geometric budget.")
        if len(self.episode_categories) == self.episode_categories.maxlen:
            self.category_counts[self.episode_categories[0]] -= 1
        self.episode_categories.append(required_agents)
        self.category_counts[required_agents] += 1
        self.windows[required_agents].append((int(active_agents_at_end), bool(success)))

    def metrics(self):
        result = {}
        full = len(self.episode_categories) == self.episode_categories.maxlen
        for required_agents, window in self.windows.items():
            prefix = f"req_agent_{required_agents}"
            if full:
                result[f"{prefix}/total_eps_in_eps_window"] = (
                    float(self.category_counts[required_agents]) / self.episode_categories.maxlen
                )
            if window:
                result[f"{prefix}/active_agents_at_EP_end_mean"] = float(np.mean([row[0] for row in window]))
                result[f"{prefix}/success_rate"] = float(np.mean([row[1] for row in window]))
        return result
