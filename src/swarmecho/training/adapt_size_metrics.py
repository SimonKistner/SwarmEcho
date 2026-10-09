"""Completed-episode adaptive diagnostics; the caller publishes only to W&B."""
from collections import deque
import numpy as np

REWARD_NAMES = (
    "reinforcement_cost", "decommission_cost", "rejected_call_penalty",
    "all_decommissioned_penalty", "useless_agetnst_at_success_penalty",
)
INFO_NAMES = (
    "adapt_calls", "adapt_decommissions", "adapt_rejected_calls", "adapt_end_agents",
    "adapt_all_decommissioned", "adapt_useless_agents", "adapt_zone_valid",
)


def validate_zones(valid):
    if not np.all(np.asarray(valid)):
        raise ValueError("No reachable decommission sphere fits above a base. "
                         "Reduce adapt_size.decommission_radius_m or adjust the base layout.")


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
        calls, decomm = values("adapt_calls"), values("adapt_decommissions")
        success = values("success").astype(bool)
        end = values("adapt_end_agents")
        returned = np.minimum(calls, decomm)
        result = {
            "calls_mean": calls.mean(), "decommissions_mean": decomm.mean(),
            "active_agents_at_EP_end_mean/all": end.mean(),
            "called_decommissioned_mean": returned.mean(),
            "total_calls": calls.sum(), "total_decommissions": decomm.sum(),
            "decommission_termination_rate": values("adapt_all_decommissioned").mean(),
            "rejected_calls_mean": values("adapt_rejected_calls").mean(),
        }
        for label, subset in (("success", success), ("fail", ~success)):
            if subset.any():
                result[f"active_agents_at_EP_end_mean/{label}"] = end[subset].mean()
        if success.any():
            result["useless_agents_at_success"] = values("adapt_useless_agents")[success].mean()
        if (calls > 0).any():
            result["called_decommissioned_fraction"] = (returned[calls > 0] / calls[calls > 0]).mean()
            result["called_decommissioned_fraction_pooled"] = returned.sum() / calls.sum()
        if decomm.sum() > 0:
            result["total_calls_to_decommissions_ratio"] = calls.sum() / decomm.sum()
        return {f"adapt_size/{k}": float(v) for k, v in result.items()}
