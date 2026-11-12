"""Pool recipient counts before computing information-source percentages."""

import numpy as np


INFORMATION_PERCENTAGES = ("peer-informed-pct", "base-informed-pct")
EVALUATION_INFORMATION_COUNTS = (
    "eval_informed_count", "eval_peer_informed_count", "eval_base_informed_count",
)
EVALUATION_INFORMATION_KEYS = (*EVALUATION_INFORMATION_COUNTS,
                               *("eval_" + name for name in INFORMATION_PERCENTAGES))


def information_percentages(counts):
    """Share of all informed agent lives, including direct visual discoverers.

    Counts may be a single [total, peer, base] vector or an episode-by-count
    array. Pool agents, rather than average episode percentages. No observation
    is reported when nobody learned the target; zero would imply an observation.
    """
    total, peer, base = np.asarray(counts, dtype=np.int64).reshape(-1, 3).sum(axis=0)
    if total == 0:
        return {}
    return dict(zip(INFORMATION_PERCENTAGES, (100.0 * peer / total, 100.0 * base / total)))


def evaluation_information_metrics(counts):
    """Keep raw counts in evaluation artifacts so map/run aggregation is exact."""
    counts = np.asarray(counts, dtype=np.int64).reshape(-1, 3).sum(axis=0)
    return {**dict(zip(EVALUATION_INFORMATION_COUNTS, map(int, counts))),
            **{"eval_" + name: value for name, value in information_percentages(counts).items()}}


def pool_evaluation_information(metrics):
    """Combine all actual evaluated episodes across maps or robustness passes."""
    if not all(all(key in row for key in EVALUATION_INFORMATION_COUNTS) for row in metrics):
        return {}
    return evaluation_information_metrics([[row[key] for key in EVALUATION_INFORMATION_COUNTS]
                                           for row in metrics])
