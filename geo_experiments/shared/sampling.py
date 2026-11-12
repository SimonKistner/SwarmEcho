"""Episode/agent balanced sampling shared by snapshot probes and their reports."""
import numpy as np


def retained_candidates(steps, cap):
    """Uniform indices over the eligible stride grid; no padding or duplicates."""
    available = np.flatnonzero(np.asarray(steps) >= 0)
    if len(available) <= cap:
        return available
    return available[np.rint(np.linspace(0, len(available) - 1, cap)).astype(int)]


def hierarchical_weights(episode, agent):
    """Normalized snapshot weights: equal episode, then agent, then snapshot."""
    episode, agent = np.asarray(episode), np.asarray(agent)
    if not len(episode):
        return np.empty(0, dtype=float)
    pairs = np.column_stack((episode, agent))
    unique, inverse, counts = np.unique(pairs, axis=0, return_inverse=True, return_counts=True)
    episodes, agents_per_episode = np.unique(unique[:, 0], return_counts=True)
    per_pair = agents_per_episode[np.searchsorted(episodes, unique[:, 0])]
    return 1. / (len(episodes) * per_pair[inverse] * counts[inverse])


class HierarchicalSampler:
    """Vectorized three-stage uniform draws, without scanning all rows per batch."""
    def __init__(self, episode, agent, rng):
        pairs, inverse, counts = np.unique(np.column_stack((episode, agent)), axis=0,
                                           return_inverse=True, return_counts=True)
        if not len(pairs):
            raise ValueError("Cannot sample an empty input population.")
        _, episode_counts = np.unique(pairs[:, 0], return_counts=True)
        self.rng = rng
        self.episode_counts = episode_counts
        self.episode_offsets = np.cumsum(np.r_[0, episode_counts[:-1]])
        self.pair_counts = counts
        self.pair_offsets = np.cumsum(np.r_[0, counts[:-1]])
        self.rows = np.argsort(inverse, kind="stable")

    def draw(self, count):
        episode = self.rng.integers(len(self.episode_counts), size=count)
        pair = self.episode_offsets[episode] + (self.rng.random(count) * self.episode_counts[episode]).astype(int)
        offset = self.pair_offsets[pair] + (self.rng.random(count) * self.pair_counts[pair]).astype(int)
        return self.rows[offset]


def mask_scores(probabilities, target, threshold=.5):
    """Exact matches and symmetric path overlap; empty background earns no points."""
    predicted = np.asarray(probabilities) >= threshold
    target = np.asarray(target, dtype=bool)
    shared = (predicted & target).sum(axis=-1)
    total = predicted.sum(axis=-1) + target.sum(axis=-1)
    similarity = np.divide(2. * shared, total, out=np.ones_like(shared, dtype=float), where=total != 0)
    return (predicted == target).all(axis=-1).astype(float), similarity
