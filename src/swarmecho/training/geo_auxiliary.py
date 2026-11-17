"""Path supervision, balanced sampling and the online decoder.

The environment supplies labels, never privileged policy inputs. Training labels
are built once per sampled episode and shared by all agents/steps/PPO epochs.
"""
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import json

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from swarmecho.analysis.path_targets import ReferencePlanner, NoReferenceRoute
from swarmecho.core.config import DEFAULT_PATH_GRID_SHAPE


PREFIX = "geo_probe_auxiliary/"
POLICY_FILTER = nnx.All(nnx.Param, nnx.Not(nnx.PathContains("geo_probe_head")))


class PathHead(nnx.Module):
    def __init__(self, hidden_dim, config, seed, grid_shape):
        self.grid_shape = grid_shape
        rng = nnx.Rngs(seed ^ 0x6E0A)
        widths = (hidden_dim, *config.head.hidden_sizes, int(np.prod(grid_shape)))
        self.layers = nnx.List([nnx.Linear(a, b, rngs=rng) for a,b in zip(widths[:-1], widths[1:])])
        self.supervised_updates = nnx.Variable(jnp.int32(0))
        self.lambda_target = config.lambda_target
        self.warmup = config.schedule.warmup_updates
        self.ramp = config.schedule.ramp_updates
        self.threshold = config.metric_threshold

    def __call__(self, value):
        for layer in self.layers[:-1]:
            value = jax.nn.relu(layer(value))
        return self.layers[-1](value)

    def weight(self):
        u = self.supervised_updates[...].astype(jnp.float32)
        return self.lambda_target * jnp.clip((u - self.warmup) / max(1, self.ramp), 0., 1.)


def scores(probabilities, target, threshold):
    """Per-snapshot exact mask, thresholded overlap and probability overlap."""
    selected = probabilities >= threshold
    truth = target.astype(bool)
    count = jnp.sum(selected, -1) + jnp.sum(truth, -1)
    exact = jnp.all(selected == truth, -1).astype(jnp.float32)
    overlap = 2 * jnp.sum(selected & truth, -1) / jnp.maximum(count, 1)
    soft = 2 * jnp.sum(probabilities * target, -1) / jnp.maximum(jnp.sum(probabilities, -1) + jnp.sum(target, -1), 1e-8)
    return jnp.stack((exact, overlap, soft), -1)


@nnx.jit
def score_head(head, features, targets, threshold):
    return scores(jax.nn.sigmoid(head(features)), targets, threshold)


def balanced_weights(episodes, agents, valid):
    """Equal represented episodes, agents within episode and snapshots per agent."""
    result = np.zeros(valid.shape, np.float32)
    flat = np.flatnonzero(valid)
    if not len(flat):
        return result
    ep = episodes.ravel()[flat]
    ag = agents.ravel()[flat]
    pairs, inverse, counts = np.unique(np.column_stack((ep,ag)), axis=0, return_inverse=True, return_counts=True)
    _, per_episode = np.unique(pairs[:,0], return_counts=True)
    ids = np.searchsorted(np.unique(pairs[:,0]), pairs[:,0])
    result.ravel()[flat] = 1. / (len(per_episode) * per_episode[ids][inverse] * counts[inverse])
    return result


class PathLabels:
    def __init__(self, paths, cfg, *, buildings=None, records=None, directory=None,
                 grid_shape=DEFAULT_PATH_GRID_SHAPE):
        from swarmecho.env.buildings import load_building
        self.grid_width = int(np.prod(grid_shape))
        self.planners = []
        for i, path in enumerate(paths):
            path = Path(path)
            building = buildings[i] if buildings is not None else load_building(path)
            graph = None
            if records is not None:
                r = records[i]
                graph = SimpleNamespace(vertices=r["vertices"], distances=r["distances"],
                    clearance=cfg.drone_radius + cfg.obstacle_planning_clearance_m)
            self.planners.append(ReferencePlanner(building, cfg,
                path.parent.parent / "records" / (path.stem + ".npz"), graph=graph, grid_shape=grid_shape))
        self.paths = list(map(str, paths))
        self.cache = OrderedDict()
        self.requested = self.missing = 0
        self.directory = Path(directory) if directory is not None else None

    def get(self, map_id, base, target):
        key = (int(map_id), np.asarray(base, np.float32).tobytes(), np.asarray(target,np.float32).tobytes())
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        self.requested += 1
        try:
            mask = self.planners[int(map_id)].route(base, target)["mask"]
        except NoReferenceRoute as error:
            self.missing += 1
            mask = None
            if self.directory is not None:
                self.directory.mkdir(parents=True, exist_ok=True)
                with (self.directory / "missing_reference.jsonl").open("a", encoding="utf-8") as out:
                    out.write(json.dumps(dict(map=self.paths[int(map_id)], base=np.asarray(base).tolist(),
                        target=np.asarray(target).tolist(), error=str(error))) + "\n")
        except ValueError as error:
            raise ValueError(f"Auxiliary path label failed: map={self.paths[int(map_id)]}, base={base}, target={target}") from error
        self.cache[key] = mask
        # Retain many concurrent episodes while bounding host memory over a run.
        if len(self.cache) > 100000:
            self.cache.popitem(last=False)
        return mask


class AuxiliaryData:
    """Host-only, streaming selections and summaries; never stores GRU snapshots."""
    def __init__(self, config, seed, labels):
        self.config, self.seed, self.labels = config, seed, labels
        self.next_episode = 0
        self.live = None
        self.pending = {}

    def prepare(self, rollout, update):
        T,E,N = rollout["active_masks"].shape
        if self.live is None:
            self.live = np.arange(E, dtype=np.int64)
            self.next_episode = E
        episodes = np.empty((T,E), np.int64)
        for t in range(T):
            episodes[t] = self.live
            done = np.flatnonzero(rollout["dones"][t])
            self.live[done] = np.arange(self.next_episode, self.next_episode+len(done))
            self.next_episode += len(done)
        K = self.config.sampling.snapshots_per_agent
        times = np.full((E,N,K), -1, np.int32)
        targets = np.zeros((E, N, K, self.labels.grid_width), bool)
        ep = np.full((E,N,K), -1, np.int64)
        valid = np.zeros((E,N,K), bool)
        if update % self.config.sampling.every_updates == 0:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, 0x6E0A, update]))
            eligibility = rollout["geo_target_known"] & rollout["active_masks"]
            counts = eligibility.sum(axis=0)
            bands = np.minimum(K, counts)
            rank = np.cumsum(eligibility, axis=0)-1
            group = rank * bands[None] // np.maximum(counts[None],1)
            random = rng.random(eligibility.shape, dtype=np.float32)
            for k in range(K):
                candidates = eligibility & (group == k)
                chosen = np.argmin(np.where(candidates, random, np.inf), axis=0)
                times[:,:,k] = np.where(candidates.any(axis=0), chosen, -1)
            valid = times >= 0
            lane = np.broadcast_to(np.arange(E)[:,None,None], valid.shape)
            tt,ee = times[valid],lane[valid]
            selected_episodes = episodes[tt,ee]
            unique,first,inverse = np.unique(selected_episodes,return_index=True,return_inverse=True)
            masks = np.zeros((len(unique), self.labels.grid_width), bool)
            available = np.zeros(len(unique),bool)
            for i,index in enumerate(first):
                t,e = tt[index],ee[index]
                label = self.labels.get(rollout["geo_map_id"][t,e], rollout["geo_base"][t,e], rollout["geo_target"][t,e])
                if label is not None:
                    masks[i],available[i] = label,True
            targets[valid],ep[valid] = masks[inverse],selected_episodes
            valid[valid] = available[inverse]
            times[~valid] = -1
        self.selection = dict(aux_times=times, aux_targets=targets, aux_valid=valid, aux_episodes=ep)
        self.selection_agents = np.broadcast_to(np.arange(N)[None,:,None], valid.shape)
        self.episode_frames = episodes
        return self.selection

    def add_minibatches(self, minibatches):
        for mb in minibatches:
            # The buffer returns the same permutation IDs used for sequence slicing.
            ix = np.asarray(mb["env_indices"])
            array = np.asarray if isinstance(mb["obs"],np.ndarray) else jnp.asarray
            for key,value in self.selection.items():
                mb[key] = array(value[ix])
            mb["aux_weights"] = array(balanced_weights(self.selection["aux_episodes"][ix],
                self.selection_agents[ix], self.selection["aux_valid"][ix]))

    def record_scores(self, env_indices, values):
        ix = np.asarray(env_indices)
        valid = self.selection["aux_valid"][ix]
        pairs = np.column_stack((self.selection["aux_episodes"][ix][valid],self.selection_agents[ix][valid]))
        if not len(pairs):
            return
        pairs,inverse,counts = np.unique(pairs,axis=0,return_inverse=True,return_counts=True)
        totals = np.zeros((len(pairs),3),float)
        np.add.at(totals,inverse,values[valid])
        for (episode,a),total,count in zip(pairs,totals,counts):
            agents = self.pending.setdefault(int(episode), {})
            previous,n = agents.get(int(a), (np.zeros(3),0))
            agents[int(a)] = (previous+total,n+int(count))

    def finish_episode(self, episode):
        agents = self.pending.pop(int(episode), {})
        return np.mean([total/count for total,count in agents.values()], axis=0) if agents else None


def gradients_diagnostics(ppo_grads, path_grads, coefficient):
    """Compare PPO and path gradients on the actor encoder, GRU, and TarMAC projections."""
    def shared(state):
        return {k:v for k,v in state["actor"].items() if k in
            {"encoder", "gru", "tarmac_query", "tarmac_signature", "tarmac_value"}}
    a,b = jax.tree.leaves(shared(ppo_grads)), jax.tree.leaves(shared(path_grads))
    aa=sum(jnp.sum(x*x) for x in a);bb=sum(jnp.sum(x*x) for x in b)
    ab=sum(jnp.sum(x*y) for x,y in zip(a,b))
    return {PREFIX+"gradient/aux_to_ppo_ratio":jnp.where(aa>1e-20,coefficient*jnp.sqrt(bb/aa),jnp.nan),
            PREFIX+"gradient/cosine_similarity":jnp.where((aa>1e-20)&(bb>1e-20),ab/jnp.sqrt(aa*bb),jnp.nan)}
