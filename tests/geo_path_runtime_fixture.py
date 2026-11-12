"""Tiny deterministic actor/environment exercising the real shared JAX collector."""
from types import SimpleNamespace
from typing import NamedTuple

import jax.numpy as jnp
from flax import nnx
import numpy as np


class State(NamedTuple):
    pos: object
    active: object
    target_known: object
    directly_sees_target: object
    base_pos: object
    target_pos: object
    solid_min: object
    solid_max: object
    step: object
    done: object
    success: object
    base_target_known: object
    target_required_agents: object
    spawn_pair_attempts: object
    adaptive: object = None


class Actor(nnx.Module):
    vote_mode = None

    def __call_team__(self, obs, hidden, signature, value, **kwargs):
        updated = jnp.broadcast_to(obs[:, :1] + 1., hidden.shape)
        return updated, signature + 1., value + 1., jnp.zeros((2, 3)), None


class Model(nnx.Module):
    hidden_dim = 3
    tarmac_sig_dim = 1
    tarmac_val_dim = 1
    memory_comm_every_k_steps = 2

    def __init__(self):
        self.actor = Actor()

    def initial_actor_hidden(self, shape, *, num_agents):
        return jnp.zeros((*shape, num_agents, 3))

    def initial_actor_signature(self, shape, *, num_agents):
        return jnp.zeros((*shape, num_agents, 1))

    initial_actor_value = initial_actor_signature


def environment():
    def reset(key):
        return State(jnp.array([[2.5, 2.5, 2.5], [3., 2.5, 2.5]]), jnp.array([True, True]),
            jnp.array([False, False]), jnp.array([False, False]), jnp.array([2.5, 2.5, 2.5]),
            jnp.array([7.5, 2.5, 2.5]), jnp.empty((0, 3)), jnp.empty((0, 3)),
            jnp.int32(0), jnp.bool_(False), jnp.bool_(False), jnp.bool_(False), jnp.int32(1), jnp.int32(1))

    def step(state, action):
        t = state.step + 1
        return state._replace(step=t, done=t >= 6, active=jnp.array([t < 4, True]),
            target_known=jnp.array([t >= 1, t >= 3]), directly_sees_target=jnp.array([t >= 1, False]),
            base_target_known=t >= 3)

    def observations(state):
        return jnp.full((2, 1), state.step, dtype=jnp.float32)

    return reset, step, observations, None


def level():
    return SimpleNamespace(env=SimpleNamespace(max_steps=6, num_agents=2, comm_radius=10., comm_radius_base=10., base_keeps_informing=False),
        building_name="Deterministic smoke fixture", map_names=["fixture"],
        building=SimpleNamespace(solid_min_m=np.empty((0, 3)), solid_max_m=np.empty((0, 3)),
            interior_cells=np.array([[0,0,0],[1,0,0]]), cell_size_m=5., vertices=np.empty((0,3))))
