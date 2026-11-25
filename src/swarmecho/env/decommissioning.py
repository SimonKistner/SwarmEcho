"""Automatic deployment and permanent decommissioning."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from swarmecho.core.config import DecommissioningConfig


class AdaptiveState(NamedTuple):
    ids: jax.Array
    zone_center: jax.Array
    zone_valid: jax.Array
    dwell: jax.Array
    decommissions: jax.Array
    spawned: jax.Array
    decommissioned: jax.Array
    all_decommissioned: jax.Array
    peer_credit_count: jax.Array
    step: jax.Array
    last_spawn_step: jax.Array


def initial_state(num_agents, zone_center, zone_valid=True):
    flags = jnp.zeros(num_agents, dtype=jnp.bool_)
    zero = jnp.int32(0)
    return AdaptiveState(
        jnp.arange(num_agents), zone_center, jnp.asarray(zone_valid),
        jnp.zeros(num_agents, dtype=jnp.int32),
        zero, flags, flags, jnp.bool_(False), zero, zero, zero,
    )


def transition(state, active, pos, cfg: DecommissioningConfig, *, spawn_delay):
    """Advance staggered launches and remove drones that completed their dwell."""
    inside = (jnp.linalg.norm(pos - state.zone_center, axis=-1)
              <= cfg.decommission_radius_m) & active & state.zone_valid
    dwell = jnp.where(inside, state.dwell + 1, 0)
    removed = active & (dwell >= cfg.decommission_hold_steps)
    alive = active & ~removed
    tick = state.step + 1
    reserved = ~active & (state.ids >= 0)
    due = reserved & (tick >= jnp.arange(active.shape[0]) * spawn_delay)
    spawned = due & ((tick - state.last_spawn_step) >= spawn_delay)
    if spawn_delay > 0:
        spawned &= jnp.arange(active.shape[0]) == jnp.argmax(due)
    live = alive | spawned
    return state._replace(
        ids=jnp.where(removed, -1, state.ids), step=tick, spawned=spawned, decommissioned=removed,
        dwell=jnp.where(live & ~spawned, dwell, 0),
        last_spawn_step=jnp.where(jnp.any(spawned), tick, state.last_spawn_step),
        decommissions=state.decommissions + jnp.sum(removed, dtype=jnp.int32),
        all_decommissioned=~jnp.any(live | (reserved & ~spawned)),
    ), live


def base_population_feedback(active, connected_to_base, population_reference):
    """Give the instantaneous active fleet size to connected drones."""
    population = jnp.sum(active).astype(jnp.float32) / population_reference
    return jnp.where(connected_to_base & active, population, 0.0)


def lifetime_ended(previous, current):
    """Return the lifetimes permanently removed on this transition."""
    return previous.active & ~current.active


def policy_resets(state):
    return ~state.active if state.adaptive is None else (~state.active | state.adaptive.spawned)
