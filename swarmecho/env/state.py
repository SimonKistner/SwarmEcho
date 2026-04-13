"""
swarmecho/env/state.py
======================
JAX PyTree definition of the SwarmEcho environment state.

Design notes
------------
* Registered as a JAX PyTree via `jax.tree_util.register_dataclass` so it
  can be passed through jit / vmap / lax.scan without any modifications.
* All fields are JAX arrays — no Python scalars inside the state.
* Use `dataclasses.replace(state, field=new_val)` to produce updated copies
  (JAX functional style — never mutate in-place).
* Shape comments assume N = num_agents, G = grid_resolution.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import chex
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# EnvState
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class EnvState:
    """
    Immutable snapshot of the environment at a single timestep.

    Fields
    ------
    pos           : (N, 2)  Agent x,y positions  [metres]
    vel           : (N, 2)  Agent x,y velocities  [m/s]
    base_pos      : (2,)    Base station position  [fixed across episode]
    target_pos    : (2,)    Target position        [fixed across episode]
    coverage_grid : (G, G)  Boolean exploration map — True where visited
    step          : ()      int32 timestep counter
    key           : (2,)    JAX PRNGKey for in-step randomness
    """

    pos:           jax.Array   # (N, 2)  float32
    vel:           jax.Array   # (N, 2)  float32
    base_pos:      jax.Array   # (2,)    float32  — fixed, replicated for vmap
    target_pos:    jax.Array   # (2,)    float32  — fixed, replicated for vmap
    coverage_grid: jax.Array   # (G, G)  bool
    step:          jax.Array   # ()      int32
    key:           jax.Array   # (2,)    uint32   PRNGKey


# Register so jit/vmap/scan can traverse the fields automatically.
# meta_fields=[] means ALL fields are dynamic (traced) — correct for arrays.
jax.tree_util.register_dataclass(
    EnvState,
    data_fields=[
        "pos", "vel", "base_pos", "target_pos",
        "coverage_grid", "step", "key",
    ],
    meta_fields=[],
)


# ---------------------------------------------------------------------------
# Shape / dtype assertions — call freely during dev, stripped in production
# via chex.disable_asserts().
# ---------------------------------------------------------------------------

def assert_env_state(state: EnvState, N: int, G: int) -> None:
    """Verify all state fields have the expected shapes and dtypes."""
    chex.assert_shape(state.pos,           (N, 2))
    chex.assert_shape(state.vel,           (N, 2))
    chex.assert_shape(state.base_pos,      (2,))
    chex.assert_shape(state.target_pos,    (2,))
    chex.assert_shape(state.coverage_grid, (G, G))
    chex.assert_shape(state.step,          ())
    chex.assert_shape(state.key,           (2,))

    chex.assert_type(state.pos,           jnp.float32)
    chex.assert_type(state.vel,           jnp.float32)
    chex.assert_type(state.base_pos,      jnp.float32)
    chex.assert_type(state.target_pos,    jnp.float32)
    chex.assert_type(state.coverage_grid, jnp.bool_)
    chex.assert_type(state.step,          jnp.int32)
