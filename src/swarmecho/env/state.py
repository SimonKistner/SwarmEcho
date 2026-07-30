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
* Shape comments assume N = num_agents, GW/GH = grid width/height.
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
    target_pos    : (2,) Target position [fixed across episode]
    coverage_grid : (GW, GH) Boolean exploration map — True where visited
    step          : ()      int32 timestep counter
    key           : (2,)    JAX PRNGKey for in-step randomness
    active        : (N,)    bool  — False until step == i*spawn_delay
    target_known  : (N,)    bool  — persistent: True once informed via comm
    collides      : (N,)    bool  — True if agent hit wall/obstacle this step
    last_cov_delta: (N,)    int32 — number of new cells covered this step
    box_width     : ()      float32 — dynamic world width
    box_height    : ()      float32 — dynamic world height
    base_target_known: ()   bool  — persistent: True once base is informed
    chain_held_steps:  ()   int32 — consecutive timesteps chain has been held
    is_conn_base  : (N,)    bool  — True if agent is connected to base
    is_conn_target: (N,)    bool  — True if agent is connected to target
    directly_sees_target: (N,) bool — direct range-and-LoS target visibility
    adj_matrix    : (N+1, N+1) bool — direct communication adjacency matrix
    """

    pos:           jax.Array   # (N, 2)  float32
    vel:           jax.Array   # (N, 2)  float32
    base_pos:      jax.Array   # (2,)    float32  — fixed, replicated for vmap
    target_pos:    jax.Array   # (2,)    float32
    coverage_grid: jax.Array   # (GW, GH) bool
    step:          jax.Array   # ()      int32
    key:           jax.Array   # (2,)    uint32   PRNGKey
    active:        jax.Array   # (N,)    bool  — False until step == i*spawn_delay
    target_known:  jax.Array   # (N,)    bool  — persistent: True once informed via comm
    collides:      jax.Array   # (N,)    bool  — True if agent hit wall/obstacle this step
    last_cov_delta:jax.Array   # (N,)    int32 — number of new cells covered this step
    box_width:     jax.Array   # ()      float32 — dynamic world dimensions
    box_height:    jax.Array   # ()      float32
    base_target_known: jax.Array # ()    bool
    chain_held_steps:  jax.Array # ()    int32
    is_conn_base:      jax.Array # (N,)  bool
    is_conn_target:    jax.Array # (N,)  bool
    directly_sees_target: jax.Array # (N,) bool
    finder_path_cells: jax.Array = dataclasses.field(
        default_factory=lambda: jnp.zeros((0, 0, 2), dtype=jnp.int16)
    )
    finder_path_lens: jax.Array = dataclasses.field(
        default_factory=lambda: jnp.zeros((0,), dtype=jnp.int16)
    )
    finder_path_active: jax.Array = dataclasses.field(
        default_factory=lambda: jnp.zeros((0,), dtype=jnp.bool_)
    )
    target_known_path_cells: jax.Array = dataclasses.field(
        default_factory=lambda: jnp.zeros((0, 0, 2), dtype=jnp.int16)
    )
    target_known_path_lens: jax.Array = dataclasses.field(
        default_factory=lambda: jnp.zeros((0,), dtype=jnp.int16)
    )
    target_known_path_valid: jax.Array = dataclasses.field(
        default_factory=lambda: jnp.zeros((0,), dtype=jnp.bool_)
    )
    finders_path: jax.Array = dataclasses.field(
        default_factory=lambda: jnp.zeros((0, 2), dtype=jnp.int16)
    )
    finders_path_len: jax.Array = dataclasses.field(
        default_factory=lambda: jnp.int16(0)
    )
    finders_path_valid: jax.Array = dataclasses.field(
        default_factory=lambda: jnp.bool_(False)
    )
    finders_path_index_grid: jax.Array = dataclasses.field(
        default_factory=lambda: jnp.zeros((0, 0), dtype=jnp.int16)
    )
    # Direct communication adjacency matrix (excluding self-loops)
    adj_matrix:        jax.Array = dataclasses.field(
        default_factory=lambda: jnp.zeros((0, 0), dtype=jnp.bool_)
    )
    # (Removed static world data from PyTree to save VRAM)

    def replace(self, **kwargs) -> EnvState:
        return dataclasses.replace(self, **kwargs)

# Register so jit/vmap/scan can traverse the fields automatically.
# meta_fields=[] means ALL fields are dynamic (traced) — correct for arrays.
jax.tree_util.register_dataclass(
    EnvState,
    data_fields=[
        "pos", "vel", "base_pos", "target_pos",
        "coverage_grid", "step", "key",
        "active", "target_known", "collides", "last_cov_delta",
        "box_width", "box_height", "base_target_known", "chain_held_steps",
        "is_conn_base", "is_conn_target", "directly_sees_target",
        "finder_path_cells", "finder_path_lens", "finder_path_active",
        "target_known_path_cells", "target_known_path_lens", "target_known_path_valid",
        "finders_path", "finders_path_len", "finders_path_valid",
        "finders_path_index_grid", "adj_matrix",
    ],
    meta_fields=[],
)


# ---------------------------------------------------------------------------
# Shape / dtype assertions — call freely during dev, stripped in production
# via chex.disable_asserts().
# ---------------------------------------------------------------------------

def assert_env_state(state: EnvState, N: int, GW: int, GH: int) -> None:
    """Verify all state fields have the expected shapes and dtypes."""
    chex.assert_shape(state.pos,           (N, 2))
    chex.assert_shape(state.vel,           (N, 2))
    chex.assert_shape(state.base_pos,      (2,))
    chex.assert_shape(state.target_pos, (2,))
    chex.assert_shape(state.coverage_grid, (GW, GH))
    chex.assert_shape(state.step,          ())
    chex.assert_shape(state.key,           (2,))
    chex.assert_shape(state.active,        (N,))
    chex.assert_shape(state.target_known,  (N,))
    chex.assert_shape(state.collides,      (N,))
    chex.assert_shape(state.last_cov_delta, (N,))
    chex.assert_shape(state.base_target_known, ())
    chex.assert_shape(state.chain_held_steps, ())
    chex.assert_shape(state.is_conn_base, (N,))
    chex.assert_shape(state.is_conn_target, (N,))
    chex.assert_shape(state.directly_sees_target, (N,))
    if state.adj_matrix.size:
        chex.assert_shape(state.adj_matrix, (N + 1, N + 1))

    chex.assert_type(state.pos,           jnp.float32)
    chex.assert_type(state.vel,           jnp.float32)
    chex.assert_type(state.base_pos,      jnp.float32)
    chex.assert_type(state.target_pos,    jnp.float32)
    chex.assert_type(state.coverage_grid, jnp.bool_)
    chex.assert_type(state.step,          jnp.int32)
    chex.assert_type(state.active,        jnp.bool_)
    chex.assert_type(state.target_known,  jnp.bool_)
    chex.assert_type(state.collides,      jnp.bool_)
    chex.assert_type(state.base_target_known, jnp.bool_)
    chex.assert_type(state.chain_held_steps, jnp.int32)
    chex.assert_type(state.is_conn_base, jnp.bool_)
    chex.assert_type(state.is_conn_target, jnp.bool_)
    chex.assert_type(state.directly_sees_target, jnp.bool_)
    if state.adj_matrix.size:
        chex.assert_type(state.adj_matrix, jnp.bool_)
