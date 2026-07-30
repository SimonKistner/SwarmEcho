"""Nested JAX PyTrees for the SwarmEcho environment state.

The top-level state is deliberately an ownership container. General physical
state does not carry 2D exploration grids or relay-task path bookkeeping.
Each nested dataclass contains arrays only and can pass through JIT, VMAP, and
SCAN transformations.
"""

from __future__ import annotations

import dataclasses

import chex
import jax
import jax.numpy as jnp


@dataclasses.dataclass
class PhysicsState:
    """Kinematic world and episode-clock state."""

    pos: jax.Array
    vel: jax.Array
    base_pos: jax.Array
    target_pos: jax.Array
    step: jax.Array
    key: jax.Array
    active: jax.Array
    box_width: jax.Array
    box_height: jax.Array

    def replace(self, **kwargs) -> PhysicsState:
        return dataclasses.replace(self, **kwargs)


@dataclasses.dataclass
class CommunicationState:
    """Direct graph, graph reachability, and persistent target knowledge."""

    target_known: jax.Array
    base_target_known: jax.Array
    is_conn_base: jax.Array
    is_conn_target: jax.Array
    directly_sees_target: jax.Array
    adj_matrix: jax.Array

    def replace(self, **kwargs) -> CommunicationState:
        return dataclasses.replace(self, **kwargs)


@dataclasses.dataclass
class ExplorationState:
    """2D coverage scaffold kept outside the general physical state."""

    coverage_grid: jax.Array

    def replace(self, **kwargs) -> ExplorationState:
        return dataclasses.replace(self, **kwargs)


@dataclasses.dataclass
class RelayTaskState:
    """Relay objective progress and discrete 2D finder-path bookkeeping."""

    chain_held_steps: jax.Array
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

    def replace(self, **kwargs) -> RelayTaskState:
        return dataclasses.replace(self, **kwargs)


@dataclasses.dataclass
class StepDiagnostics:
    """Per-transition outputs consumed by rewards and optional diagnostics."""

    collides: jax.Array
    last_cov_delta: jax.Array

    def replace(self, **kwargs) -> StepDiagnostics:
        return dataclasses.replace(self, **kwargs)


@dataclasses.dataclass
class EnvState:
    """Immutable environment snapshot grouped by subsystem ownership."""

    physics: PhysicsState
    communication: CommunicationState
    exploration: ExplorationState
    relay: RelayTaskState
    diagnostics: StepDiagnostics

    def replace(self, **kwargs) -> EnvState:
        return dataclasses.replace(self, **kwargs)


def _register(cls: type, fields: list[str]) -> None:
    jax.tree_util.register_dataclass(cls, data_fields=fields, meta_fields=[])


_register(
    PhysicsState,
    [
        "pos",
        "vel",
        "base_pos",
        "target_pos",
        "step",
        "key",
        "active",
        "box_width",
        "box_height",
    ],
)
_register(
    CommunicationState,
    [
        "target_known",
        "base_target_known",
        "is_conn_base",
        "is_conn_target",
        "directly_sees_target",
        "adj_matrix",
    ],
)
_register(ExplorationState, ["coverage_grid"])
_register(
    RelayTaskState,
    [
        "chain_held_steps",
        "finder_path_cells",
        "finder_path_lens",
        "finder_path_active",
        "target_known_path_cells",
        "target_known_path_lens",
        "target_known_path_valid",
        "finders_path",
        "finders_path_len",
        "finders_path_valid",
        "finders_path_index_grid",
    ],
)
_register(StepDiagnostics, ["collides", "last_cov_delta"])
_register(
    EnvState,
    ["physics", "communication", "exploration", "relay", "diagnostics"],
)


def assert_env_state(state: EnvState, N: int, GW: int, GH: int) -> None:
    """Verify nested state shapes and dtypes."""
    physics = state.physics
    communication = state.communication
    exploration = state.exploration
    diagnostics = state.diagnostics

    chex.assert_shape(physics.pos, (N, 2))
    chex.assert_shape(physics.vel, (N, 2))
    chex.assert_shape(physics.base_pos, (2,))
    chex.assert_shape(physics.target_pos, (2,))
    chex.assert_shape(physics.step, ())
    chex.assert_shape(physics.key, (2,))
    chex.assert_shape(physics.active, (N,))
    chex.assert_shape(physics.box_width, ())
    chex.assert_shape(physics.box_height, ())

    chex.assert_shape(communication.target_known, (N,))
    chex.assert_shape(communication.base_target_known, ())
    chex.assert_shape(communication.is_conn_base, (N,))
    chex.assert_shape(communication.is_conn_target, (N,))
    chex.assert_shape(communication.directly_sees_target, (N,))
    if communication.adj_matrix.size:
        chex.assert_shape(communication.adj_matrix, (N + 1, N + 1))

    chex.assert_shape(exploration.coverage_grid, (GW, GH))
    chex.assert_shape(diagnostics.collides, (N,))
    chex.assert_shape(diagnostics.last_cov_delta, (N,))
    chex.assert_shape(state.relay.chain_held_steps, ())

    chex.assert_type(physics.pos, jnp.float32)
    chex.assert_type(physics.vel, jnp.float32)
    chex.assert_type(physics.base_pos, jnp.float32)
    chex.assert_type(physics.target_pos, jnp.float32)
    chex.assert_type(physics.step, jnp.int32)
    chex.assert_type(physics.active, jnp.bool_)
    chex.assert_type(physics.box_width, jnp.float32)
    chex.assert_type(physics.box_height, jnp.float32)

    chex.assert_type(communication.target_known, jnp.bool_)
    chex.assert_type(communication.base_target_known, jnp.bool_)
    chex.assert_type(communication.is_conn_base, jnp.bool_)
    chex.assert_type(communication.is_conn_target, jnp.bool_)
    chex.assert_type(communication.directly_sees_target, jnp.bool_)
    if communication.adj_matrix.size:
        chex.assert_type(communication.adj_matrix, jnp.bool_)

    chex.assert_type(exploration.coverage_grid, jnp.bool_)
    chex.assert_type(diagnostics.collides, jnp.bool_)
    chex.assert_type(diagnostics.last_cov_delta, jnp.int32)
    chex.assert_type(state.relay.chain_held_steps, jnp.int32)
