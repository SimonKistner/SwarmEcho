"""
swarmecho/models/recurrent.py
=============================
Small recurrent building blocks shared by actor and critic modules.

The GRU cell is intentionally local and NNX-native so recurrent MAPPO does
not depend on Linen scanned modules or hidden framework wrappers.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


class GRUCell(nnx.Module):
    """
    Minimal GRU cell supporting arbitrary leading batch dimensions.

    Input
    -----
    h : (..., hidden_dim)
        Previous recurrent state.
    x : (..., in_features)
        Current encoded observation.

    Output
    ------
    h_next : (..., hidden_dim)
        Updated recurrent state.
    """

    def __init__(self, in_features: int, hidden_dim: int, rngs: nnx.Rngs) -> None:
        self.hidden_dim = hidden_dim
        self.gates = nnx.Linear(in_features + hidden_dim, 2 * hidden_dim, rngs=rngs)
        self.candidate = nnx.Linear(in_features + hidden_dim, hidden_dim, rngs=rngs)

    def __call__(self, h: jax.Array, x: jax.Array) -> jax.Array:
        hx = jnp.concatenate([x, h], axis=-1)
        z, r = jnp.split(jax.nn.sigmoid(self.gates(hx)), 2, axis=-1)
        candidate_in = jnp.concatenate([x, r * h], axis=-1)
        n = jnp.tanh(self.candidate(candidate_in))
        return (1.0 - z) * n + z * h
