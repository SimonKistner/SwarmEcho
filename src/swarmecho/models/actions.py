"""Map pre-squash movement commands to the environment action space."""
import jax.numpy as jnp


def physical_actions(actions):
    return jnp.tanh(actions)
