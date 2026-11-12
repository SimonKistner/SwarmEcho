"""First target-information acquisitions, counted once per agent lifetime."""

import jax.numpy as jnp


def new_information_counts(previous_known, known, active, sees, from_base):
    """Attribute recipients using the actual communication event: sight > base > peer.

    Use the source before sender reward-credit filtering: a peer's message still
    informed its recipient even if that sender decommissions during this step.
    The caller supplies final known/active masks, excluding newly spawned slots.
    """
    newly_informed = active & known & ~previous_known
    base = newly_informed & ~sees & from_base
    peer = newly_informed & ~sees & ~from_base
    return jnp.stack((jnp.sum(newly_informed), jnp.sum(peer), jnp.sum(base))).astype(jnp.int32)


def information_counts(previous, current):
    """Transition counts plus reset-time sightings on the first transition only.

    Slot reuse counts as a new life because spawning clears target_known.
    """
    initial_sightings = (previous.step == 0) & previous.active & previous.target_known
    return current.new_information_counts.at[0].add(jnp.sum(initial_sightings))
