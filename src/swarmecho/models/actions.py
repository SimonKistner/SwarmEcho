"""Mixed movement/discrete actions: never squash or perturb the vote channel."""
import jax
import jax.numpy as jnp


def physical_actions(actions):
    return actions.at[..., :3].set(jnp.tanh(actions[..., :3]))


def deterministic_actions(actor, hidden, means):
    if actor.vote_mode is None:
        return means
    logits = actor.vote_logits(hidden)
    vote = (logits[..., 0] >= 0 if actor.vote_mode == "hold" else jnp.argmax(logits, axis=-1))
    return jnp.concatenate([means, vote.astype(means.dtype)[..., None]], axis=-1)


def vote_statistics(logits, action, mode):
    if mode == "hold":
        z = logits[..., 0]
        lp = jnp.where(action >= 0.5, jax.nn.log_sigmoid(z), jax.nn.log_sigmoid(-z))
        p = jax.nn.sigmoid(z)
        entropy = jax.nn.softplus(z) - p * z
        return lp, entropy
    log_p = jax.nn.log_softmax(logits, axis=-1)
    action = jnp.clip(action.astype(jnp.int32), 0, 2)
    lp = jnp.take_along_axis(log_p, action[..., None], axis=-1)[..., 0]
    return lp, -jnp.sum(jnp.exp(log_p) * log_p, axis=-1)


def sample_vote(logits, key, mode, deterministic=False):
    if mode == "hold":
        random = jax.random.bernoulli(key, jax.nn.sigmoid(logits[..., 0]))
        action = jnp.where(deterministic, logits[..., 0] >= 0, random)
    else:
        random = jax.random.categorical(key, logits, axis=-1)
        action = jnp.where(deterministic, jnp.argmax(logits, axis=-1), random)
    action = action.astype(jnp.float32)
    lp, ent = vote_statistics(logits, action, mode)
    return action, lp, ent
