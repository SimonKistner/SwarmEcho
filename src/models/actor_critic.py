"""
swarmecho/models/actor_critic.py
=================================
Flax NNX Actor-Critic for 

Architecture
------------
A shared MLP trunk feeds into:
  - An Actor head that outputs a diagonal Gaussian (mu, log_std)
  - A Critic head that outputs a scalar value estimate

The same network is shared across ALL agents (parameter sharing). At training
time, it is vmapped over B*N_agents observations; at inference it's called
on each agent's local obs.

Action space
------------
Continuous 2D force: actions ∈ [-max_force, max_force]^2
Sampled from Normal(mu, exp(log_std)), clipped at +/- max_force.
No tanh squashing (avoids log-prob correction complexity).

Usage
-----
    import jax, jax.numpy as jnp
    from flax import nnx
    from models.actor_critic import ActorCritic

    rngs = nnx.Rngs(0)
    model = ActorCritic(obs_dim=57, act_dim=2, hidden_dim=256, num_layers=3, rngs=rngs)

    obs = jnp.zeros((57,))
    action, log_prob, value, entropy = model.act(obs, rngs.default())
    pi_mu, pi_log_std, value = model(obs)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

# Minimum log_std for numerical stability (no collapse to Dirac delta)
LOG_STD_MIN = -5.0
LOG_STD_MAX =  2.0


# ---------------------------------------------------------------------------
# Building block: MLP with LayerNorm + Tanh
# ---------------------------------------------------------------------------

class MLP(nnx.Module):
    """
    Multi-layer perceptron with LayerNorm + Tanh activations.

    Parameters
    ----------
    in_features  : input dimension
    hidden_dim   : width of each hidden layer
    num_layers   : number of hidden layers (≥ 1)
    out_features : output dimension (no activation on final layer)
    rngs         : Flax NNX random-number generators
    """

    def __init__(
        self,
        in_features:  int,
        hidden_dim:   int,
        num_layers:   int,
        out_features: int,
        rngs:         nnx.Rngs,
    ) -> None:
        assert num_layers >= 1, "Need at least one hidden layer."

        layers = []
        norms  = []

        prev_dim = in_features
        for _ in range(num_layers):
            layers.append(nnx.Linear(prev_dim, hidden_dim, rngs=rngs))
            norms.append(nnx.LayerNorm(hidden_dim, rngs=rngs))
            prev_dim = hidden_dim

        self.layers     = nnx.List(layers)
        self.norms      = nnx.List(norms)
        self.out_linear = nnx.Linear(hidden_dim, out_features, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        for linear, norm in zip(self.layers, self.norms):
            x = jnp.tanh(norm(linear(x)))
        return self.out_linear(x)


# ---------------------------------------------------------------------------
# Actor-Critic
# ---------------------------------------------------------------------------

class ActorCritic(nnx.Module):
    """
    Shared-trunk Actor-Critic.

    Both actor and critic share the same trunk MLP for feature extraction,
    then split into two dedicated heads.

    Parameters
    ----------
    obs_dim    : observation dimension per agent
    act_dim    : action dimension (2 for SwarmEcho)
    hidden_dim : width of MLP hidden layers
    num_layers : depth of shared trunk
    rngs       : Flax NNX random generators
    """

    def __init__(
        self,
        obs_dim:    int,
        act_dim:    int,
        hidden_dim: int,
        num_layers: int,
        rngs:       nnx.Rngs,
    ) -> None:
        self.act_dim = act_dim

        # Shared trunk
        self.trunk = MLP(
            in_features  = obs_dim,
            hidden_dim   = hidden_dim,
            num_layers   = num_layers,
            out_features = hidden_dim,
            rngs         = rngs,
        )

        # Actor head: output mu + log_std for each action dimension
        self.actor_mu      = nnx.Linear(hidden_dim, act_dim, rngs=rngs)
        self.actor_log_std = nnx.Linear(hidden_dim, act_dim, rngs=rngs)

        # Critic head: scalar value estimate
        self.critic = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(
        self, obs: jax.Array,   # (..., obs_dim)
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """
        Forward pass — returns raw distribution parameters + value.

        Returns
        -------
        mu      : (..., act_dim)  — action distribution mean
        log_std : (..., act_dim)  — action distribution log std (clamped)
        value   : (...,)          — scalar value estimate
        """
        features = self.trunk(obs)
        mu       = self.actor_mu(features)
        log_std  = jnp.clip(self.actor_log_std(features), LOG_STD_MIN, LOG_STD_MAX)
        value    = self.critic(features).squeeze(-1)
        return mu, log_std, value

    def act(
        self,
        obs:          jax.Array,   # (obs_dim,) — single agent
        key:          jax.Array,   # PRNGKey for sampling
        max_force:    float = 50.0,
        deterministic: bool = False,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Sample an action for one agent.

        Returns
        -------
        action   : (act_dim,)  — sampled and clipped action
        log_prob : ()          — sum log-prob of the action
        value    : ()          — value estimate
        entropy  : ()          — distribution entropy (for logging)
        """
        mu, log_std, value = self(obs)
        std = jnp.exp(log_std)

        if deterministic:
            action = mu
        else:
            eps    = jax.random.normal(key, shape=mu.shape)
            action = mu + std * eps

        action = jnp.clip(action, -max_force, max_force)

        # Log-prob under the Gaussian (before clipping — clipping bias is tiny)
        log_prob = -0.5 * jnp.sum(
            ((action - mu) / (std + 1e-8)) ** 2
            + 2 * log_std
            + jnp.log(2 * jnp.pi),
        )

        # Per-dimension entropy: 0.5 * log(2πe σ²)
        entropy = jnp.sum(0.5 + 0.5 * jnp.log(2 * jnp.pi) + log_std)

        return action, log_prob, value, entropy

    def evaluate_actions(
        self,
        obs:     jax.Array,   # (..., obs_dim)
        actions: jax.Array,   # (..., act_dim)
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """
        Evaluate log-probs and value for ALREADY-SAMPLED actions.
        Used during PPO update to recompute log_prob under the updated policy.

        Returns
        -------
        log_prob : (...,)   — sum log-prob of `actions` under current policy
        value    : (...,)   — value estimate
        entropy  : (...,)   — distribution entropy
        """
        mu, log_std, value = self(obs)
        std = jnp.exp(log_std)

        log_prob = -0.5 * jnp.sum(
            ((actions - mu) / (std + 1e-8)) ** 2
            + 2 * log_std
            + jnp.log(2 * jnp.pi),
            axis=-1,
        )

        entropy = jnp.sum(
            0.5 + 0.5 * jnp.log(2 * jnp.pi) + log_std,
            axis=-1,
        )

        return log_prob, value, entropy


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from core.config import load_config, validate_config, compute_obs_dim, compute_action_dim

    print("── ActorCritic Self-Test ────────────────────────────────────")

    cfg     = load_config(cli_overrides=False)
    validate_config(cfg)
    obs_dim = compute_obs_dim(cfg)
    act_dim = compute_action_dim(cfg)
    N       = cfg.env.num_agents

    print(f"  obs_dim   : {obs_dim}")
    print(f"  act_dim   : {act_dim}")
    print(f"  hidden    : {cfg.network.hidden_dim} × {cfg.network.num_layers} layers")

    rngs  = nnx.Rngs(0)
    model = ActorCritic(
        obs_dim    = obs_dim,
        act_dim    = act_dim,
        hidden_dim = cfg.network.hidden_dim,
        num_layers = cfg.network.num_layers,
        rngs       = rngs,
    )

    # Count parameters
    graphdef, params = nnx.split(model)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Parameters: {n_params:,}")

    # Forward pass on a batch of N agents
    obs_batch = jnp.zeros((N, obs_dim))
    mu, log_std, value = model(obs_batch)
    assert mu.shape      == (N, act_dim), f"mu shape wrong: {mu.shape}"
    assert log_std.shape == (N, act_dim), f"log_std shape wrong: {log_std.shape}"
    assert value.shape   == (N,),         f"value shape wrong: {value.shape}"
    print(f"  Forward pass shapes: mu={mu.shape}, log_std={log_std.shape}, value={value.shape}  ✓")

    # act() — single agent
    key    = jax.random.PRNGKey(42)
    action, log_prob, val, ent = model.act(obs_batch[0], key, max_force=cfg.env.max_force)
    assert action.shape   == (act_dim,), f"action shape wrong: {action.shape}"
    assert log_prob.shape == (),         f"log_prob shape wrong: {log_prob.shape}"
    assert val.shape      == (),         f"val shape wrong: {val.shape}"
    print(f"  act() shapes: action={action.shape}, log_prob={log_prob.shape}  ✓")
    print(f"  action     : {action}")
    print(f"  log_prob   : {float(log_prob):.4f}")
    print(f"  value      : {float(val):.4f}")
    print(f"  entropy    : {float(ent):.4f}")

    # evaluate_actions() — batch
    keys      = jax.random.split(key, N)
    actions_b = jax.vmap(lambda o, k: model.act(o, k, max_force=cfg.env.max_force)[0])(obs_batch, keys)
    log_probs, values, entropies = model.evaluate_actions(obs_batch, actions_b)
    assert log_probs.shape == (N,), f"log_probs shape wrong: {log_probs.shape}"
    assert values.shape    == (N,), f"values shape wrong: {values.shape}"
    print(f"  evaluate_actions() shapes: log_probs={log_probs.shape}, values={values.shape}  ✓")

    # No NaN/Inf check
    for name, arr in [("mu", mu), ("log_std", log_std), ("value", value)]:
        assert not jnp.any(jnp.isnan(arr)), f"NaN in {name}!"
        assert not jnp.any(jnp.isinf(arr)), f"Inf in {name}!"

    print("\nActorCritic self-test passed ✓")


