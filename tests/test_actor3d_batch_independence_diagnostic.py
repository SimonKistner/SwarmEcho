"""Manual, layer-by-layer batch-independence diagnostic for the 3D actor.

Run in WSL only:

    SWARMECHO_RUN_ACTOR_BATCH_DIAGNOSTIC=1 \
      uv run pytest -q -s --tb=short tests/test_actor3d_batch_independence_diagnostic.py

The test repeats *identical* actor inputs into every batch lane and compares
lane zero with direct single-lane inference.  It intentionally does not run a
full environment evaluation.
"""

from __future__ import annotations

import os
from pathlib import Path

# Must be set before pytest imports JAX: compiler dumps are not diagnostic data.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_CPP_MIN_VLOG_LEVEL", "0")
os.environ.setdefault("GLOG_minloglevel", "3")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from flax import nnx

from swarmecho.core.config import load_level
from swarmecho.env.environment import make_env_fns
from swarmecho.training.checkpoints import restore_model_checkpoint
from swarmecho.training.train import build_model


_DEFAULT_CHECKPOINT = Path("outputs/M00_tall_v1/checkpoints/ckpt_000351")


def _checkpoint() -> Path:
    return Path(
        os.environ.get("SWARMECHO_DIAGNOSTIC_CHECKPOINT", _DEFAULT_CHECKPOINT)
    )


def _checkpoint_level(checkpoint: Path) -> str:
    data = yaml.safe_load(
        (checkpoint.parents[1] / "config.yaml").read_text(encoding="utf-8")
    )
    name = data.get("name") if isinstance(data, dict) else None
    assert isinstance(name, str) and name, "Checkpoint config has no level name."
    return name


def _actor_components(
    model,
    obs: jax.Array,
    hidden: jax.Array,
    signature: jax.Array,
    value: jax.Array,
    comm_mask: jax.Array,
    active: jax.Array,
    base_signature: jax.Array,
    base_value: jax.Array,
    base_memory_mask: jax.Array,
) -> dict[str, jax.Array]:
    """Expose the exact TarMAC actor operations without changing model code."""
    actor = model.actor
    components: dict[str, jax.Array] = {}

    def expand_mlp(prefix: str, mlp, inputs: jax.Array) -> jax.Array:
        """Mirror MLP while retaining its dense, norm, and activation outputs."""
        output = inputs
        for index, (linear, norm) in enumerate(zip(mlp.layers, mlp.norms)):
            dense = linear(output)
            normalized = norm(dense)
            output = jnp.tanh(normalized)
            components[f"{prefix}.dense_{index}"] = dense
            components[f"{prefix}.layer_norm_{index}"] = normalized
            components[f"{prefix}.tanh_{index}"] = output
        output = mlp.out_linear(output)
        components[f"{prefix}.output"] = output
        return output

    encoded = expand_mlp("observation_encoder", actor.encoder, obs)
    agents = hidden.shape[-2]
    active_pair = active[:, None] & active[None, :]
    eye = jnp.eye(agents, dtype=jnp.bool_)
    external_mask = comm_mask & active_pair & ~eye
    has_external_sender = jnp.any(external_mask, axis=-1)
    if actor.tarmac_include_self:
        share_mask = external_mask | (
            eye & active[:, None] & has_external_sender[:, None]
        )
    else:
        share_mask = external_mask
    sender_signature = jnp.concatenate(
        [signature, base_signature[None, :]], axis=-2
    )
    sender_value = jnp.concatenate([value, base_value[None, :]], axis=-2)
    attention_mask = jnp.concatenate(
        [share_mask, base_memory_mask[:, None] & active[:, None]], axis=-1
    )
    query = actor.tarmac_query(hidden)
    raw_logits = jnp.einsum("ns,ms->nm", query, sender_signature) / jnp.sqrt(
        jnp.asarray(query.shape[-1], query.dtype)
    )
    has_any = jnp.any(attention_mask, axis=-1, keepdims=True)
    first_sender = jnp.arange(attention_mask.shape[-1]) == 0
    safe_mask = attention_mask | ((~has_any) & first_sender[None, :])
    masked_logits = jnp.where(
        safe_mask, raw_logits, jnp.finfo(raw_logits.dtype).min
    )
    attention_weights = jax.nn.softmax(masked_logits, axis=-1)
    aggregated_message = jnp.einsum(
        "nm,mv->nv", attention_weights, sender_value
    )
    context = jnp.where(has_any, aggregated_message, jnp.zeros_like(aggregated_message))
    # This is the model's own helper; checking it against the expanded pieces
    # catches masking or attention-axis errors independently of later layers.
    helper_context = actor._tarmac_context(
        hidden,
        signature,
        value,
        comm_mask,
        active,
        base_signature,
        base_value,
        base_memory_mask,
    )
    recurrent_input = jnp.concatenate([encoded, context], axis=-1)
    updated_hidden = actor.gru(hidden, recurrent_input)
    next_signature = actor.tarmac_signature(updated_hidden)
    next_value = actor.tarmac_value(updated_hidden)
    policy_features = expand_mlp("policy_trunk", actor.policy_trunk, updated_hidden)
    action_mean = actor.mu_head(policy_features)
    action_log_std = jnp.clip(actor.log_std_head(policy_features), -5.0, 2.0)
    (
        public_hidden,
        public_signature,
        public_value,
        public_action_mean,
        public_action_log_std,
    ) = actor.__call_team__(
        obs,
        hidden,
        signature,
        value,
        reset=jnp.zeros((agents,), dtype=jnp.bool_),
        comm_mask=comm_mask,
        active=active,
        base_signature=base_signature,
        base_value=base_value,
        base_memory_mask=base_memory_mask,
        deterministic=True,
    )
    components.update({
        "observation_encoder": encoded,
        "message_query": query,
        "attention_logits": raw_logits,
        "attention_softmax": attention_weights,
        "aggregated_message": aggregated_message,
        "helper_context": helper_context,
        "recurrent_input": recurrent_input,
        "recurrent_update": updated_hidden,
        "message_encoder_signature": next_signature,
        "message_encoder_value": next_value,
        "policy_features": policy_features,
        "action_head": action_mean,
        "action_log_std": action_log_std,
        "final_action_mean": public_action_mean,
        "public_recurrent_update": public_hidden,
        "public_message_signature": public_signature,
        "public_message_value": public_value,
        "public_action_log_std": public_action_log_std,
    })
    return components


@nnx.jit
def _run_single(model, inputs):
    return _actor_components(model, *inputs)


@nnx.jit
def _run_repeated_batch(model, batched_inputs):
    return jax.vmap(lambda *inputs: _actor_components(model, *inputs))(
        *batched_inputs
    )


def _repeat_inputs(inputs: tuple[jax.Array, ...], batch_size: int):
    return tuple(
        jnp.broadcast_to(item, (batch_size, *item.shape)) for item in inputs
    )


def _actual_initial_inputs(model, level) -> tuple[jax.Array, ...]:
    reset, _, observations, _ = make_env_fns(level.building, level.env)
    state = reset(jax.random.PRNGKey(level.training.seed + 10_000))
    agents = level.env.num_agents
    delta = state.pos[:, None, :] - state.pos[None, :, :]
    comm_mask = (
        (jnp.linalg.norm(delta, axis=-1) <= level.env.comm_radius)
        & state.active[:, None]
        & state.active[None, :]
        & ~jnp.eye(agents, dtype=jnp.bool_)
    )
    in_base_range = (
        jnp.linalg.norm(state.pos - state.base_pos[None, :], axis=-1)
        <= level.env.comm_radius_base
    ) & state.active
    return (
        observations(state),
        model.initial_actor_hidden(()),
        model.initial_actor_signature(()),
        model.initial_actor_value(()),
        comm_mask,
        state.active,
        jnp.zeros((model.tarmac_sig_dim,), dtype=jnp.float32),
        jnp.zeros((model.tarmac_val_dim,), dtype=jnp.float32),
        jnp.zeros((agents,), dtype=jnp.bool_) & in_base_range,
    )


def _nonzero_communication_inputs(model, level) -> tuple[jax.Array, ...]:
    """Exercise every TarMAC branch with identical non-zero recurrent state."""
    agents = level.env.num_agents
    key = jax.random.PRNGKey(20260806)
    keys = jax.random.split(key, 6)
    obs_dim = model.obs_dim
    return (
        jax.random.normal(keys[0], (agents, obs_dim)),
        jax.random.normal(keys[1], (agents, model.hidden_dim)),
        jax.random.normal(keys[2], (agents, model.tarmac_sig_dim)),
        jax.random.normal(keys[3], (agents, model.tarmac_val_dim)),
        ~jnp.eye(agents, dtype=jnp.bool_),
        jnp.ones((agents,), dtype=jnp.bool_),
        jax.random.normal(keys[4], (model.tarmac_sig_dim,)),
        jax.random.normal(keys[5], (model.tarmac_val_dim,)),
        jnp.ones((agents,), dtype=jnp.bool_),
    )


def _report(label: str, single, repeated) -> list[str]:
    failures: list[str] = []
    for component in single:
        expected = np.asarray(single[component])
        actual = np.asarray(repeated[component][0])
        max_abs = float(np.max(np.abs(expected - actual), initial=0.0))
        print(f"[ACTOR-DIAG] {label:22} {component:22} max_abs={max_abs:.9g}")
        if not np.allclose(expected, actual, rtol=1e-6, atol=1e-6):
            failures.append(f"{component}={max_abs:.9g}")
    return failures


def test_identical_actor_inputs_are_batch_independent():
    if os.environ.get("SWARMECHO_RUN_ACTOR_BATCH_DIAGNOSTIC") != "1":
        pytest.skip("Manual CUDA diagnostic; set SWARMECHO_RUN_ACTOR_BATCH_DIAGNOSTIC=1.")

    checkpoint = _checkpoint()
    assert checkpoint.is_dir(), f"Checkpoint not found: {checkpoint}"
    level = load_level(_checkpoint_level(checkpoint))
    model = build_model(level)
    restore_model_checkpoint(model, checkpoint)
    assert model.actor.memory_comm_enabled, "This diagnostic expects the TarMAC actor."
    batch_size = int(
        os.environ.get(
            "SWARMECHO_ACTOR_DIAGNOSTIC_BATCH", level.evaluation.eval_parallel_envs
        )
    )
    assert batch_size > 0
    print(f"[ACTOR-DIAG] repeated batch size={batch_size}")

    all_failures: list[str] = []
    for label, inputs in (
        ("actual initial state", _actual_initial_inputs(model, level)),
        ("nonzero communication", _nonzero_communication_inputs(model, level)),
    ):
        single = jax.device_get(_run_single(model, inputs))
        repeated = jax.device_get(_run_repeated_batch(model, _repeat_inputs(inputs, batch_size)))
        all_failures.extend(f"{label}/{failure}" for failure in _report(label, single, repeated))

    assert not all_failures, (
        "Repeated identical batch lanes changed actor components: "
        + ", ".join(all_failures)
    )
