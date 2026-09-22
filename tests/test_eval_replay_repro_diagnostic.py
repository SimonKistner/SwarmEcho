"""Manual diagnostic for a parallel-evaluation versus selective-replay mismatch.

This is deliberately opt-in: it restores a real CUDA checkpoint and runs the
same 4,000-lane evaluation as ``swarmecho-evaluate ... mode=parallel``.
It then runs the same selected target through the selective replay path and
compares the two rollouts timestep by timestep.

Run in WSL only:

    SWARMECHO_RUN_REPLAY_DIAGNOSTIC=1 \
      uv run pytest -s tests/test_eval_replay_repro_diagnostic.py

Set ``SWARMECHO_DIAGNOSTIC_CHECKPOINT`` to diagnose another checkpoint.
"""

from __future__ import annotations

import os
from pathlib import Path

# Suppress XLA's verbose compiler diagnostics before pytest imports JAX.  The
# test prints its own compact [DIAG] result lines; the compiler fusion dumps do
# not help identify a replay mismatch.
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
from swarmecho.training.artifacts import evaluation_stage, load_eval_info_csv
from swarmecho.training.checkpoints import restore_model_checkpoint
from swarmecho.training.evaluate import run_parallel_evaluation
from swarmecho.training.train import build_model, evaluate_model


_DEFAULT_CHECKPOINT = Path("outputs/M00_tall_v1/checkpoints/ckpt_000351")
_FIELDS = (
    "observation",
    "action_mean",
    "position",
    "velocity",
    "target_known",
    "connected_to_base",
    "connected_to_target",
    "fully_connected",
    "chain_held_steps",
    "base_target_known",
    "success",
    "done",
    "actor_hidden",
    "actor_signature",
    "actor_value",
    "base_memory_valid",
    "base_signature",
    "base_value",
)


def _checkpoint() -> Path:
    return Path(os.environ.get("SWARMECHO_DIAGNOSTIC_CHECKPOINT", _DEFAULT_CHECKPOINT))


def _checkpoint_level(checkpoint: Path) -> str:
    config_path = checkpoint.parents[1] / "config.yaml"
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    name = data.get("name") if isinstance(data, dict) else None
    if not isinstance(name, str) or not name:
        raise AssertionError(f"Checkpoint config has no level name: {config_path}")
    return name


def _success_zero_index(records: dict[str, np.ndarray]) -> int:
    """Return the original parallel lane selected by result=success offset=0."""
    successful = np.flatnonzero(records["stages"] == "chain_success")
    assert len(successful), "Parallel evaluation did not produce a successful lane."
    ranking = successful[
        np.argsort(-records["final_chain_length"][successful], kind="stable")
    ]
    return int(ranking[0])


def _snapshot(
    *,
    observation: jax.Array,
    action_mean: jax.Array,
    state,
    actor_hidden: jax.Array,
    actor_signature: jax.Array,
    actor_value: jax.Array,
    base_memory_valid: jax.Array,
    base_signature: jax.Array,
    base_value: jax.Array,
) -> dict[str, jax.Array]:
    """Keep all information needed to identify the *first* divergence."""
    return {
        "observation": observation,
        "action_mean": action_mean,
        "position": state.pos,
        "velocity": state.vel,
        "target_known": state.target_known,
        "connected_to_base": state.is_conn_base,
        "connected_to_target": state.is_conn_target,
        "fully_connected": state.fully_connected,
        "chain_held_steps": state.chain_held_steps,
        "base_target_known": state.base_target_known,
        "success": state.success,
        "done": state.done,
        "actor_hidden": actor_hidden,
        "actor_signature": actor_signature,
        "actor_value": actor_value,
        "base_memory_valid": base_memory_valid,
        "base_signature": base_signature,
        "base_value": base_value,
    }


def _build_tracer(model, level):
    """Return the production actor/environment transition with trace outputs.

    The transition is intentionally line-for-line equivalent to the actor,
    TarMAC/base-memory, and physics portions of the parallel evaluator and
    replay collector.  Unlike either production path, it exposes inputs and
    outputs at every step for comparison.
    """
    cfg = level.env
    reset, step, observations, _ = make_env_fns(level.building, cfg)
    num_agents = cfg.num_agents
    cadence = int(model.memory_comm_every_k_steps)

    def advance(model, carry, step_index):
        (
            state,
            actor_hidden,
            actor_signature,
            actor_value,
            base_memory_valid,
            base_signature,
            base_value,
        ) = carry
        delta = state.pos[:, None, :] - state.pos[None, :, :]
        comm_mask = (
            (jnp.linalg.norm(delta, axis=-1) <= cfg.comm_radius)
            & state.active[:, None]
            & state.active[None, :]
            & ~jnp.eye(num_agents, dtype=jnp.bool_)
        )
        in_base_range = (
            jnp.linalg.norm(state.pos - state.base_pos[None, :], axis=-1)
            <= cfg.comm_radius_base
        ) & state.active
        share_now = (step_index % cadence) == 0
        observation = observations(state)
        (
            next_actor_hidden,
            emitted_signature,
            emitted_value,
            means,
            _,
        ) = model.actor.__call_team__(
            observation,
            actor_hidden,
            actor_signature,
            actor_value,
            reset=jnp.zeros(num_agents, dtype=jnp.bool_),
            comm_mask=comm_mask & share_now,
            active=state.active,
            base_signature=base_signature,
            base_value=base_value,
            base_memory_mask=(
                base_memory_valid
                & in_base_range
                & ~state.target_known
                & share_now
            ),
            deterministic=True,
        )
        next_state = step(state, jnp.tanh(means))
        reporters = state.target_known & in_base_range
        reporter_index = jnp.argmax(reporters.astype(jnp.int32))
        should_store = ~base_memory_valid & jnp.any(reporters)
        next_base_signature = jnp.where(
            should_store, emitted_signature[reporter_index], base_signature
        )
        next_base_value = jnp.where(
            should_store, emitted_value[reporter_index], base_value
        )
        next_base_memory_valid = base_memory_valid | should_store
        next_base_memory_valid = jnp.where(
            next_state.done, False, next_base_memory_valid
        )
        next_base_signature = jnp.where(
            next_state.done, 0.0, next_base_signature
        )
        next_base_value = jnp.where(next_state.done, 0.0, next_base_value)
        next_carry = (
            next_state,
            next_actor_hidden,
            emitted_signature,
            emitted_value,
            next_base_memory_valid,
            next_base_signature,
            next_base_value,
        )
        return next_carry, _snapshot(
            observation=observation,
            action_mean=means,
            state=next_state,
            actor_hidden=next_actor_hidden,
            actor_signature=emitted_signature,
            actor_value=emitted_value,
            base_memory_valid=next_base_memory_valid,
            base_signature=next_base_signature,
            base_value=next_base_value,
        )

    return reset, advance


def _initial_carry(model, state):
    return (
        state,
        model.initial_actor_hidden(()),
        model.initial_actor_signature(()),
        model.initial_actor_value(()),
        jnp.bool_(False),
        jnp.zeros((model.tarmac_sig_dim,), dtype=jnp.float32),
        jnp.zeros((model.tarmac_val_dim,), dtype=jnp.float32),
    )


def _trace_single(model, level, key, *, target: np.ndarray | None):
    reset, advance = _build_tracer(model, level)
    state = reset(key) if target is None else reset(key, target_pos=jnp.asarray(target))
    initial = _initial_carry(model, state)

    @nnx.jit
    def run(model, carry):
        return jax.lax.scan(
            lambda current, step_index: advance(model, current, step_index),
            carry,
            jnp.arange(level.env.max_steps),
        )

    _, trace = jax.device_get(run(model, initial))
    return jax.device_get(state), trace


def _trace_parallel_lane(model, level, *, lane: int):
    """Trace one lane while retaining 4,000-way batched actor computations."""
    reset, advance = _build_tracer(model, level)
    parallel_envs = level.evaluation.eval_parallel_envs
    root_key = jax.random.PRNGKey(level.training.seed + 10_000)
    keys = jax.random.split(root_key, parallel_envs)
    states = jax.vmap(reset)(keys)
    initial = (
        states,
        jnp.broadcast_to(
            model.initial_actor_hidden(()),
            (parallel_envs, model.num_agents, model.hidden_dim),
        ),
        jnp.broadcast_to(
            model.initial_actor_signature(()),
            (parallel_envs, model.num_agents, model.tarmac_sig_dim),
        ),
        jnp.broadcast_to(
            model.initial_actor_value(()),
            (parallel_envs, model.num_agents, model.tarmac_val_dim),
        ),
        jnp.zeros((parallel_envs,), dtype=jnp.bool_),
        jnp.zeros((parallel_envs, model.tarmac_sig_dim), dtype=jnp.float32),
        jnp.zeros((parallel_envs, model.tarmac_val_dim), dtype=jnp.float32),
    )
    # Keep the per-lane reset state, not merely the CSV-rounded target.
    selected_initial = jax.tree_util.tree_map(lambda value: value[lane], states)

    @nnx.jit
    def run(model, carry):
        def scan_step(current, step_index):
            next_carry, snapshots = jax.vmap(
                lambda *one_carry: advance(model, one_carry, step_index)
            )(*current)
            selected = jax.tree_util.tree_map(lambda value: value[lane], snapshots)
            return next_carry, selected

        return jax.lax.scan(
            scan_step, carry, jnp.arange(level.env.max_steps, dtype=jnp.int32)
        )

    _, trace = jax.device_get(run(model, initial))
    return jax.device_get(selected_initial), trace


def _first_difference(
    reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]
) -> tuple[int, str, float] | None:
    """Return the first step/field that differs, with its largest error."""
    for step in range(len(reference["action_mean"])):
        for field in _FIELDS:
            expected = np.asarray(reference[field][step])
            actual = np.asarray(candidate[field][step])
            if expected.dtype.kind in "bui" or actual.dtype.kind in "bui":
                equal = np.array_equal(expected, actual)
                error = 0.0 if equal else float("inf")
            else:
                error = float(np.max(np.abs(expected - actual), initial=0.0))
                equal = np.allclose(expected, actual, rtol=0.0, atol=1e-6)
            if not equal:
                return step + 1, field, error
    return None


def _print_outcome(label: str, trace: dict[str, np.ndarray]) -> None:
    success_steps = np.flatnonzero(np.asarray(trace["success"], dtype=bool))
    delivered_steps = np.flatnonzero(
        np.asarray(trace["base_target_known"], dtype=bool)
    )
    print(
        f"[DIAG] {label}: success_steps="
        f"{success_steps[:3].tolist() or 'none'}, delivered_first="
        f"{None if not len(delivered_steps) else int(delivered_steps[0] + 1)}, "
        f"final_held={int(trace['chain_held_steps'][-1])}",
        flush=True,
    )


def _first_production_state_difference(
    trace: dict[str, np.ndarray], states: list,
) -> tuple[int, str, float] | None:
    """Compare the traced NNX path against ``evaluate_model`` itself."""
    fields = (
        ("position", "pos"),
        ("velocity", "vel"),
        ("target_known", "target_known"),
        ("connected_to_base", "is_conn_base"),
        ("connected_to_target", "is_conn_target"),
        ("fully_connected", "fully_connected"),
        ("chain_held_steps", "chain_held_steps"),
        ("base_target_known", "base_target_known"),
        ("success", "success"),
        ("done", "done"),
    )
    for step, state in enumerate(states[1:], start=1):
        for trace_field, state_field in fields:
            expected = np.asarray(trace[trace_field][step - 1])
            actual = np.asarray(getattr(state, state_field))
            if expected.dtype.kind in "bui" or actual.dtype.kind in "bui":
                equal, error = np.array_equal(expected, actual), 0.0
            else:
                error = float(np.max(np.abs(expected - actual), initial=0.0))
                equal = np.allclose(expected, actual, rtol=0.0, atol=1e-6)
            if not equal:
                return step, trace_field, error
    return None


def test_parallel_success_replays_identically_step_by_step():
    """Diagnose the exact divergence behind a CSV-success/replay-fail report."""
    if os.environ.get("SWARMECHO_RUN_REPLAY_DIAGNOSTIC") != "1":
        pytest.skip(
            "Manual CUDA diagnostic. Set SWARMECHO_RUN_REPLAY_DIAGNOSTIC=1 "
            "and run this file explicitly in WSL."
        )

    checkpoint = _checkpoint()
    assert checkpoint.is_dir(), f"Checkpoint not found: {checkpoint}"
    level = load_level(_checkpoint_level(checkpoint))
    # Make the test's first operation identical to the user's parallel command.
    model = build_model(level)
    restore_model_checkpoint(model, checkpoint)
    csv_path = run_parallel_evaluation(model, level, checkpoint, checkpoint.parents[1])
    records = load_eval_info_csv(csv_path)
    lane = _success_zero_index(records)
    target = records["positions"][lane]
    print(
        f"[DIAG] CSV SUCCESS_0: lane={lane}; target={target.tolist()}; "
        f"chain_length={float(records['final_chain_length'][lane]):.6f}",
        flush=True,
    )

    # 1. The alternative scan(vmap(...)) execution.  This is deliberately
    # traced, unlike the production vmap(scan(...)) evaluator.  If it does
    # not reproduce the CSV outcome, that is itself material evidence: JAX
    # transformation order changes this policy's deterministic trajectory.
    parallel_initial, parallel_trace = _trace_parallel_lane(model, level, lane=lane)
    _print_outcome("scan(vmap) lane", parallel_trace)
    traced_batch_succeeds = bool(np.any(parallel_trace["success"]))
    print(
        f"[DIAG] scan(vmap) reproduces CSV success: {traced_batch_succeeds}",
        flush=True,
    )

    root_key = jax.random.PRNGKey(level.training.seed + 10_000)
    lane_key = jax.random.split(root_key, level.evaluation.eval_parallel_envs)[lane]

    # 2. Same reset key and sampled target, but one environment at a time.
    single_initial, single_trace = _trace_single(model, level, lane_key, target=None)
    _print_outcome("single lane key", single_trace)

    # 3. The exact selective-replay setup: root key plus CSV-selected target.
    replay_initial, replay_trace = _trace_single(model, level, root_key, target=target)
    _print_outcome("single CSV target", replay_trace)

    for label, initial in (
        ("parallel versus single-key reset", single_initial),
        ("parallel versus forced-target reset", replay_initial),
    ):
        assert np.array_equal(parallel_initial.pos, initial.pos), (
            f"{label}: drone positions differ at reset"
        )
        assert np.array_equal(parallel_initial.base_pos, initial.base_pos), (
            f"{label}: base position differs at reset"
        )
        assert np.array_equal(parallel_initial.target_pos, initial.target_pos), (
            f"{label}: target position differs at reset"
        )

    single_difference = _first_difference(parallel_trace, single_trace)
    replay_difference = _first_difference(parallel_trace, replay_trace)
    print(f"[DIAG] first batch-vs-single-key difference: {single_difference}", flush=True)
    print(f"[DIAG] first batch-vs-forced-target difference: {replay_difference}", flush=True)

    # Finally call the production selective replay function, rather than only
    # its equivalent traced transition, and print exactly the stage the CLI
    # reports before making the diagnostic fail on any discrepancy.
    replay_states, _ = evaluate_model(model, level, target_position=target)
    final = replay_states[-1]
    final_stage = evaluation_stage(
        success=bool(np.asarray(final.success)),
        delivered=bool(np.asarray(final.base_target_known)),
        visually_found=bool(np.any(np.asarray(final.target_known))),
    )
    print(
        f"[DIAG] production selective replay: steps={len(replay_states) - 1}; "
        f"final_stage={final_stage}; final_held={int(final.chain_held_steps)}",
        flush=True,
    )
    production_difference = _first_production_state_difference(
        replay_trace, replay_states
    )
    print(
        f"[DIAG] first traced-vs-production-replay difference: "
        f"{production_difference}",
        flush=True,
    )

    assert (
        traced_batch_succeeds
        and single_difference is None
        and replay_difference is None
        and production_difference is None
    ), (
        "The production CSV (vmap(scan)) and the traced/replay execution "
        "(scan(vmap) or one lane) do not agree. Read the [DIAG] lines above: "
        "they distinguish a transformation-order mismatch from a reset-key "
        "or CSV-coordinate mismatch."
    )
