"""Scientific contracts for the auto-launch curriculum comparison."""
from dataclasses import replace
from pathlib import Path
import importlib.util
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from swarmecho.core.config import AdaptSizeConfig, TargetCurriculumConfig, load_level
from swarmecho.env.adapt_size import initial_state, transition
from swarmecho.models.mappo import MAPPOModel
from swarmecho.training.ppo import replay
from swarmecho.training.target_curriculum import UniformTargetCurriculum


@pytest.mark.parametrize("mode,start,next_group", [("forward", 2, 3), ("reverse", 7, 6)])
def test_progression_uses_only_newest_validation_group(mode, start, next_group, tmp_path):
    controller = UniformTargetCurriculum(TargetCurriculumConfig(mode), 7)
    required = np.array([start] * 4 + [4] * 100)
    successes = np.array([True, True, False, False] + [True] * 100)
    metrics = controller.observe_curriculum("validation", required, successes, np.ones(104), 700)
    assert metrics["success_rate"] == .5
    assert controller.group == start
    controller.observe_curriculum("validation", [4], [True], [1], 700)
    assert controller.group == start
    controller.observe_curriculum("evaluation", [start] * 4, [True] * 4, [1] * 4, 700)
    assert controller.group == start
    controller.observe_curriculum("validation", [start] * 4, [True, True, True, False], [1] * 4, 700)
    assert controller.group == next_group
    controller.save(tmp_path)
    restored = UniformTargetCurriculum(TargetCurriculumConfig(mode), 7)
    restored.restore(tmp_path)
    assert restored.reset_kwargs() == controller.reset_kwargs()
    assert restored.group == next_group
    with pytest.raises(ValueError, match="settings changed"):
        UniformTargetCurriculum(TargetCurriculumConfig("reverse" if mode == "forward" else "forward"), 7).restore(tmp_path)


def test_retirement_is_permanent_and_pending_launches_prevent_empty_termination():
    cfg = AdaptSizeConfig(enabled=True, recruitment_enabled=False, initial_agents=3,
                          decommission_hold_steps=1)
    state = initial_state(3, 3, jnp.zeros(3), spawn_delay=5)
    active = jnp.array([True, False, False])
    spawned = []
    for tick in range(1, 13):
        state, active = transition(state, active, jnp.zeros((3, 3)), jnp.ones(3),
            jnp.ones((3, 3), bool), jnp.ones(3, bool), True, cfg, spawn_delay=5)
        spawned.append(np.asarray(state.spawned))
        assert bool(state.all_decommissioned) == (tick >= 11)
    np.testing.assert_array_equal(np.flatnonzero(np.any(spawned, axis=0)), [1, 2])
    assert int(state.calls) == 0
    assert int(state.decommissions) == 3
    assert np.all(np.asarray(state.ids) == -1)


def base_replay_fixture():
    model = MAPPOModel(3, 3, 2, 8, 1, 1, nnx.Rngs(7), actor_memory=True,
        critic_memory=True, memory_comm_enabled=True, tarmac_sig_dim=2, tarmac_val_dim=3)
    T, N = 4, 2
    obs = jax.random.normal(jax.random.PRNGKey(1), (T, 1, N, 3))
    masks = jnp.zeros((T, 1, N), bool).at[1:3, 0, 1].set(True)
    resets = jnp.zeros((T, 1, N), bool).at[3].set(True)
    h = jnp.zeros((N, 8)); sig = jnp.zeros((N, 2)); val = jnp.zeros((N, 3))
    bs = jnp.zeros(2); bv = jnp.zeros(3)
    signatures, values = [], []
    for t in range(T):
        signatures.append(bs); values.append(bv)
        h, sig, val, _, _ = model.actor.__call_team__(obs[t, 0], h, sig, val,
            reset=resets[t, 0], comm_mask=jnp.zeros((N, N), bool), active=jnp.ones(N, bool),
            base_signature=bs, base_value=bv, base_memory_mask=masks[t, 0])
        if t == 0:
            bs, bv = sig[0], val[0]
        if t == 2:
            bs, bv = jnp.zeros(2), jnp.zeros(3)
    mb = dict(obs=obs, actions=jnp.zeros((T, 1, N, 3)), rnn_resets=resets,
        active_masks=jnp.ones((T, 1, N), bool), comm_masks=jnp.zeros((T, 1, N, N), bool),
        base_signatures=jnp.stack(signatures)[:, None], base_values=jnp.stack(values)[:, None],
        base_memory_masks=masks, base_write_indices=jnp.array([[0], [-1], [-1], [-1]]),
        episode_dones=jnp.array([[False], [False], [True], [False]]),
        initial_actor_h=jnp.zeros((1, N, 8)), initial_critic_h=jnp.zeros((1, N, 8)),
        initial_actor_signature=jnp.zeros((1, N, 2)), initial_actor_value=jnp.zeros((1, N, 3)),
        aux_times=jnp.full((1, N, 1), 2))
    return model, mb


def test_base_replay_matches_forward_and_restores_sender_gradient():
    model, mb = base_replay_fixture()
    model.base_memory_replay = "stored"
    before = replay(model, mb, jax.random.PRNGKey(0), capture_auxiliary=True)
    def recipient_loss(obs):
        return replay(model, {**mb, "obs": obs}, jax.random.PRNGKey(0), capture_auxiliary=True)[5][0, 1].sum()
    blocked = jax.grad(recipient_loss)(mb["obs"])[0, 0, 0]
    model.base_memory_replay = "differentiable"
    after = replay(model, mb, jax.random.PRNGKey(0), capture_auxiliary=True)
    for a, b in zip(before, after):
        np.testing.assert_allclose(a, b, atol=1e-6)
    restored = jax.grad(recipient_loss)(mb["obs"])[0, 0, 0]
    assert float(jnp.linalg.norm(blocked)) == 0
    assert float(jnp.linalg.norm(restored)) > 1e-8
    # A recipient in the next episode must not credit the previous reporter.
    mb["aux_times"] = jnp.full((1, 2, 1), 3)
    assert float(jnp.linalg.norm(jax.grad(recipient_loss)(mb["obs"])[0, 0, 0])) == 0


def test_sweep_expands_to_21_loadable_unique_runs():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("cluster_plan", root / "cluster/tools/plan.py")
    plan = importlib.util.module_from_spec(spec); spec.loader.exec_module(plan)
    config = json.loads((root / "cluster/manifests/target-curriculum/g03_curriculum_auto7_21.json").read_text())
    runs = plan.expand_config(config, root, "test")
    assert len(runs) == len({name for name, _ in runs}) == 21
    counts = {}
    for _, run in runs:
        level = load_level(run["level"], run["overrides"])
        key = (level.training.target_curriculum.mode, level.training.geo_probe_auxiliary.lambda_target)
        counts[key] = counts.get(key, 0) + 1
        assert level.adapt_size.policy_vote_mode is None
        assert level.adapt_size.initial_agents == 7
        assert level.training.base_memory_replay == "differentiable"
        assert level.training.geo_probe_auxiliary.enabled
    assert counts == {(mode, coefficient): 1 if coefficient == 0 else 2
                      for mode in ("none", "forward", "reverse") for coefficient in (0, .03, .3, 1)}
