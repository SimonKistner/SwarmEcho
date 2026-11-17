"""PPO, recurrent communication, and training lifecycle checks."""

from collections import deque
from dataclasses import replace
from types import SimpleNamespace
import json

from flax import nnx
from omegaconf import OmegaConf
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from swarmecho.core.config import EnvConfig, load_level, MAP_DIR, TrainingConfig
from swarmecho.env.buildings import load_building
from swarmecho.env.environment import make_env_fns, observation_dim, RewardConfig
from swarmecho.env.knowledge import information_counts, new_information_counts
from swarmecho.models.critic import RecurrentAgentCentricCritic
from swarmecho.training.checkpoints import validate_checkpoint_contract
from swarmecho.training.information_metrics import (
    evaluation_information_metrics,
    information_percentages,
    pool_evaluation_information,
)
from swarmecho.training.ppo import loss, RunningValueNormalizer
from swarmecho.training.run_lifecycle import resolve_resume_state, resolve_run_layout, schedule_due
from swarmecho.training.train import _communication_inputs, _update_base_memory


# PPO controls

class Actor(nnx.Module):
    def __init__(self):
        self.mean = nnx.Param(jnp.float32(3.0))

    def __call_team__(self, obs, h, sig, val, **kwargs):
        shape = (*obs.shape[:-1], 3)
        return h, sig, val, jnp.broadcast_to(self.mean[...], shape), jnp.full(shape, -2.0)


class Critic(nnx.Module):
    def __init__(self):
        self.prediction = nnx.Param(jnp.float32(10.0))

    def values_sequence(self, obs, hidden, resets, **kwargs):
        return hidden, jnp.broadcast_to(self.prediction[...], obs.shape[:-1])


class Model(nnx.Module):
    def __init__(self):
        self.actor, self.critic = Actor(), Critic()
        self.value_normalizer = RunningValueNormalizer()


def minibatch():
    shape = (2, 1, 2)
    return {
        "obs": jnp.zeros((*shape, 1)),
        "actions": jnp.full((*shape, 3), 3.0), "old_log_probs": jnp.zeros(shape),
        "old_values": jnp.zeros(shape), "returns": jnp.full(shape, 20.0),
        "advantages": jnp.zeros(shape), "rnn_resets": jnp.zeros(shape, dtype=bool),
        "active_masks": jnp.broadcast_to(jnp.array([True, False]), shape),
        "comm_masks": jnp.zeros((*shape, 2), dtype=bool),
        "base_signatures": jnp.zeros((2, 1, 1)), "base_values": jnp.zeros((2, 1, 1)),
        "base_memory_masks": jnp.zeros(shape, dtype=bool),
        "base_write_indices": jnp.full(shape[:2], -1, dtype=jnp.int32),
        "episode_dones": jnp.zeros(shape[:2], dtype=bool),
        "initial_actor_h": jnp.zeros((1, 2, 1)),
        "initial_actor_signature": jnp.zeros((1, 2, 1)),
        "initial_actor_value": jnp.zeros((1, 2, 1)),
        "initial_critic_h": jnp.zeros((1, 2, 1)),
    }


def objective(model, batch, **overrides):
    settings = dict(actor_clip_eps=0.2, value_clip_eps=0.2, vf_coef=0.5, ent_coef=0.01)
    settings.update(overrides)
    return loss(model, batch, jax.random.PRNGKey(0), **settings)


def test_value_clipping_includes_zero_clip():
    model, batch = Model(), minibatch()
    assert float(objective(model, batch)[1]["value_loss"]) == pytest.approx(196.02)
    assert float(objective(model, batch, value_clip_eps=0.0)[1]["value_loss"]) == pytest.approx(200.0)
    assert float(objective(model, batch, actor_clip_eps=0.9)[1]["value_loss"]) == pytest.approx(196.02)


def test_inactive_targets_do_not_change_loss_or_gradients():
    model, batch = Model(), minibatch()
    changed = {**batch, "returns": batch["returns"].at[:, :, 1].set(1e10),
               "advantages": batch["advantages"].at[:, :, 1].set(1e10)}
    value, grad = nnx.value_and_grad(lambda m: objective(m, batch)[0])(model)
    other, other_grad = nnx.value_and_grad(lambda m: objective(m, changed)[0])(model)
    np.testing.assert_allclose(value, other)
    for a, b in zip(jax.tree.leaves(grad), jax.tree.leaves(other_grad)):
        np.testing.assert_allclose(a, b)


def test_squashed_entropy_pushes_saturated_mean_back_toward_center():
    model, batch = Model(), minibatch()
    grad = nnx.grad(lambda m: objective(m, batch, vf_coef=0.0, ent_coef=1.0)[0])(model)
    assert float(grad["actor"]["mean"].value) > 0.0


def test_value_normalizer_merges_moments_and_round_trips():
    normalizer = RunningValueNormalizer()
    normalizer.update(jnp.float32(2), jnp.float32(2), jnp.float32(1))  # samples 1, 3
    normalizer.update(jnp.float32(2), jnp.float32(6), jnp.float32(1))  # samples 5, 7
    assert float(normalizer.mean[...]) == pytest.approx(4.0)
    assert float(normalizer.variance[...]) == pytest.approx(5.0)
    raw = jnp.array([-100000.0, 0.0, 125.0])
    np.testing.assert_allclose(normalizer.denormalize(normalizer.normalize(raw)), raw, atol=0.01)
    normalizer.update(jnp.float32(0), jnp.float32(0), jnp.float32(0))
    assert float(normalizer.mean[...]) == pytest.approx(4.0)


def test_inactive_critic_tokens_are_invisible_and_all_inactive_is_finite():
    critic = RecurrentAgentCentricCritic(3, 8, 1, nnx.Rngs(0))
    obs, hidden = jnp.ones((1, 2, 3)), jnp.zeros((1, 2, 8))
    active = jnp.array([[True, False]])
    _, expected = critic(obs, hidden, active=active)
    _, actual = critic(obs.at[:, 1].set(1e6), hidden.at[:, 1].set(1e6), active=active)
    np.testing.assert_allclose(actual, expected)
    h, values = critic(obs, hidden, active=jnp.zeros_like(active))
    np.testing.assert_array_equal(h, 0.0)
    np.testing.assert_array_equal(values, 0.0)


def test_checkpoint_rejects_silent_value_unit_changes(tmp_path):
    model = SimpleNamespace(checkpoint_contract=(("format", 1), ("value_normalization", "running")))
    with pytest.raises(ValueError, match="no value-normalization metadata"):
        validate_checkpoint_contract(model, tmp_path)
    (tmp_path / "training_contract.json").write_text(json.dumps({"format": 1, "value_normalization": "none"}))
    with pytest.raises(ValueError, match="value_normalization"):
        validate_checkpoint_contract(model, tmp_path)


# Base memory

def test_base_stores_first_reporter_replays_and_clears_on_reset():
    cfg = EnvConfig(num_agents=4, comm_radius=10.0, comm_radius_base=10.0)
    building = load_building(MAP_DIR / "open_cuboid.yaml")
    reset, _, _, _ = make_env_fns(building, cfg)
    state = reset(jax.random.PRNGKey(0))
    state = state._replace(
        active=jnp.ones_like(state.active),
        target_known=state.target_known.at[2].set(True),
    )
    states = jax.tree_util.tree_map(lambda item: item[None], state)
    valid = jnp.asarray([False])
    saved_signature = jnp.zeros((1, 4))
    saved_value = jnp.zeros((1, 3))
    emitted_signature = jnp.arange(16, dtype=jnp.float32).reshape(1, 4, 4)
    emitted_value = jnp.arange(12, dtype=jnp.float32).reshape(1, 4, 3)

    _, in_base, masks, _, _ = _communication_inputs(
        states,
        cfg,
        valid,
        saved_signature,
        saved_value,
    )
    assert not masks.any()
    valid, saved_signature, saved_value = _update_base_memory(
        states,
        in_base,
        emitted_signature,
        emitted_value,
        valid,
        saved_signature,
        saved_value,
        jnp.asarray([False]),
    )
    assert valid[0]
    assert jnp.array_equal(saved_signature[0], emitted_signature[0, 2])
    assert jnp.array_equal(saved_value[0], emitted_value[0, 2])

    _, _, masks, _, _ = _communication_inputs(
        states,
        cfg,
        valid,
        saved_signature,
        saved_value,
    )
    assert masks[0, 0]
    assert not masks[0, 2]

    valid, saved_signature, saved_value = _update_base_memory(
        states,
        in_base,
        emitted_signature,
        emitted_value,
        valid,
        saved_signature,
        saved_value,
        jnp.asarray([True]),
    )
    assert not valid[0]
    assert not saved_signature.any()
    assert not saved_value.any()


def test_obstacle_blocks_tarmac_base_reporting_during_rollout_and_replay():
    cfg = EnvConfig(num_agents=4, comm_radius=10.0, comm_radius_base=10.0)
    building = load_building(MAP_DIR / "open_cuboid.yaml")
    reset, _, _, _ = make_env_fns(building, cfg)
    state = reset(jax.random.PRNGKey(1))._replace(
        pos=jnp.asarray([
            [2.0, 0.0, 1.0],
            [8.0, 0.0, 1.0],
            [2.0, 3.0, 1.0],
            [2.0, 6.0, 1.0],
        ]),
        base_pos=jnp.asarray([0.0, 0.0, 1.0]),
        active=jnp.ones(4, dtype=jnp.bool_),
        target_known=jnp.asarray([False, True, False, False]),
        obstacle_min=jnp.asarray([[4.0, -1.0, 0.0]]),
        obstacle_max=jnp.asarray([[6.0, 1.0, 2.0]]),
    )
    states = jax.tree_util.tree_map(lambda item: item[None], state)
    valid = jnp.asarray([False])
    saved_signature = jnp.zeros((1, 4))
    saved_value = jnp.zeros((1, 3))

    communication = jax.jit(
        lambda batched, base_valid: _communication_inputs(
            batched, cfg, base_valid, saved_signature, saved_value
        )
    )
    comm_mask, in_base, memory_mask, _, _ = communication(states, valid)
    assert not comm_mask[0, 0, 1]  # In range, but the cuboid blocks TarMAC.
    assert comm_mask[0, 0, 2]  # An unobstructed in-range peer remains reachable.
    assert not in_base[0, 1]  # The informed drone cannot report through the cuboid.

    emitted_signature = jnp.arange(16, dtype=jnp.float32).reshape(1, 4, 4)
    emitted_value = jnp.arange(12, dtype=jnp.float32).reshape(1, 4, 3)
    valid, _, _ = _update_base_memory(
        states, in_base, emitted_signature, emitted_value,
        valid, saved_signature, saved_value, jnp.asarray([False]),
    )
    assert not valid[0]

    valid = jnp.asarray([True])
    state = state._replace(target_known=jnp.zeros(4, dtype=jnp.bool_))
    states = jax.tree_util.tree_map(lambda item: item[None], state)
    _, _, memory_mask, _, _ = communication(states, valid)
    assert not memory_mask[0, 1]  # Base memory cannot replay through the cuboid.
    assert memory_mask[0, 2]  # An unobstructed drone can receive base memory.


def test_base_memory_is_delivered_only_to_uninformed_agents():
    cfg = EnvConfig(num_agents=4, comm_radius=10.0, comm_radius_base=10.0)
    building = load_building(MAP_DIR / "open_cuboid.yaml")
    reset, _, _, _ = make_env_fns(building, cfg)
    state = reset(jax.random.PRNGKey(19))._replace(
        active=jnp.ones(cfg.num_agents, dtype=jnp.bool_),
        target_known=jnp.ones(cfg.num_agents, dtype=jnp.bool_),
    )
    states = jax.tree_util.tree_map(lambda item: item[None], state)
    valid = jnp.asarray([True])
    signature = jnp.zeros((1, 4))
    value = jnp.zeros((1, 3))
    _, in_base, default_mask, _, _ = _communication_inputs(
        states, cfg, valid, signature, value,
    )
    assert not default_mask.any()
    assert in_base.any()


# Training lifecycle

def test_run_layout_is_identical_for_dictconfig_and_dataclasses(tmp_path):
    level = load_level("G03_CLU_random_buildings")
    logging = replace(level.logging, run_name="shared", log_dir=str(tmp_path))
    evaluation = replace(level.evaluation, checkpoint_dir=None)
    typed = resolve_run_layout(logging, evaluation, media_dir="replays")
    omega = resolve_run_layout(
        OmegaConf.create({"run_name": "shared", "log_dir": str(tmp_path)}),
        OmegaConf.create({"checkpoint_dir": None}),
        media_dir="vids",
    )

    assert typed.run_dir == omega.run_dir == tmp_path / "shared"
    assert typed.checkpoint_dir == omega.checkpoint_dir == tmp_path / "shared/checkpoints"
    assert typed.train_media_dir == tmp_path / "shared/artifacts/train/replays"
    assert omega.train_media_dir == tmp_path / "shared/artifacts/train/vids"


def test_resume_history_and_schedules_are_dimension_agnostic(tmp_path):
    level = load_level("G03_CLU_random_buildings")
    checkpoint = tmp_path / "run/checkpoints/ckpt_000020"
    checkpoint.mkdir(parents=True)
    (checkpoint / "step_history.json").write_text(
        json.dumps({"total_steps": 8_000_000, "history": [{"run_name": "run", "steps": 8_000_000}]})
    )
    training = replace(level.training, checkpoint_path=str(checkpoint), ckpt_loading_mode="resume")
    resume = resolve_resume_state(training, "run", 4000, 100)

    assert resume.start_update == 20
    assert resume.step_offset == 0
    assert resume.prior_history == []
    assert schedule_due(21, 20, 1)
    assert not schedule_due(20, 20, 1)


def test_init_restores_weights_without_inheriting_parent_timeline(tmp_path):
    level = load_level("G03_CLU_random_buildings")
    checkpoint = tmp_path / "parent/checkpoints/ckpt_001250"
    checkpoint.mkdir(parents=True)
    (checkpoint / "step_history.json").write_text(
        json.dumps({"total_steps": 500_000_000, "history": [{"run_name": "parent", "steps": 500_000_000}]})
    )
    training = replace(level.training, checkpoint_path=str(checkpoint), ckpt_loading_mode="init")

    resume = resolve_resume_state(training, "tall", 4000, 100)

    assert resume.start_update == 0
    assert resume.step_offset == 0
    assert resume.prior_history == []


# Information attribution

def test_first_acquisitions_prioritize_sight_then_base_and_exclude_inactive():
    # Existing knower; direct sight + base; base + peer; peer; inactive; unknown.
    counts = jax.jit(new_information_counts)(
        jnp.array([1, 0, 0, 0, 0, 0], bool),
        jnp.array([1, 1, 1, 1, 1, 0], bool),
        jnp.array([1, 1, 1, 1, 0, 1], bool),
        jnp.array([0, 1, 0, 0, 0, 0], bool),
        jnp.array([1, 1, 1, 0, 1, 0], bool))
    np.testing.assert_array_equal(counts, [3, 1, 1])


def test_reset_sightings_count_once_and_reused_slots_can_learn_again():
    previous = SimpleNamespace(step=0, active=jnp.array([1, 1], bool),
                               target_known=jnp.array([1, 0], bool))
    current = SimpleNamespace(new_information_counts=jnp.array([1, 0, 1]))
    np.testing.assert_array_equal(information_counts(previous, current), [2, 0, 1])
    previous.step = 1
    np.testing.assert_array_equal(information_counts(previous, current), [1, 0, 1])
    # Respawning clears knowledge; the new occupant counts on acquisition.
    np.testing.assert_array_equal(new_information_counts(
        jnp.array([0], bool), jnp.array([1], bool), jnp.array([1], bool),
        jnp.array([0], bool), jnp.array([1], bool)), [1, 0, 1])


def test_completed_episode_window_pools_agents_and_evicts_old_episodes():
    window = deque(maxlen=2)
    window.extend(([10, 9, 0], [2, 0, 1]))
    assert information_percentages(window) == {"peer-informed-pct": 75., "base-informed-pct": 100/12}
    window.append([0, 0, 0])  # Zero-information episodes still occupy the window.
    assert information_percentages(window) == {"peer-informed-pct": 0., "base-informed-pct": 50.}
    window.append([0, 0, 0])
    assert information_percentages(window) == {}


@pytest.mark.parametrize("counts", [([0, 0, 0], [10, 9, 1]), ([2, 0, 1], [10, 9, 0])])
def test_map_and_robustness_pooling_uses_counts_even_if_first_pass_has_no_information(counts):
    result = pool_evaluation_information([evaluation_information_metrics(row) for row in counts])
    assert result == evaluation_information_metrics(counts)
    assert result["eval_peer-informed-pct"] == pytest.approx(900 / sum(row[0] for row in counts))


def test_environment_emits_source_counts_with_one_hop_delivery():
    from dataclasses import replace
    from pathlib import Path
    from swarmecho.core.config import EnvConfig
    from swarmecho.env.buildings import load_building
    from swarmecho.env.environment import make_env_fns

    building = load_building(Path("src/swarmecho/curriculum_config/maps/custom_maps/open_cuboid.yaml"))
    cfg = replace(EnvConfig(), num_agents=3, spawn_delay=0, comm_radius=1.5,
                  comm_radius_base=1.5, visual_radius=1., max_steps=20,
                  hold_chain_for=20)
    reset, step, _, _ = make_env_fns(building, cfg)
    state = reset(jax.random.PRNGKey(91), target_pos=jnp.array([17., 17., 12.]))._replace(
        pos=jnp.array([[3., 2., 2.], [4., 2., 2.], [5., 2., 2.]]),
        vel=jnp.zeros((3, 3)), base_pos=jnp.array([2., 2., 2.]),
        active=jnp.ones(3, bool), target_known=jnp.array([1, 0, 0], bool),
        base_target_known=jnp.bool_(True), step=jnp.int32(3))
    # Recipient 1 receives a peer message; recipient 2 waits for a later tick.
    action = jnp.zeros((3, 3))
    current = step(state, action)
    np.testing.assert_array_equal(information_counts(state, current), [1, 1, 0])
    state = current
    # Both base and informed peer are in range: base takes precedence.
    state = state._replace(pos=state.pos.at[2].set(jnp.array([3.2, 2., 2.])), step=jnp.int32(6))
    current = step(state, action)
    np.testing.assert_array_equal(information_counts(state, current), [1, 0, 1])


def test_validation_counts_only_its_episodes_excluding_batch_padding(tmp_path, monkeypatch):
    from dataclasses import replace
    from pathlib import Path
    import json
    from swarmecho.core.config import load_level
    from swarmecho.training import train
    from swarmecho.training.validation import ValidationRunner

    level = load_level("G03_CLU_random_buildings")
    level = replace(level, validation=replace(level.validation, num_maps=1,
        num_episodes=3, parallel_envs=2, fixed_targets=False))
    runner = ValidationRunner(level, tmp_path,
        [Path("src/swarmecho/curriculum_config/maps/custom_maps/open_cuboid.yaml")])
    monkeypatch.setattr(runner, "_prepare", lambda: (None, None, None, None))
    calls = []

    def batch(*args, **kwargs):
        assert kwargs["return_population"] and kwargs["return_required_agents"]
        assert kwargs["return_useless_agents"] and kwargs["return_information"]
        values = [np.zeros(2) for _ in range(19)]
        values[8] = np.zeros((2, 3))
        values[12] = np.ones(2)  # Valid target samples.
        values[14] = np.ones(2, bool)  # Valid decommission zones.
        values[16] = np.ones(2, dtype=np.int32)  # Required agent counts.
        values[18] = np.array([[4, 1, 2], [2, 1, 1]] if not calls else
                              [[1, 1, 0], [999, 0, 999]])
        calls.append(True)
        return tuple(values)

    monkeypatch.setattr(train, "_run_parallel_evaluation_jit", batch)
    metrics = runner.run(SimpleNamespace(), update=1, steps=100)
    assert metrics["peer-informed-pct"] == pytest.approx(300/7)
    assert metrics["base-informed-pct"] == pytest.approx(300/7)
    result = json.loads((tmp_path / "validation/results_u000001.json").read_text())
    assert result["information_counts"] == [7, 3, 3]


def test_real_training_update_reports_sources_from_completed_episodes(tmp_path):
    from dataclasses import replace
    from test_population import tiny_level
    from swarmecho.training.train import train

    level = tiny_level()
    level = replace(level,
        env=replace(level.env, visual_radius=100.),
        training=replace(level.training, num_envs=2, num_steps=6,
                         total_timesteps=12, num_minibatches=1),
        evaluation=replace(level.evaluation, save_model=False, eval_video=False,
                           eval_parallel_envs=2),
        logging=replace(level.logging, wandb_mode="disabled"))
    _, metrics = train(level, output_dir=tmp_path)
    assert metrics["completed_episodes"] >= 2
    # Every deployed drone sees the target directly in this tiny environment.
    assert metrics["peer-informed-pct"] == 0
    assert metrics["base-informed-pct"] == 0
