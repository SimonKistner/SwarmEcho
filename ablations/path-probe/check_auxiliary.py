"""Auxiliary gradients, masking, execution equivalence and checkpoint state."""
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from swarmecho.core.config import (TrainingConfig, GeoProbeAuxiliaryConfig,
    GeoAuxHeadConfig, GeoAuxScheduleConfig, GeoAuxSamplingConfig, load_level)
from swarmecho.models.mappo import MAPPOModel
from swarmecho.training.geo_auxiliary import (PathHead, AuxiliaryData, balanced_weights,
    POLICY_FILTER)
from swarmecho.training.ppo import loss, replay, PPOTrainer
from swarmecho.training.ppo_execution import restore_optimizer


@pytest.fixture(autouse=True)
def precise_equivalence_checks():
    # GPU TF32 kernels may change with execution shapes. Compare the mathematics
    # in full precision; production retains the project's existing precision.
    with jax.default_matmul_precision("highest"):
        yield


def setup(microbatches=1, enabled=True, gradient_diagnostics=False):
    geo = GeoProbeAuxiliaryConfig(enabled=enabled, head=GeoAuxHeadConfig((8,),.001),
        schedule=GeoAuxScheduleConfig(0,0), sampling=GeoAuxSamplingConfig(snapshots_per_agent=2),
        gradient_diagnostics=gradient_diagnostics)
    cfg = replace(TrainingConfig(),num_envs=4,num_minibatches=1,num_epochs=1,
        optimizer_microbatches=microbatches,geo_probe_auxiliary=geo,diagnostics_every=100)
    model = MAPPOModel(3,3,2,8,1,1,nnx.Rngs(7),actor_memory=True,critic_memory=True,
        memory_comm_enabled=True,tarmac_sig_dim=2,tarmac_val_dim=3)
    if enabled:
        model.geo_probe_head = PathHead(8,geo,7)
    T,B,N,H = 4,4,2,8
    masks = np.ones((T,B,N),bool)
    masks[2:,1,1] = False
    mb = dict(obs=jax.random.normal(jax.random.PRNGKey(1),(T,B,N,3)),
        actions=jnp.zeros((T,B,N,3)),old_log_probs=jnp.zeros((T,B,N)),
        old_values=jnp.zeros((T,B,N)),returns=jnp.ones((T,B,N)),
        advantages=jnp.arange(T*B*N,dtype=jnp.float32).reshape(T,B,N)/10,
        rnn_resets=jnp.zeros((T,B,N),bool),active_masks=jnp.asarray(masks),
        comm_masks=jnp.ones((T,B,N,N),bool),base_signatures=jnp.zeros((T,B,2)),
        base_values=jnp.zeros((T,B,3)),base_memory_masks=jnp.zeros((T,B,N),bool),
        initial_actor_h=jnp.zeros((B,N,H)),initial_critic_h=jnp.zeros((B,N,H)),
        initial_actor_signature=jnp.zeros((B,N,2)),initial_actor_value=jnp.zeros((B,N,3)),
        env_indices=jnp.arange(B))
    if enabled:
        times = np.broadcast_to([0,3],(B,N,2)).copy()
        times[1,1,1] = -1
        valid = times>=0
        targets = np.zeros((B,N,2,125),bool); targets[...,62]=True
        episodes = np.broadcast_to(np.arange(B)[:,None,None],valid.shape)
        agents = np.broadcast_to(np.arange(N)[None,:,None],valid.shape)
        mb.update(aux_times=jnp.array(times),aux_valid=jnp.array(valid),aux_targets=jnp.array(targets),
            aux_episodes=jnp.asarray(episodes),aux_weights=jnp.array(balanced_weights(episodes,agents,valid)))
    return model,cfg,mb


SETTINGS = dict(actor_clip_eps=.2,value_clip_eps=.2,entropy_mode="squashed",vf_coef=.5,ent_coef=.01)


def test_config_defaults_disable_feature_and_accept_scoped_settings():
    assert not TrainingConfig().geo_probe_auxiliary.enabled
    level = load_level("G03_CLU_random_buildings",["training.geo_probe_auxiliary.enabled=true",
        "training.geo_probe_auxiliary.sampling.snapshots_per_agent=5"])
    assert level.training.geo_probe_auxiliary.sampling.snapshots_per_agent==5
    with pytest.raises(ValueError,match="positive"):
        load_level("G03_CLU_random_buildings",["training.geo_probe_auxiliary.sampling.snapshots_per_agent=0"])


def test_stratified_sampling_never_duplicates_and_keeps_lifetime_eligibility():
    labels=SimpleNamespace(get=lambda *args:np.ones(125,bool))
    data=AuxiliaryData(GeoProbeAuxiliaryConfig(sampling=GeoAuxSamplingConfig(snapshots_per_agent=5)),7,labels)
    def select(known):
        T=len(known)
        return data.prepare(dict(active_masks=np.ones((T,1,1),bool),
            geo_target_known=np.array(known)[:,None,None],dones=np.zeros((T,1),bool),
            geo_map_id=np.zeros((T,1),int),geo_base=np.zeros((T,1,3)),geo_target=np.ones((T,1,3))),1)["aux_times"][0,0]
    selected=select(np.arange(50)>=10)
    assert len(set(selected))==5
    assert np.all(selected>=np.arange(10,50,8))
    assert np.all(selected<np.arange(18,58,8))
    np.testing.assert_array_equal(select([False,True,False,True]),[1,3,-1,-1,-1])


def test_selected_replay_states_match_direct_actor_unroll():
    model,_,mb = setup()
    actual = replay(model,mb,jax.random.PRNGKey(0),capture_auxiliary=True)[5]
    h,sig,val = mb["initial_actor_h"],mb["initial_actor_signature"],mb["initial_actor_value"]
    history=[]
    for t in range(4):
        h,sig,val,_,_ = jax.vmap(lambda o,a,b,c,r,cm,ac,bs,bv,bm:model.actor.__call_team__(
            o,a,b,c,reset=r|~ac,comm_mask=cm,active=ac,base_signature=bs,base_value=bv,base_memory_mask=bm))(
            mb["obs"][t],h,sig,val,mb["rnn_resets"][t],mb["comm_masks"][t],mb["active_masks"][t],
            mb["base_signatures"][t],mb["base_values"][t],mb["base_memory_masks"][t])
        history.append(np.asarray(h))
    history=np.stack(history)
    for b,a,k in np.ndindex(actual.shape[:-1]):
        t=int(mb["aux_times"][b,a,k])
        np.testing.assert_allclose(actual[b,a,k],history[t,b,a] if t>=0 else 0,atol=2e-6)


def test_auxiliary_policy_gradient_scales_but_head_gradient_does_not():
    model,_,mb=setup()
    gradients=[]
    for coefficient in (0.,.03,1.):
        gradients.append(nnx.grad(lambda m:loss(m,mb,jax.random.PRNGKey(2),aux_lambda=coefficient,**SETTINGS)[0])(model))
    g0,g03,g1=gradients
    for a,b in zip(jax.tree.leaves(g0["geo_probe_head"]),jax.tree.leaves(g1["geo_probe_head"])):
        np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-7)
    magnitude=0.
    for a,b,c in zip(jax.tree.leaves(g0.filter(POLICY_FILTER)),jax.tree.leaves(g03.filter(POLICY_FILTER)),jax.tree.leaves(g1.filter(POLICY_FILTER))):
        np.testing.assert_allclose(b,a+.03*(c-a),atol=2e-6,rtol=1e-5)
        magnitude+=float(jnp.sum((c-a)**2))
    assert magnitude>1e-8


@pytest.mark.parametrize("enabled",[False,True])
def test_accumulation_preserves_effective_update_with_unequal_masks(enabled):
    full,cfg,mb=setup(enabled=enabled)
    chunked,chunkcfg,chunkmb=setup(microbatches=2,enabled=enabled)
    t1,t2=PPOTrainer(full,cfg),PPOTrainer(chunked,chunkcfg)
    t1.update([mb]);t2.update([chunkmb])
    for a,b in zip(jax.tree.leaves(nnx.state(full,nnx.Param)),jax.tree.leaves(nnx.state(chunked,nnx.Param))):
        np.testing.assert_allclose(a,b,atol=2e-5,rtol=2e-5)
    assert int(t1.optimizer.step[...])==int(t2.optimizer.step[...])==1
    if enabled:
        assert int(t1.head_optimizer.step[...])==int(t2.head_optimizer.step[...])==1


def test_optimizers_and_schedule_checkpoint_round_trip(tmp_path):
    from swarmecho.training.checkpoints import save_training_checkpoint, restore_model_checkpoint
    model,cfg,mb=setup()
    trainer=PPOTrainer(model,cfg);trainer.update([mb])
    checkpoint=tmp_path/"ckpt_000001"
    save_training_checkpoint(model,checkpoint,trainer=trainer,run_name="test",update=1,
        num_envs=4,num_steps=4,prior_history=[])
    other,_,_=setup();target=PPOTrainer(other,cfg)
    restore_model_checkpoint(other,checkpoint)
    restore_optimizer(target,checkpoint)
    target.updates=1
    for a,b in zip(jax.tree.leaves(nnx.state(trainer.optimizer)),jax.tree.leaves(nnx.state(target.optimizer))):
        np.testing.assert_array_equal(a,b)
    assert int(target.head_optimizer.step[...])==1
    assert int(other.geo_probe_head.supervised_updates[...])==1
    trainer.update([mb]);target.update([mb])
    for a,b in zip(jax.tree.leaves(nnx.state(model)),jax.tree.leaves(nnx.state(other))):
        np.testing.assert_allclose(a,b,atol=2e-6,rtol=2e-6)


def test_empty_auxiliary_samples_leave_head_adam_unchanged():
    model,cfg,mb=setup();trainer=PPOTrainer(model,cfg)
    mb["aux_valid"]=jnp.zeros_like(mb["aux_valid"]);mb["aux_weights"]=jnp.zeros_like(mb["aux_weights"])
    trainer.update([mb])
    assert int(trainer.head_optimizer.step[...])==0
    assert int(model.geo_probe_head.supervised_updates[...])==0


def test_warmup_and_ramp_count_only_supervised_updates():
    head = PathHead(8, GeoProbeAuxiliaryConfig(lambda_target=.3), 7)
    for update, expected in ((0,0.),(10,0.),(11,.003),(60,.15),(110,.3),(120,.3)):
        head.supervised_updates[...] = jnp.int32(update)
        assert float(head.weight()) == pytest.approx(expected)


def test_zero_lambda_preserves_policy_update_with_independent_head_clipping():
    policy, baseline, mb = setup(enabled=False)
    joint, cfg, jointmb = setup()
    joint.geo_probe_head.lambda_target = 0.
    PPOTrainer(policy, baseline).update([mb])
    PPOTrainer(joint, cfg).update([jointmb])
    for a,b in zip(jax.tree.leaves(nnx.state(policy,nnx.Param)),
                   jax.tree.leaves(nnx.state(joint,POLICY_FILTER))):
        np.testing.assert_allclose(a,b,atol=2e-6,rtol=2e-6)


def test_selection_tracks_auto_resets_and_skips_only_missing_auxiliary_labels():
    class Labels:
        def __init__(self): self.requests=[]
        def get(self, map_id, base, target):
            self.requests.append(int(map_id))
            return None if map_id==3 else np.ones(125,bool)
    labels=Labels()
    data=AuxiliaryData(GeoProbeAuxiliaryConfig(sampling=GeoAuxSamplingConfig(snapshots_per_agent=5)),7,labels)
    known=np.array([[[False,False]],[[True,False]],[[False,False]],[[True,True]],[[True,True]]])
    rollout=dict(active_masks=np.ones((5,1,2),bool),geo_target_known=known,
        dones=np.array([[False],[True],[False],[False],[True]]),
        geo_map_id=np.array([[0],[0],[3],[3],[3]]),geo_base=np.zeros((5,1,3)),geo_target=np.ones((5,1,3)))
    selected=data.prepare(rollout,1)
    np.testing.assert_array_equal(data.episode_frames[:,0],[0,0,1,1,1])
    assert sorted(labels.requests)==[0,3] # Once per selected episode, not per agent/snapshot.
    assert selected["aux_valid"].sum()==1
    assert selected["aux_times"][selected["aux_valid"]].item()==1
    assert selected["aux_episodes"][selected["aux_valid"]].item()==0
    assert data.live.item()==2
    assert rollout["dones"].sum()==2 # PPO's episodes remain present.


def test_balancing_equalizes_episodes_then_agents_then_snapshots():
    episode=np.array([0,0,0,1]);agent=np.array([0,0,1,0]);valid=np.ones(4,bool)
    np.testing.assert_allclose(balanced_weights(episode,agent,valid),[.125,.125,.25,.5])


def test_oom_recovery_changes_only_execution_shapes():
    from swarmecho.training.launcher import next_memory_level
    level=load_level("G03_CLU_random_buildings",[
        "training.oom_recovery.enabled=true","training.oom_recovery.parameter=optimizer_microbatches",
        "training.oom_recovery.factor=2.0"])
    scientific=(level.training.num_envs,level.training.num_steps,level.training.num_minibatches,level.training.num_epochs)
    for phase in ("optimizer","rollout","validation"):
        changed=next_memory_level(level,phase)
        assert (changed.training.num_envs,changed.training.num_steps,changed.training.num_minibatches,changed.training.num_epochs)==scientific
        assert changed.training.geo_probe_auxiliary==level.training.geo_probe_auxiliary
        if phase=="optimizer": assert changed.training.optimizer_microbatches==2
        elif phase=="rollout": assert changed.training.rollout_batch_size==level.training.num_envs//2
        else: assert changed.validation.parallel_envs==level.validation.parallel_envs//2
    exhausted=replace(level,training=replace(level.training,optimizer_microbatches=1000))
    with pytest.raises(RuntimeError,match="one environment"):
        next_memory_level(exhausted,"optimizer")


def test_rollout_chunking_preserves_all_lanes_and_random_keys():
    from swarmecho.training.memory_execution import run_rollout
    @jax.jit
    def collect(model,states,keys):
        draws=jax.vmap(lambda k:jax.random.uniform(k,(3,)))(keys)
        return states+1,keys,draws.sum(axis=1),dict(observations=jnp.stack((states,states+draws)))
    states=jnp.arange(15,dtype=jnp.float32).reshape(5,3)
    keys=jax.random.split(jax.random.PRNGKey(7),5)
    full=run_rollout(collect,None,states,keys)
    chunks=run_rollout(collect,None,states,keys,chunk_size=2)
    for a,b in zip(jax.tree.leaves(full),jax.tree.leaves(chunks)):
        np.testing.assert_array_equal(a,b)


def test_all24_partition_exactly_into_one_test_and_remaining23():
    import json
    import sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/"cluster/tools"))
    from plan import expand_config
    plans=[]
    for name,count in (("all24",24),("test",1),("remaining23",23)):
        config=json.loads((root/f"cluster/manifests/geo-auxiliary/geo_auxiliary_{name}.json").read_text())
        plan=expand_config(config,root,"test")
        assert len(plan)==count
        for _,run in plan:
            level=load_level(run["level"],run["overrides"])
            assert level.training.oom_recovery.parameter=="optimizer_microbatches"
            assert level.training.max_training_seconds==14400
        plans.append({(name,tuple(run["overrides"])) for name,run in plan})
    assert not plans[1]&plans[2]
    assert plans[0]==plans[1]|plans[2]


def test_retries_share_one_learning_time_budget(tmp_path, monkeypatch):
    from swarmecho.training.memory_execution import recovery_training_origin
    monkeypatch.delenv("SWARMECHO_RECOVERY_ID",raising=False)
    assert recovery_training_origin(tmp_path,100)==100
    monkeypatch.setenv("SWARMECHO_RECOVERY_ID","attempt-a")
    assert recovery_training_origin(tmp_path,200)==200
    assert recovery_training_origin(tmp_path,300)==200
    monkeypatch.setenv("SWARMECHO_RECOVERY_ID","attempt-b")
    assert recovery_training_origin(tmp_path,400)==400


def test_gradient_diagnostics_preserve_update_and_measure_finite_pressure():
    model,cfg,mb=setup(microbatches=2,gradient_diagnostics=True)
    ordinary,ordinary_cfg,ordinary_mb=setup(microbatches=2)
    stats=PPOTrainer(model,cfg).update([mb])
    PPOTrainer(ordinary,ordinary_cfg).update([ordinary_mb])
    ratio=stats["geo_probe_auxiliary/gradient/aux_to_ppo_ratio"]
    cosine=stats["geo_probe_auxiliary/gradient/cosine_similarity"]
    assert np.isfinite(ratio) and ratio>0
    assert -1.00001<=cosine<=1.00001
    for a,b in zip(jax.tree.leaves(nnx.state(model)),jax.tree.leaves(nnx.state(ordinary))):
        np.testing.assert_allclose(a,b,atol=2e-6,rtol=2e-6)
