"""Execution microbatches preserve one update per original PPO minibatch.

Every loss is normalized against the original minibatch, not each execution
chunk. Entropy draws are generated once and sliced so chunking preserves RNGs.
"""
import functools
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


def slice_minibatch(mb, start, end):
    result = {}
    for key,value in mb.items():
        if key == "ppo_normalizer":
            result[key] = value
        elif key.startswith(("initial_","aux_")) or key == "env_indices":
            result[key] = value[start:end]
        else:
            result[key] = value[:,start:end]
    return result


def gradient(model, mb, key, coefficient, **settings):
    from swarmecho.training.ppo import loss
    (_,stats),grads = nnx.value_and_grad(lambda m:loss(m,mb,key,aux_lambda=coefficient,**settings),has_aux=True)(model)
    return stats,grads


@nnx.jit
def apply(model, optimizer, grads):
    optimizer.update(model,grads)


def update(trainer, minibatches):
    from swarmecho.training.geo_auxiliary import POLICY_FILTER, PREFIX, gradients_diagnostics
    trainer.updates += 1
    active_counts = np.asarray(jax.device_get(jnp.stack([jnp.sum(m["active_masks"]) for m in minibatches])))
    minibatches = [mb for mb,count in zip(minibatches,active_counts) if count > 0]
    if not minibatches:
        return {k:0. for k in ("policy_loss","value_loss","entropy","total_loss","approx_kl","clip_fraction","critic/value_clip_blocked_fraction")}
    count = sum(jnp.sum(mb["active_masks"]) for mb in minibatches)
    mean = sum(jnp.sum(jnp.where(mb["active_masks"],mb["returns"],0.)) for mb in minibatches)/jnp.maximum(count,1)
    if hasattr(trainer.model,"value_normalizer"):
        var = sum(jnp.sum(jnp.where(mb["active_masks"],(mb["returns"]-mean)**2,0.)) for mb in minibatches)/jnp.maximum(count,1)
        trainer.model.value_normalizer.update(count,mean,var)
    measure = (trainer.updates-1)%trainer.diagnostics_every == 0
    before = trainer._measure(minibatches,mean) if measure else None
    has_geo = trainer.has_auxiliary
    aux_counts = (np.asarray(jax.device_get(jnp.stack([jnp.sum(mb["aux_valid"]) for mb in minibatches])))
                  if has_geo else np.zeros(len(minibatches)))
    if has_geo and aux_counts.sum():
        trainer.model.geo_probe_head.supervised_updates[...] += 1
    coefficient = trainer.model.geo_probe_head.weight() if has_geo else jnp.float32(0)
    if not hasattr(trainer,"_gradient"):
        trainer._gradient = nnx.jit(functools.partial(gradient,**trainer._settings))
    root = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(trainer.seed),0xE170),trainer.updates)
    summed = None
    recorded = []
    gradient_stats = {}
    for epoch in range(trainer.num_epochs):
        for index, original in enumerate(minibatches):
            key = jax.random.fold_in(root,epoch*len(minibatches)+index)
            B = original["obs"].shape[1]
            if B % trainer.microbatches:
                raise ValueError("Execution microbatches must divide the original PPO minibatch.")
            mb = {**original,"ppo_normalizer":jnp.sum(original["active_masks"]),
                  "entropy_noise":jax.random.normal(key,(*original["actions"].shape[:-1],3))}
            size = B//trainer.microbatches
            grads = scalar_stats = None
            scores = []
            diag = has_geo and trainer.geo_config.gradient_diagnostics and measure and epoch==0 and index==0 and aux_counts[index]>0
            gp = ga = None
            for start in range(0,B,size):
                micro = slice_minibatch(mb,start,start+size)
                if diag:
                    # Two reverse passes replace the ordinary one for this one
                    # minibatch. Their affine combination is the desired update.
                    stats,g0 = trainer._gradient(trainer.model,micro,key,jnp.float32(0))
                    _,g1 = trainer._gradient(trainer.model,micro,key,jnp.float32(1))
                    delta = jax.tree.map(lambda a,b:b-a,g0,g1)
                    current = jax.tree.map(lambda a,b:a+coefficient*b,g0,delta)
                    gp = g0 if gp is None else jax.tree.map(jnp.add,gp,g0)
                    ga = delta if ga is None else jax.tree.map(jnp.add,ga,delta)
                else:
                    stats,current = trainer._gradient(trainer.model,micro,key,coefficient)
                grads = current if grads is None else jax.tree.map(jnp.add,grads,current)
                scalars = {k:v for k,v in stats.items() if k!="aux_scores"}
                scalar_stats = scalars if scalar_stats is None else jax.tree.map(jnp.add,scalar_stats,scalars)
                if has_geo and epoch==trainer.num_epochs-1:
                    scores.append(stats["aux_scores"])
            if diag:
                gradient_stats = gradients_diagnostics(gp,ga,coefficient)
            if has_geo:
                apply(trainer.model,trainer.optimizer,grads.filter(POLICY_FILTER))
                if aux_counts[index]>0:
                    apply(trainer.model.geo_probe_head,trainer.head_optimizer,grads["geo_probe_head"])
            else:
                apply(trainer.model,trainer.optimizer,grads)
            if scores:
                recorded.append((original["env_indices"],jnp.concatenate(scores,axis=0)))
            n = scalar_stats["count"]
            weighted = {k:v if k=="count" else v*n for k,v in scalar_stats.items()}
            summed = weighted if summed is None else jax.tree.map(jnp.add,summed,weighted)
    host,grad_host,record_host = jax.device_get((summed,gradient_stats,recorded))
    denom = max(float(host["count"]),1.)
    result = {k:float(v)/denom for k,v in host.items() if k!="count"}
    result.update({k:float(v) for k,v in grad_host.items() if np.isfinite(v)})
    if has_geo:
        result[PREFIX+"data/supervised_samples_per_update"] = float(aux_counts.sum())
        if trainer.auxiliary_data is not None:
            for indices,values in record_host:
                trainer.auxiliary_data.record_scores(indices,values)
    if measure:
        after = trainer._measure(minibatches,mean)
        n = max(float(after["count"]),1.)
        mse,var = float(after["error_sq"])/n,float(after["target_centered_sq"])/n
        result.update({"critic/rmse_before":math.sqrt(float(before["error_sq"])/n),
            "critic/rmse_after":math.sqrt(mse),"critic/relative_mse_after":mse/var if var>1e-8 else float("nan"),
            "policy/gaussian_entropy":float(after["gaussian"])/n,"policy/squashed_entropy":float(after["squashed"])/n,
            **{f"policy/action_saturation_{a}":float(after["saturation"][i])/n for i,a in enumerate("xyz")}})
    return result


def save_optimizer(trainer, directory):
    import orbax.checkpoint as ocp
    states = {"policy":nnx.state(trainer.optimizer)}
    if trainer.head_optimizer is not None:
        states["head"] = nnx.state(trainer.head_optimizer)
    ocp.Checkpointer(ocp.StandardCheckpointHandler()).save(str(Path(directory).absolute()/"optimizer_state"),args=ocp.args.StandardSave(states))


def restore_optimizer(trainer, directory):
    import orbax.checkpoint as ocp
    path = Path(directory).absolute()/"optimizer_state"
    if not path.is_dir():
        if trainer.has_auxiliary:
            raise ValueError("Auxiliary resume requires checkpointed optimizer state; use branch/init for legacy checkpoints.")
        return
    states = {"policy":nnx.state(trainer.optimizer)}
    if trainer.head_optimizer is not None:
        states["head"] = nnx.state(trainer.head_optimizer)
    restored = ocp.Checkpointer(ocp.StandardCheckpointHandler()).restore(str(path),args=ocp.args.StandardRestore(states))
    nnx.update(trainer.optimizer,restored["policy"])
    if trainer.head_optimizer is not None:
        nnx.update(trainer.head_optimizer,restored["head"])
