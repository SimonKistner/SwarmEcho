"""Execution-only memory controls; effective rollout and PPO batches stay fixed."""
from contextlib import contextmanager


def recovery_training_origin(directory, now):
    """Keep the learning-time budget across disposable workers of one attempt.

    The first worker starts this clock after setup, just like ordinary training.
    Failed work and retry setup then consume the same budget, rather than giving
    every OOM retry another four hours. A new supervisor gets a new recovery ID.
    """
    import json
    import os
    from pathlib import Path
    identity = os.environ.get("SWARMECHO_RECOVERY_ID")
    if not identity:
        return now
    path = Path(directory) / "oom_recovery_clock.json"
    if path.is_file():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["recovery_id"] == identity:
            return float(saved["training_started"])
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(dict(recovery_id=identity, training_started=now)), encoding="utf-8")
    os.replace(temporary, path)
    return now


@contextmanager
def memory_phase(name):
    try:
        yield
    except Exception as error:
        from swarmecho.training.launcher import is_out_of_memory
        if is_out_of_memory(error):
            error.swarmecho_oom_phase = name
        raise


def run_rollout(function, model, *args, chunk_size=None, **kwargs):
    import jax
    import jax.numpy as jnp
    import numpy as np
    E = args[1].shape[0]  # Explicit per-lane RNG keys.
    size = min(chunk_size or E,E)
    with memory_phase("rollout"):
        if size == E:
            result = function(model,*args,**kwargs)
            jax.block_until_ready(result)
            return result
        pieces=[]
        for start in range(0,E,size):
            n=min(size,E-start)
            indices=jnp.minimum(jnp.arange(start,start+size),E-1)
            sliced=jax.tree.map(lambda value:jnp.take(value,indices,axis=0),args)
            result=function(model,*sliced,**kwargs)
            *carry,values,rollout=result
            values,rollout=jax.device_get((values,rollout))
            pieces.append((*jax.tree.map(lambda value:value[:n],tuple(carry)),
                values[:n],{key:value[:,:n] for key,value in rollout.items()}))
            del result,carry,values,rollout,sliced
        carry=tuple(jax.tree.map(lambda *values:jnp.concatenate(values,axis=0),
            *[piece[i] for piece in pieces]) for i in range(len(pieces[0])-2))
        values=np.concatenate([piece[-2] for piece in pieces],axis=0)
        rollout={key:np.concatenate([piece[-1][key] for piece in pieces],axis=1) for key in pieces[0][-1]}
        jax.block_until_ready(carry)
        return (*carry,values,rollout)
