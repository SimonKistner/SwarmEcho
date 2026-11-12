"""Source attribution uses the production collector without changing its actions."""
import jax
import jax.numpy as jnp
import numpy as np

from geo_path_runtime_fixture import Model, level, environment
from geo_experiments.shared.rollout import make_collector


def test_first_informed_actor_states_are_counted_once_and_done_is_unchanged():
    env=environment()
    keys=jax.random.split(jax.random.PRNGKey(7),1)
    ordinary=jax.device_get(make_collector(level(),environment=env)(Model(),keys))
    measured=jax.device_get(make_collector(level(),environment=env,knowledge_capture=True)(Model(),keys))
    for key in ordinary:
        np.testing.assert_array_equal(ordinary[key],measured[key])
    counts=measured['knowledge_first_counts'][0]
    assert counts.sum()==2
    assert counts[1]==1 # The finder saw the target directly.
    assert counts[4]==1 # Its peer learned before the base was informed.
    assert measured['knowledge_informed_steps'].sum()==6
    assert measured['knowledge_unattributed_events'].sum()==0


def test_base_only_acquisition_and_actual_cached_message_are_distinct_counters():
    reset,original_step,observations,_=environment()
    def step(state,action):
        next_state=original_step(state,action)
        # Peer 0 has left at step 4; at the next radio tick peer 1 can only
        # learn from the informed base, whose message was cached at step 2.
        return next_state._replace(target_known=jnp.array([next_state.step>=1,next_state.step>=5]))
    collect=make_collector(level(),environment=(reset,step,observations,None),knowledge_capture=True)
    batch=jax.device_get(collect(Model(),jax.random.split(jax.random.PRNGKey(7),1)))
    assert batch['knowledge_first_counts'][0,1]==1
    assert batch['knowledge_first_counts'][0,2]==1
    assert batch['knowledge_first_base_token_counts'][0,2]==1
    assert batch['knowledge_first_base_token_counts'].sum()==1
    assert batch['knowledge_unattributed_events'].sum()==0
