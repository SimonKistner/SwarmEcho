
import sys
from pathlib import Path

# Add src directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
from core.config import load_config
from env.physics import make_env_fns
from env.observations import make_obs_fns
from env.rewards import make_reward_fn

def smoke_test():
    print("🧪 Starting Environment Smoke Test...")
    
    # 1. Load config
    cfg = load_config()
    print("✅ Config loaded.")
    
    # 2. Setup env
    env_step, env_reset, _, _ = make_env_fns(cfg)
    print("✅ Physics functions created.")
    
    # 3. Setup obs
    compute_obs, obs_dim = make_obs_fns(cfg)
    print(f"✅ Observation functions created (dim={obs_dim}).")

    # 4. Setup reward
    compute_reward = make_reward_fn(cfg)
    print("✅ Reward function created.")
    
    # 5. Initialize state
    key = jax.random.PRNGKey(0)
    state = env_reset(key)
    print("✅ State reset.")
    
    # 6. Compute observations
    # This is where NameErrors usually hide!
    obs = compute_obs(state)
    print(f"✅ Observations computed successfully. Shape: {obs.shape}")
    
    # 7. Step env
    actions = jnp.zeros((cfg.env.num_agents, 2))
    next_state = env_step(state, actions)
    print("✅ Environment step successful.")

    # 8. Compute reward
    reward, info = compute_reward(state, next_state, is_done=False)
    print(f"✅ Reward computed. Mean reward: {jnp.mean(reward):.4f}")
    
    print("\n🎉 SMOKE TEST PASSED!")

if __name__ == "__main__":
    smoke_test()


