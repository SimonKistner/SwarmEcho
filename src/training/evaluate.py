"""
training/evaluate.py
====================
Load a SwarmEcho MAPPO/IPPO checkpoint and render a deterministic evaluation video.

Usage
-----
    # Evaluate a MAPPO checkpoint
    uv run python training/evaluate.py checkpoint=outputs/my_run/checkpoints/ckpt_001000

    # Override any config value (must match original training config)
    uv run python training/evaluate.py checkpoint=outputs/my_run/checkpoints/ckpt_001000 \\
        training.use_mappo=true env.num_agents=4 logging.wandb_mode=disabled

    # Render more episodes
    uv run python training/evaluate.py checkpoint=... eval_episodes=10

Output
------
    outputs/eval_{timestamp}.mp4
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
from flax import nnx
from omegaconf import OmegaConf

from core.config import load_config, validate_config, compute_obs_dim, compute_action_dim
from env.physics import make_env_fns
from env.observations import make_obs_fns
from env.rewards import make_reward_fn
from models.mappo import MAPPOModel
from models.actor_critic import ActorCritic
from training.runner import _evaluate
from visualize.renderer import render_video

from datetime import datetime

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_EVAL_EPISODES = 1

def main():
    # ── Parse args ───────────────────────────────────────────────────────
    # Accept --checkpoint / checkpoint= style from CLI
    args = sys.argv[1:]
    checkpoint_path = None
    eval_episodes   = DEFAULT_EVAL_EPISODES
    renderer        = None
    overrides       = []

    for arg in args:
        if arg.startswith("checkpoint="):
            checkpoint_path = Path(arg.split("=", 1)[1])
        elif arg.startswith("eval_episodes="):
            eval_episodes = int(arg.split("=", 1)[1])
        elif arg.startswith("--renderer="):
            renderer = arg.split("=", 1)[1]
        else:
            overrides.append(arg)

    if checkpoint_path is None:
        print("ERROR: Must specify checkpoint=<path>")
        print("  e.g.  uv run python training/evaluate.py checkpoint=outputs/my_run/checkpoints/ckpt_001000")
        sys.exit(1)

    # ── Config ───────────────────────────────────────────────────────────
    cfg = load_config(cli_overrides=True, overrides=overrides)
    validate_config(cfg)

    use_mappo     = bool(cfg.training.get("use_mappo", True))
    obs_dim       = compute_obs_dim(cfg)
    act_dim       = compute_action_dim(cfg)
    N             = int(cfg.env.num_agents)
    num_episodes  = eval_episodes
    max_force     = float(cfg.env.max_force)

    print(f"\n{'═'*54}")
    print(f"  SwarmEcho — Evaluation")
    print(f"{'═'*54}")
    print(f"  checkpoint   : {checkpoint_path}")
    print(f"  mode         : {'MAPPO' if use_mappo else 'IPPO'}")
    print(f"  num_episodes : {num_episodes}")
    print(f"  devices      : {jax.devices()}")

    # ── Build model ───────────────────────────────────────────────────────
    rngs = nnx.Rngs(0)
    if use_mappo:
        model = MAPPOModel(
            obs_dim    = obs_dim,
            act_dim    = act_dim,
            num_agents = N,
            hidden_dim = int(cfg.network.hidden_dim),
            num_layers = int(cfg.network.num_layers),
            rngs       = rngs,
        )
    else:
        model = ActorCritic(
            obs_dim    = obs_dim,
            act_dim    = act_dim,
            hidden_dim = int(cfg.network.hidden_dim),
            num_layers = int(cfg.network.num_layers),
            rngs       = rngs,
        )

    # ── Load checkpoint ───────────────────────────────────────────────────
    import orbax.checkpoint as ocp

    graphdef, empty_state = nnx.split(model)
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    restored_state = checkpointer.restore(
        str(checkpoint_path.absolute()),
        args=ocp.args.StandardRestore(empty_state),
    )
    nnx.update(model, restored_state)
    print(f"  Checkpoint loaded ✓")

    # Count params
    _, params = nnx.split(model)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Model params : {n_params:,}")

    # ── Environment ───────────────────────────────────────────────────────
    env_step, reset, _ = make_env_fns(cfg)
    compute_obs, _     = make_obs_fns(cfg)
    compute_reward     = make_reward_fn(cfg)

    reset_jit      = jax.jit(reset)
    env_step_jit   = jax.jit(env_step)
    compute_obs_jit = jax.jit(compute_obs)
    compute_reward_jit = jax.jit(compute_reward)

    # ── Evaluate ──────────────────────────────────────────────────────────
    key = jax.random.PRNGKey(0)
    (all_states, all_rewards,
     mean_ret, mean_gap, success_rate, found_rate) = _evaluate(
        model, reset_jit, env_step_jit, compute_obs_jit, compute_reward_jit,
        cfg, key, num_episodes, use_mappo,
    )

    print(f"\n  Results over {num_episodes} episode(s):")
    print(f"    mean_return   : {mean_ret:.2f}")
    print(f"    chain_gap     : {mean_gap:.1f} m")
    print(f"    success_rate  : {success_rate:.1%}")
    print(f"    target_found  : {found_rate:.1%}")

    # ── Render all episodes ───────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Try to save the video inside the run's folder (e.g., outputs/run_xyz/videos/eval)
    # The checkpoint path usually looks like outputs/run_xyz/checkpoints/ckpt_000
    if checkpoint_path.parent.name == "checkpoints":
        run_dir = checkpoint_path.parents[1]
    else:
        run_dir = Path(cfg.logging.get("log_dir", "outputs")).absolute()
        
    out_dir = run_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep_idx, (ep_states, ep_rewards) in enumerate(zip(all_states, all_rewards)):
        traj = jax.tree.map(lambda *xs: jnp.stack(xs), *ep_states)
        rewards_arr = jnp.array(ep_rewards)
        
        # Include the checkpoint name in the video filename if possible
        ckpt_name = checkpoint_path.name
        
        vid_path = render_video(
            traj, cfg,
            filename     = str(out_dir / f"eval_{ckpt_name}_{ts}_ep{ep_idx:02d}.mp4"),
            fps          = 20,
            renderer     = renderer,
            rewards      = rewards_arr,
        )
        print(f"  [ep {ep_idx}] video → {vid_path}")

    print("\n  Done.")


if __name__ == "__main__":
    main()


