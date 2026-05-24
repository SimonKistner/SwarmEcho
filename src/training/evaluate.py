"""
training/evaluate.py
====================
Load a SwarmEcho MAPPO checkpoint and render a deterministic evaluation video.

Usage
-----
    uv run python training/evaluate.py checkpoint=outputs/my_run/checkpoints/ckpt_001000

    # Override any config value (must match original training config)
    uv run python training/evaluate.py checkpoint=... env.num_agents=4

    # Render more episodes
    uv run python training/evaluate.py checkpoint=... eval_episodes=10

    # Skip video rendering (only print metrics)
    uv run python training/evaluate.py checkpoint=... video=False

Output
------
    outputs/<run>/videos/eval_<ckpt>_<timestamp>_ep<N>.mp4
"""

import sys
from pathlib import Path
import csv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from omegaconf import OmegaConf
from datetime import datetime

from core.config import load_config, validate_config, compute_obs_dim, compute_action_dim
from env.physics import make_env_fns
from env.observations import make_obs_fns
from env.rewards import make_reward_fn
from models.mappo import MAPPOModel
from training.runner import _evaluate
from visualize.renderer import render_video


DEFAULT_EVAL_EPISODES = 10


def main():
    # ── Parse CLI args ────────────────────────────────────────────────────
    args = sys.argv[1:]
    checkpoint_path = None
    eval_episodes   = DEFAULT_EVAL_EPISODES
    save_video      = True
    renderer        = None
    rep_first_to_last = False # If True, reports a table of metrics from the last 10 steps before success
    overrides       = []

    for arg in args:
        if arg.startswith("checkpoint="):
            checkpoint_path = Path(arg.split("=", 1)[1])
        elif arg.startswith("eval_episodes="):
            eval_episodes = int(arg.split("=", 1)[1])
        elif arg.startswith("--renderer="):
            renderer = arg.split("=", 1)[1]
        elif arg.lower() in ["video=false", "--no-video"]:
            save_video = False
        elif arg.lower() in ["rep_first_to_last=true", "--rep-first-to-last"]:
            rep_first_to_last = True
        else:
            overrides.append(arg)

    if checkpoint_path is None:
        print("ERROR: Must specify checkpoint=<path>")
        print("  e.g.  uv run python training/evaluate.py checkpoint=outputs/my_run/checkpoints/ckpt_001000")
        sys.exit(1)

    # ── Config ───────────────────────────────────────────────────────────
    cfg         = load_config(cli_overrides=True, overrides=overrides)
    validate_config(cfg)
    obs_dim     = compute_obs_dim(cfg)
    act_dim     = compute_action_dim(cfg)
    N           = int(cfg.env.num_agents)
    critic_type = str(cfg.network.critic_type)

    print(f"\n{'═'*54}")
    print(f"  SwarmEcho — Evaluation  [{critic_type} critic]")
    print(f"{'═'*54}")
    print(f"  checkpoint   : {checkpoint_path}")
    print(f"  num_episodes : {eval_episodes}")
    print(f"  devices      : {jax.devices()}")

    # ── Build model ───────────────────────────────────────────────────────
    rngs  = nnx.Rngs(0)
    model = MAPPOModel(
        obs_dim          = obs_dim,
        act_dim          = act_dim,
        num_agents       = N,
        hidden_dim       = int(cfg.network.hidden_dim),
        num_layers       = int(cfg.network.num_layers),
        actor_num_layers = int(cfg.network.actor_num_layers),
        critic_type      = critic_type,
        rngs             = rngs,
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
    print("  Checkpoint loaded ✓")

    _, params = nnx.split(model)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Model params : {n_params:,}")

    # ── Environment ───────────────────────────────────────────────────────
    env_step, reset, _, (resolved_W, resolved_H, occ_grid) = make_env_fns(cfg)
    compute_obs, _     = make_obs_fns(cfg, resolved_W, resolved_H, occ_grid)
    compute_reward     = make_reward_fn(cfg)

    # ── Evaluate ──────────────────────────────────────────────────────────
    key = jax.random.PRNGKey(0)
    (all_states, all_rewards, all_metrics,
     mean_ret, mean_len, mean_gap, mean_prog, success_rate, found_rate) = _evaluate(
        model,
        jax.jit(reset),
        jax.jit(env_step),
        jax.jit(compute_obs),
        jax.jit(compute_reward),
        cfg, key, eval_episodes,
    )

    print(f"\n  Results over {eval_episodes} episode(s):")
    print(f"    mean_return  : {mean_ret:.2f}")
    print(f"    mean_length  : {mean_len:.1f} steps")
    print(f"    chain_prog   : {mean_prog:.1f}%")
    print(f"    chain_gap    : {mean_gap:.1f} m")
    print(f"    success_rate : {success_rate:.1%}")
    print(f"    target_found : {found_rate:.1%}")
    
    # ── Double-check: Last 10 Steps ──────────────────────────────────────
    if rep_first_to_last:
        # Number of steps to look back
        WINDOW = 10
        
        # Accumulators for the last WINDOW steps across all episodes
        # We'll use a list of lists: [[ep0_last9, ..., ep0_last0], [ep1_last9, ...]]
        all_window_progs = []
        all_window_gaps  = []
        
        for met in all_metrics:
            progs = met["chain_pct"]
            gaps  = met["chain_gap"]
            length = len(progs)
            
            # Extract the last 10 steps (or fewer if episode is short)
            start_idx = max(0, length - WINDOW)
            win_progs = progs[start_idx:]
            win_gaps  = gaps[start_idx:]
            
            # Right-align (pad with None if short) to keep success at the end
            if len(win_progs) < WINDOW:
                pad = [None] * (WINDOW - len(win_progs))
                win_progs = pad + list(win_progs)
                win_gaps  = pad + list(win_gaps)
                
            all_window_progs.append(win_progs)
            all_window_gaps.append(win_gaps)
            
        # Calculate means per "offset from end"
        mean_progs = []
        mean_gaps  = []
        for i in range(WINDOW):
            vals_p = [ep[i] for ep in all_window_progs if ep[i] is not None]
            vals_g = [ep[i] for ep in all_window_gaps if ep[i] is not None]
            mean_progs.append(np.mean(vals_p) if vals_p else 0.0)
            mean_gaps.append(np.mean(vals_g) if vals_g else 0.0)
            
        print(f"\n  Last {WINDOW} Steps Analysis (Mean over {eval_episodes} eps):")
        print(f"  {'Step':<12} | {'Progress':<12} | {'Gap':<10}")
        print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*10}")
        for i in range(WINDOW):
            offset = (WINDOW - 1) - i
            label = f"Success-{offset}" if offset > 0 else "Success (Final)"
            print(f"  {label:<12} | {mean_progs[i]:>10.1f}%   | {mean_gaps[i]:>8.1f} m")

    # ── Render ───────────────────────────────────────────────────────────
    if save_video:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if checkpoint_path.parent.name == "checkpoints":
            run_dir = checkpoint_path.parents[1]
        else:
            run_dir = Path(cfg.logging.get("log_dir", "outputs")).absolute()

        out_dir = run_dir / "videos"
        out_dir.mkdir(parents=True, exist_ok=True)

        for ep_idx, (ep_states, ep_rewards, ep_metrics) in enumerate(zip(all_states, all_rewards, all_metrics)):
            traj        = jax.tree.map(lambda *xs: jnp.stack(xs), *ep_states)
            rewards_arr = jnp.array(ep_rewards)
            ckpt_name   = checkpoint_path.name
            vid_path    = render_video(
                traj, cfg,
                filename = str(out_dir / f"eval_{ckpt_name}_{ts}_ep{ep_idx:02d}.mp4"),
                fps      = 20,
                renderer = renderer,
                rewards  = rewards_arr,
                extra_metrics = ep_metrics,
            )
            print(f"  [ep {ep_idx}] video → {vid_path}")

    else:
        print("\n  [info] Video rendering skipped (video=False)")

    print("\n  Done.")


if __name__ == "__main__":
    main()
