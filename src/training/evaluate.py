"""
training/evaluate.py
====================
Load a SwarmEcho MAPPO checkpoint and render deterministic evaluation video(s).

Usage
-----
    # Legacy mode: compute and render N episodes
    uv run python training/evaluate.py \\
        checkpoint=outputs/my_run/checkpoints/ckpt_001000 \\
        eval_render_videos=3

    # Selective mode: compute up to 100 episodes, render 5 successes + 5 failures
    uv run python training/evaluate.py \\
        checkpoint=outputs/my_run/checkpoints/ckpt_001000 \\
        selective=true \\
        eval_max_compute_episodes=100 \\
        render_successes=5 \\
        render_failures=5

    # Stats-only mode (both buckets = 0): compute 50 episodes, print stats, no video
    uv run python training/evaluate.py \\
        checkpoint=outputs/my_run/checkpoints/ckpt_001000 \\
        selective=true \\
        eval_max_compute_episodes=50 \\
        render_successes=0 \\
        render_failures=0

    # Skip all video
    uv run python training/evaluate.py checkpoint=... video=False

Output
------
    outputs/<run>/videos/eval/
        SUCCESS_eval_<ckpt>_ep00.mp4
        FAIL_eval_<ckpt>_ep00.mp4
        ...  (or eval_<ckpt>_ep00.mp4 in legacy mode)
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
from training.runner import _evaluate, _make_selective_eval_callback
from training.video_worker import render_eval_video


def main():
    # ── Parse CLI args ────────────────────────────────────────────────────
    args = sys.argv[1:]
    checkpoint_path = None
    save_video      = True
    renderer_override = None          # explicit --renderer= override
    rep_first_to_last = False

    # Selective mode args (override cfg values when provided)
    selective_override            = None   # selective=true/false
    eval_max_compute_override     = None   # eval_max_compute_episodes=N
    render_successes_override     = None   # render_successes=Y
    render_failures_override      = None   # render_failures=Z
    eval_render_videos_override   = None   # eval_render_videos=N (legacy)
    target_pos_override           = None   # target_pos=x,y

    overrides = []

    for arg in args:
        if arg.startswith("checkpoint="):
            checkpoint_path = Path(arg.split("=", 1)[1].replace("\\", "/"))
        elif arg.startswith("--renderer="):
            renderer_override = arg.split("=", 1)[1]
        elif arg.lower() in ["video=false", "--no-video"]:
            save_video = False
        elif arg.lower() in ["rep_first_to_last=true", "--rep-first-to-last"]:
            rep_first_to_last = True
        elif arg.lower() in ["selective=true"]:
            selective_override = True
        elif arg.lower() in ["selective=false"]:
            selective_override = False
        elif arg.startswith("eval_max_compute_episodes="):
            eval_max_compute_override = int(arg.split("=", 1)[1])
        elif arg.startswith("render_successes="):
            render_successes_override = int(arg.split("=", 1)[1])
        elif arg.startswith("render_failures="):
            render_failures_override = int(arg.split("=", 1)[1])
        elif arg.startswith("eval_render_videos="):
            eval_render_videos_override = int(arg.split("=", 1)[1])
        elif arg.startswith("target_pos="):
            parts = arg.split("=", 1)[1].split(",")
            target_pos_override = (float(parts[0]), float(parts[1]))
        elif arg.lower() in ["obs_log=true", "obs_saving=true", "--obs-log", "--obs-saving"]:
            overrides.append("logging.obs_log=true")
        elif arg.lower() in ["obs_log=false", "obs_saving=false", "--no-obs-log", "--no-obs-saving"]:
            overrides.append("logging.obs_log=false")
        elif arg.lower() in ["connectivity=true", "conn_matrix=true", "--connectivity", "--conn-matrix"]:
            overrides.append("visualize.render_conn_matrix=true")
            overrides.append("env.log_adjacency_matrix=true")
        elif arg.lower() in ["connectivity=false", "conn_matrix=false", "--no-connectivity", "--no-conn-matrix"]:
            overrides.append("visualize.render_conn_matrix=false")
            overrides.append("env.log_adjacency_matrix=false")
        else:
            overrides.append(arg)

    if checkpoint_path is None:
        print("ERROR: Must specify checkpoint=<path>")
        print("  e.g.  uv run python training/evaluate.py checkpoint=outputs/my_run/checkpoints/ckpt_001000")
        sys.exit(1)

    # ── Resolve run directory early (needed for config auto-load) ─────────
    # Standard layout: outputs/<run_name>/checkpoints/ckpt_XXXXXX
    if checkpoint_path.parent.name == "checkpoints":
        run_dir = checkpoint_path.parents[1]
    else:
        # Non-standard path: fall back after config load (log_dir unknown yet)
        run_dir = None

    # ── Config ───────────────────────────────────────────────────────────
    # Auto-load the run's saved config.yaml so the correct map / env params
    # are always used when running standalone — no manual CLI override needed.
    # Any explicit CLI key=value args are still merged on top (highest priority).
    run_config_path = (run_dir / "config.yaml") if run_dir is not None else None
    run_config_loaded = run_config_path is not None and run_config_path.exists()
    cfg         = load_config(
        config_path  = run_config_path if run_config_loaded else None,
        cli_overrides = True,
        overrides     = overrides,
    )
    validate_config(cfg)

    # If run_dir was deferred, resolve it now that cfg is available
    if run_dir is None:
        run_dir = Path(cfg.logging.get("log_dir", "outputs")).absolute()

    obs_dim     = compute_obs_dim(cfg)
    act_dim     = compute_action_dim(cfg)
    N           = int(cfg.env.num_agents)
    critic_type = str(cfg.network.critic_type)

    # Resolve render settings: CLI args override cfg values
    selective = (
        selective_override
        if selective_override is not None
        else bool(cfg.visualize.get("selective_eval_render", False))
    )
    eval_render_videos = (
        eval_render_videos_override
        if eval_render_videos_override is not None
        else int(cfg.visualize.get("eval_render_videos", 1))
    )
    eval_max_compute = (
        eval_max_compute_override
        if eval_max_compute_override is not None
        else int(cfg.visualize.get("eval_max_compute_episodes", 50))
    )
    n_success = (
        render_successes_override
        if render_successes_override is not None
        else int(cfg.visualize.get("eval_render_successes", 3))
    )
    n_fail = (
        render_failures_override
        if render_failures_override is not None
        else int(cfg.visualize.get("eval_render_failures", 3))
    )

    # Renderer priority: CLI --renderer= > cfg.visualize.final_eval_renderer > "slow"
    renderer = (
        renderer_override
        if renderer_override is not None
        else str(cfg.visualize.get("final_eval_renderer", cfg.visualize.get("renderer", "slow")))
    )

    print(f"\n{'═'*54}")
    print(f"  SwarmEcho — Evaluation  [{critic_type} critic]")
    print(f"{'═'*54}")
    print(f"  checkpoint        : {checkpoint_path}")
    if run_config_loaded:
        print(f"  config            : {run_config_path} (auto)")
    else:
        print(f"  config            : Python defaults (no config.yaml found)")
    print(f"  renderer          : {renderer}")
    if selective:
        print(f"  mode              : selective")
        print(f"  eval_max_compute  : {eval_max_compute}")
        print(f"  render_successes  : {n_success}")
        print(f"  render_failures   : {n_fail}")
    else:
        print(f"  mode              : legacy")
        print(f"  eval_render_videos: {eval_render_videos}")
    print(f"  devices           : {jax.devices()}")

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
        actor_memory     = bool(cfg.network.get("actor_memory", False)),
        critic_memory    = bool(cfg.network.get("critic_memory", False)),
        rngs             = rngs,
        memory_comm_enabled = bool(cfg.network.get("memory_comm_enabled", False)),
        memory_comm_gradient_mode = str(cfg.network.get("memory_comm_gradient_mode", "rial")),
        memory_comm_every_k_steps = int(cfg.network.get("memory_comm_every_k_steps", 5)),
        memory_comm_num_heads = int(cfg.network.get("memory_comm_num_heads", 4)),
        memory_comm_merge = str(cfg.network.get("memory_comm_merge", "residual")),
        memory_comm_attention_mode = str(cfg.network.get("memory_comm_attention_mode", "attend_global_learned_query")),
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
    env_step, reset, _, (resolved_W, resolved_H, occ_grid, comm_occ_grid) = make_env_fns(cfg)
    compute_obs, _     = make_obs_fns(cfg, resolved_W, resolved_H, occ_grid, comm_occ_grid)
    compute_reward     = make_reward_fn(cfg)

    # ── Output directory ──────────────────────────────────────────────────
    # run_dir was resolved earlier (before config load) so videos land in the
    # correct run folder whether config.yaml was auto-loaded or not.
    out_dir = run_dir / "videos" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_name = checkpoint_path.name  # e.g. "ckpt_000762"

    # ── Evaluate ──────────────────────────────────────────────────────────
    key = jax.random.PRNGKey(0)

    eval_reset = reset
    if target_pos_override is not None:
        tx, ty = target_pos_override
        def make_override_reset(r_fn, target_x, target_y):
            def override_reset(k):
                s = r_fn(k)
                return s.replace(target_pos=jnp.array([target_x, target_y], dtype=jnp.float32))
            return override_reset
        eval_reset = make_override_reset(reset, tx, ty)

    if selective:
        # Selective mode: stream episodes, fill SUCCESS_/FAIL_ buckets inline
        callback = _make_selective_eval_callback(
            n_success = n_success,
            n_fail    = n_fail,
            out_dir   = out_dir if save_video else Path("/dev/null"),
            ckpt_name = f"eval_{ckpt_name}",
            renderer  = renderer,
            cfg       = cfg,
        )
        (all_states, all_rewards, all_metrics,
         mean_ret, mean_len, mean_gap, mean_prog, success_rate, found_rate) = _evaluate(
            model,
            jax.jit(eval_reset),
            jax.jit(env_step),
            jax.jit(compute_obs),
            jax.jit(compute_reward),
            cfg, key,
            num_episodes=eval_max_compute,
            episode_callback=callback,
        )
        n_computed = len(all_states)

    else:
        # Legacy mode: compute and render eval_render_videos episodes sequentially
        (all_states, all_rewards, all_metrics,
         mean_ret, mean_len, mean_gap, mean_prog, success_rate, found_rate) = _evaluate(
            model,
            jax.jit(eval_reset),
            jax.jit(env_step),
            jax.jit(compute_obs),
            jax.jit(compute_reward),
            cfg, key,
            num_episodes=eval_render_videos,
        )
        n_computed = eval_render_videos

        if save_video:
            for ep_idx, (ep_states, ep_rewards, ep_metrics) in enumerate(zip(all_states, all_rewards, all_metrics)):
                stem = f"eval_{ckpt_name}_ep{ep_idx:02d}"
                vid_path = render_eval_video(
                    ep_states  = ep_states,
                    ep_rewards = ep_rewards,
                    ep_metrics = ep_metrics,
                    cfg        = cfg,
                    out_dir    = out_dir,
                    filename_stem = stem,
                    renderer   = renderer,
                )
                print(f"  [ep {ep_idx}] video → {vid_path}")

    # ── Print results ─────────────────────────────────────────────────────
    print(f"\n  Results over {n_computed} episode(s):")
    print(f"    mean_return  : {mean_ret:.2f}")
    print(f"    mean_length  : {mean_len:.1f} steps")
    print(f"    chain_prog   : {mean_prog:.1f}%")
    print(f"    chain_gap    : {mean_gap:.1f} m")
    print(f"    success_rate : {success_rate:.1%}")
    print(f"    target_found : {found_rate:.1%}")

    if not save_video:
        print("\n  [info] Video rendering skipped (video=False)")

    # ── Last-N-steps analysis (optional) ─────────────────────────────────
    if rep_first_to_last:
        WINDOW = 10
        all_window_progs = []
        all_window_gaps  = []

        for met in all_metrics:
            progs = met["chain_pct"]
            gaps  = met["chain_gap"]
            length = len(progs)

            start_idx = max(0, length - WINDOW)
            win_progs = progs[start_idx:]
            win_gaps  = gaps[start_idx:]

            if len(win_progs) < WINDOW:
                pad = [None] * (WINDOW - len(win_progs))
                win_progs = pad + list(win_progs)
                win_gaps  = pad + list(win_gaps)

            all_window_progs.append(win_progs)
            all_window_gaps.append(win_gaps)

        mean_progs = []
        mean_gaps  = []
        for i in range(WINDOW):
            vals_p = [ep[i] for ep in all_window_progs if ep[i] is not None]
            vals_g = [ep[i] for ep in all_window_gaps if ep[i] is not None]
            mean_progs.append(np.mean(vals_p) if vals_p else 0.0)
            mean_gaps.append(np.mean(vals_g) if vals_g else 0.0)

        print(f"\n  Last {WINDOW} Steps Analysis (Mean over {n_computed} eps):")
        print(f"  {'Step':<12} | {'Progress':<12} | {'Gap':<10}")
        print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*10}")
        for i in range(WINDOW):
            offset = (WINDOW - 1) - i
            lbl = f"Success-{offset}" if offset > 0 else "Success (Final)"
            print(f"  {lbl:<12} | {mean_progs[i]:>10.1f}%   | {mean_gaps[i]:>8.1f} m")

    print("\n  Done.")


if __name__ == "__main__":
    main()
