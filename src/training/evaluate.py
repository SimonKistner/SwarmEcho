"""
Load a SwarmEcho checkpoint, evaluate it in parallel, and optionally render
exactly one deterministic evaluation episode.

Usage
-----
    uv run python src/training/evaluate.py checkpoint=outputs/run/checkpoints/ckpt_001000
    uv run python src/training/evaluate.py checkpoint=... video=false
    uv run python src/training/evaluate.py checkpoint=... target_pos=40,25
"""

import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from core.config import compute_action_dim, compute_obs_dim, load_config, validate_config
from env.observations import make_obs_fns
from env.physics import make_env_fns
from env.rewards import make_reward_fn
from models.mappo import MAPPOModel
from training.artifacts import (
    checkpoint_artifact_suffix,
    eval_checkpoint_artifact_root,
    parse_checkpoint_update,
    save_eval_info_csv,
    steps_for_update,
    write_manifest,
)
from training.runner import (
    _collect_video_episode,
    _evaluate_parallel,
    _release_video_eval_trajectory,
)
from training.video_worker import render_eval_video


def _parse_args():
    checkpoint_path = None
    save_video = True
    target_pos = None
    overrides = []

    for arg in sys.argv[1:]:
        if arg.startswith("checkpoint="):
            checkpoint_path = Path(arg.split("=", 1)[1].replace("\\", "/"))
        elif arg.lower() in ("video=false", "--no-video"):
            save_video = False
        elif arg.startswith("target_pos="):
            x, y = arg.split("=", 1)[1].split(",", 1)
            target_pos = (float(x), float(y))
        elif arg.lower() in ("obs_log=true", "obs_saving=true", "--obs-log", "--obs-saving"):
            overrides.append("logging.obs_log=true")
        elif arg.lower() in ("obs_log=false", "obs_saving=false", "--no-obs-log", "--no-obs-saving"):
            overrides.append("logging.obs_log=false")
        elif arg.lower() in ("connectivity=true", "conn_matrix=true", "--connectivity", "--conn-matrix"):
            overrides.extend(("visualize.render_conn_matrix=true", "env.log_adjacency_matrix=true"))
        elif arg.lower() in ("connectivity=false", "conn_matrix=false", "--no-connectivity", "--no-conn-matrix"):
            overrides.extend(("visualize.render_conn_matrix=false", "env.log_adjacency_matrix=false"))
        else:
            overrides.append(arg)

    if checkpoint_path is None:
        raise ValueError("Specify checkpoint=<path>.")
    return checkpoint_path, save_video, target_pos, overrides


def _resolve_run_dir(checkpoint_path: Path):
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parents[1]
    return None


def main():
    checkpoint_path, save_video, target_pos, overrides = _parse_args()
    run_dir = _resolve_run_dir(checkpoint_path)
    run_config = run_dir / "config.yaml" if run_dir is not None else None
    has_run_config = run_config is not None and run_config.exists()

    cfg = load_config(
        config_path=run_config if has_run_config else None,
        cli_overrides=True,
        overrides=overrides,
    )
    validate_config(cfg)
    if run_dir is None:
        run_dir = Path(cfg.logging.get("log_dir", "outputs")).absolute()

    obs_dim = compute_obs_dim(cfg)
    act_dim = compute_action_dim(cfg)
    num_agents = int(cfg.env.num_agents)

    model = MAPPOModel(
        obs_dim=obs_dim,
        act_dim=act_dim,
        num_agents=num_agents,
        hidden_dim=int(cfg.network.hidden_dim),
        num_layers=int(cfg.network.num_layers),
        actor_num_layers=int(cfg.network.actor_num_layers),
        actor_memory=bool(cfg.network.get("actor_memory", False)),
        critic_memory=bool(cfg.network.get("critic_memory", False)),
        rngs=nnx.Rngs(0),
        memory_comm_enabled=bool(cfg.network.get("memory_comm_enabled", False)),
        memory_comm_every_k_steps=int(cfg.network.get("memory_comm_every_k_steps", 5)),
        tarmac_sig_dim=int(cfg.network.get("tarmac_sig_dim", 64)),
        tarmac_val_dim=int(cfg.network.get("tarmac_val_dim", 128)),
        tarmac_include_self=bool(cfg.network.get("tarmac_include_self", True)),
    )

    import orbax.checkpoint as ocp

    _, empty_state = nnx.split(model)
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    restored_state = checkpointer.restore(
        str(checkpoint_path.absolute()),
        args=ocp.args.StandardRestore(empty_state),
    )
    nnx.update(model, restored_state)

    env_step, reset, _, (resolved_width, resolved_height, occ_grid) = make_env_fns(cfg)
    compute_obs, _ = make_obs_fns(cfg, resolved_width, resolved_height, occ_grid)
    compute_reward = make_reward_fn(cfg)

    eval_reset = reset
    if target_pos is not None:
        target_x, target_y = target_pos

        def eval_reset(key):
            state = reset(key)
            return state.replace(
                target_pos=jnp.array([target_x, target_y], dtype=jnp.float32)
            )

    key = jax.random.PRNGKey(0)
    key, metrics_key, video_key = jax.random.split(key, 3)
    num_envs = int(cfg.evaluation.eval_parallel_envs)
    (
        returns,
        lengths,
        gaps,
        progress,
        successes,
        found,
        final_state,
        final_successes,
        _,
        _,
    ) = _evaluate_parallel(
        model,
        eval_reset,
        env_step,
        compute_obs,
        compute_reward,
        cfg,
        metrics_key,
        num_envs=num_envs,
    )

    print(f"\nEvaluation over {num_envs} parallel episodes")
    print(f"  mean_return:  {float(jnp.mean(returns)):.2f}")
    print(f"  mean_length:  {float(jnp.mean(lengths)):.1f}")
    print(f"  chain_prog:   {float(jnp.mean(progress)):.1f}%")
    print(f"  chain_gap:    {float(jnp.mean(gaps)):.1f} m")
    print(f"  success_rate: {float(jnp.mean(successes)):.1%}")
    print(f"  target_found: {float(jnp.mean(found)):.1%}")

    artifact_root = eval_checkpoint_artifact_root(run_dir, checkpoint_path, cfg)
    artifact_tag = checkpoint_artifact_suffix(checkpoint_path, cfg)
    manifest_dir = artifact_root / "manifests"

    if bool(cfg.evaluation.get("save_eval_info_as_csv", False)):
        data_dir = artifact_root / "data"
        info_path = save_eval_info_csv(
            data_dir / f"eval_info_{artifact_tag}.csv",
            target_positions=np.asarray(final_state.target_pos),
            base_positions=np.asarray(final_state.base_pos),
            successes=np.asarray(final_successes),
        )
        print(f"  eval_csv:     {info_path}")

    del returns, lengths, gaps, progress, successes, found, final_state, final_successes
    gc.collect()
    jax.clear_caches()

    if save_video and bool(cfg.evaluation.get("eval_video", True)):
        states, rewards, metrics = _collect_video_episode(
            model,
            jax.jit(eval_reset),
            jax.jit(env_step),
            jax.jit(compute_obs),
            jax.jit(compute_reward),
            cfg,
            video_key,
        )
        stem = f"eval_{artifact_tag}"
        video_path = render_eval_video(
            ep_states=states[0],
            ep_rewards=rewards[0],
            ep_metrics=metrics[0],
            cfg=cfg,
            out_dir=artifact_root / "vids",
            filename_stem=stem,
        )
        update = parse_checkpoint_update(checkpoint_path) or 0
        write_manifest(
            manifest_dir / f"{stem}.video.json",
            {
                "type": "video",
                "checkpoint": str(checkpoint_path),
                "checkpoint_name": checkpoint_path.name,
                "update": update,
                "steps": steps_for_update(update, cfg),
                "video_path": str(video_path),
            },
        )
        _release_video_eval_trajectory()
        print(f"  video:        {video_path}")


if __name__ == "__main__":
    main()
