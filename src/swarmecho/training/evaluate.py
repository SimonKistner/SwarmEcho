"""Evaluate one checkpoint with the maintained parallel evaluator.

Usage:
    uv run swarmecho-evaluate checkpoint=outputs/run/checkpoints/ckpt_001000
    uv run swarmecho-evaluate checkpoint=... video=false
    uv run swarmecho-evaluate checkpoint=... target_pos=40,25
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from swarmecho.core.config import load_config, validate_config
from swarmecho.training.artifacts import (
    checkpoint_artifact_suffix,
    eval_checkpoint_artifact_root,
    parse_checkpoint_update,
    save_eval_info_csv,
    steps_for_update,
    write_manifest,
)
from swarmecho.training.evaluation import (
    collect_video_episode,
    evaluate_parallel,
    release_video_evaluation_trajectory,
)
from swarmecho.training.runtime import build_evaluation_runtime
from swarmecho.training.video_worker import render_eval_video


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
        elif arg.lower() in (
            "connectivity=true",
            "conn_matrix=true",
            "--connectivity",
            "--conn-matrix",
        ):
            overrides.append("visualize.render_conn_matrix=true")
        elif arg.lower() in (
            "connectivity=false",
            "conn_matrix=false",
            "--no-connectivity",
            "--no-conn-matrix",
        ):
            overrides.append("visualize.render_conn_matrix=false")
        else:
            overrides.append(arg)

    if checkpoint_path is None:
        raise ValueError("Specify checkpoint=<path>.")
    return checkpoint_path, save_video, target_pos, overrides


def resolve_run_dir(checkpoint_path: Path) -> Path | None:
    """Resolve the owning training run for a standard checkpoint path."""
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parents[1]
    return None


def evaluate_checkpoint(
    cfg,
    checkpoint_path: str | Path,
    *,
    run_dir: str | Path | None = None,
    save_video: bool = True,
    target_pos: tuple[float, float] | None = None,
) -> Path:
    """Restore and evaluate a checkpoint through the public runtime services."""
    checkpoint_path = Path(checkpoint_path).absolute()
    resolved_run_dir = (
        Path(run_dir).absolute()
        if run_dir is not None
        else resolve_run_dir(checkpoint_path)
    )
    if resolved_run_dir is None:
        resolved_run_dir = Path(
            cfg.logging.get("log_dir", "outputs")
        ).absolute()

    runtime = build_evaluation_runtime(cfg, checkpoint_path, rng_seed=0)
    model = runtime.model
    environment = runtime.environment
    eval_reset = environment.reset

    if target_pos is not None:
        target_x, target_y = target_pos

        def eval_reset(key):
            state = environment.reset(key)
            return state.replace(
                physics=state.physics.replace(
                    target_pos=jnp.array(
                        [target_x, target_y],
                        dtype=jnp.float32,
                    ),
                ),
            )

    key = jax.random.PRNGKey(0)
    _, metrics_key, video_key = jax.random.split(key, 3)
    num_envs = int(cfg.evaluation.eval_parallel_envs)
    result = evaluate_parallel(
        model,
        eval_reset,
        environment.env_step,
        environment.compute_obs,
        environment.compute_reward,
        cfg,
        metrics_key,
        num_envs=num_envs,
    )

    print(f"\nEvaluation over {num_envs} parallel episodes")
    print(f"  mean_return:  {float(jnp.mean(result.returns)):.2f}")
    print(f"  mean_length:  {float(jnp.mean(result.lengths)):.1f}")
    print(
        f"  chain_prog:   "
        f"{float(jnp.mean(result.chain_progress)):.1f}%"
    )
    print(f"  chain_gap:    {float(jnp.mean(result.chain_gaps)):.1f} m")
    print(f"  success_rate: {float(jnp.mean(result.successes)):.1%}")
    print(f"  target_found: {float(jnp.mean(result.target_found)):.1%}")

    artifact_root = eval_checkpoint_artifact_root(
        resolved_run_dir, checkpoint_path, cfg
    )
    artifact_tag = checkpoint_artifact_suffix(checkpoint_path, cfg)
    manifest_dir = artifact_root / "manifests"

    info_path = save_eval_info_csv(
        artifact_root / "data" / f"eval_info_{artifact_tag}.csv",
        target_positions=np.asarray(result.final_state.physics.target_pos),
        base_positions=np.asarray(result.final_state.physics.base_pos),
        successes=np.asarray(result.final_successes),
        delivered=np.asarray(result.final_delivered),
        visually_found=np.asarray(result.final_visually_found),
    )
    print(f"  eval_csv:     {info_path}")

    del result
    gc.collect()
    jax.clear_caches()

    if save_video and bool(cfg.evaluation.get("eval_video", True)):
        states, rewards, metrics = collect_video_episode(
            model,
            jax.jit(eval_reset),
            jax.jit(environment.env_step),
            jax.jit(environment.compute_obs),
            jax.jit(environment.compute_reward),
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
        release_video_evaluation_trajectory()
        print(f"  video:        {video_path}")

    return artifact_root


def main() -> None:
    checkpoint_path, save_video, target_pos, overrides = _parse_args()
    run_dir = resolve_run_dir(checkpoint_path)
    run_config = run_dir / "config.yaml" if run_dir is not None else None
    cfg = load_config(
        config_path=(
            run_config
            if run_config is not None and run_config.exists()
            else None
        ),
        cli_overrides=True,
        overrides=overrides,
    )
    validate_config(cfg)
    evaluate_checkpoint(
        cfg,
        checkpoint_path,
        run_dir=run_dir,
        save_video=save_video,
        target_pos=target_pos,
    )


if __name__ == "__main__":
    main()
