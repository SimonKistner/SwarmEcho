"""
training/evaluate_heatmap.py
============================
High-throughput parallel evaluation of SwarmEcho checkpoints in JAX.
Simulates many environments in parallel, logs failed target positions to a CSV,
and generates a heatmap image overlaying failures on the map blueprint.
Also supports selective rendering of failed targets directly from the CSV list.

Usage
-----
    # 1. Run parallel evaluation sweep to generate CSV and heatmap:
    uv run python src/training/evaluate_heatmap.py \
        checkpoint=outputs/my_run/checkpoints/ckpt_001000

    # 2. Render X failed target videos from the CSV (skipping already rendered ones):
    uv run python src/training/evaluate_heatmap.py \
        checkpoint=outputs/my_run/checkpoints/ckpt_001000 \
        --render-failed-csv=5
"""

import sys
import re
import csv
import time
from pathlib import Path
import dataclasses
import numpy as np
import cv2
import yaml
import jax
import jax.numpy as jnp
from flax import nnx
from omegaconf import OmegaConf

# Add project src root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import load_config, validate_config, compute_obs_dim, compute_action_dim
from env.physics import make_env_fns
from env.observations import make_obs_fns
from env.rewards import make_reward_fn
from models.mappo import MAPPOModel
from training.runner import _evaluate
from training.video_worker import render_eval_video
from env.maps import MapDefinition
from visualize.render_preview import render_png, _resolve_map_path

# ==============================================================================
# Configuration Constants
# ==============================================================================
NUM_ENVS = 4096             # Number of environments to evaluate in parallel
SEED = 42                   # Random seed for env reset and model initialization
HEATMAP_ALPHA = 0.4        # Transparency of overlay dots in the heatmap (0.0 to 1.0)
HEATMAP_DOT_RADIUS = 3     # Radius in pixels of overlay dots
SCALE = 8.0                # Resolution scale (pixels per world-meter) for map image
SHOW_SPAWN_ZONES = False     # Set to False to disable the red target/base spawn zones overlay


def main():
    # ── Parse CLI args ────────────────────────────────────────────────────
    args = sys.argv[1:]
    checkpoint_path = None
    render_failed_csv = None
    overrides = []

    for arg in args:
        if arg.startswith("checkpoint="):
            checkpoint_path = Path(arg.split("=", 1)[1].replace("\\", "/"))
        elif arg.startswith("--render-failed-csv="):
            render_failed_csv = int(arg.split("=", 1)[1])
        elif arg.startswith("render_failed_csv="):
            render_failed_csv = int(arg.split("=", 1)[1])
        else:
            overrides.append(arg)

    if checkpoint_path is None:
        print("ERROR: Must specify checkpoint=<path>")
        print("  e.g.  uv run python src/training/evaluate_heatmap.py checkpoint=outputs/my_run/checkpoints/ckpt_001000")
        sys.exit(1)

    # ── Resolve run directory early (needed for config auto-load) ─────────
    # Standard layout: outputs/<run_name>/checkpoints/ckpt_XXXXXX
    if checkpoint_path.parent.name == "checkpoints":
        run_dir = checkpoint_path.parents[1]
    else:
        run_dir = Path("outputs")

    # ── Config ────────────────────────────────────────────────────────────
    run_config_path = run_dir / "config.yaml"
    run_config_loaded = run_config_path.exists()
    cfg = load_config(
        config_path = run_config_path if run_config_loaded else None,
        cli_overrides = True,
        overrides = overrides,
    )
    validate_config(cfg)

    map_names = cfg.env.get("map_names", [])
    if not map_names:
        print("ERROR: No map specified in configuration.")
        sys.exit(1)
    map_name = map_names[0]

    # ── Environment ───────────────────────────────────────────────────────
    env_step, reset, _, (resolved_W, resolved_H, occ_grid, comm_occ_grid) = make_env_fns(cfg)
    compute_obs, _ = make_obs_fns(cfg, resolved_W, resolved_H, occ_grid, comm_occ_grid)
    compute_reward = make_reward_fn(cfg)

    # ── Build Model ───────────────────────────────────────────────────────
    obs_dim = compute_obs_dim(cfg)
    act_dim = compute_action_dim(cfg)
    N = int(cfg.env.num_agents)
    critic_type = str(cfg.network.critic_type)

    rngs = nnx.Rngs(SEED)
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
        memory_comm_msg_dim = cfg.network.get("memory_comm_msg_dim", None),
    )

    # ── Load Checkpoint ───────────────────────────────────────────────────
    import orbax.checkpoint as ocp
    graphdef, empty_state = nnx.split(model)
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    restored_state = checkpointer.restore(
        str(checkpoint_path.absolute()),
        args=ocp.args.StandardRestore(empty_state),
    )
    nnx.update(model, restored_state)
    print("Checkpoint loaded successfully ✓")

    # ── Select Mode ───────────────────────────────────────────────────────
    video_dir = run_dir / "videos" / "eval"
    video_dir.mkdir(parents=True, exist_ok=True)

    if render_failed_csv is not None:
        # ── Mode 1: Render specific failed target positions from CSV ──────────
        csv_candidates = list(video_dir.glob("failed_target_positions*.csv"))
        if not csv_candidates:
            print(f"ERROR: No failed target positions CSV found in: {video_dir}")
            sys.exit(1)
        csv_path = max(csv_candidates, key=lambda p: p.stat().st_mtime)
        print(f"Loading failed target positions from CSV: {csv_path.name}")

        # 1. Read failed target positions from CSV
        failed_positions = []
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)  # skip header row if present
            if header:
                try:
                    x, y = float(header[0]), float(header[1])
                    failed_positions.append((x, y))
                except ValueError:
                    pass  # it was indeed a text header
            for row in reader:
                if len(row) >= 2:
                    try:
                        failed_positions.append((float(row[0]), float(row[1])))
                    except ValueError:
                        continue

        print(f"Loaded {len(failed_positions)} failed target positions from CSV.")

        # 2. Identify already rendered positions in outputs/<run>/videos/eval/
        rendered_positions = set()
        if video_dir.exists():
            for f in video_dir.glob("FAIL_target_*.mp4"):
                match = re.match(r"FAIL_target_([0-9\.]+)_([0-9\.]+)\.mp4", f.name)
                if match:
                    rx = float(match.group(1))
                    ry = float(match.group(2))
                    rendered_positions.add((round(rx, 2), round(ry, 2)))

        # 3. Find candidates that have NOT been rendered yet
        to_render = []
        for pos in failed_positions:
            tx, ty = pos
            if (round(tx, 2), round(ty, 2)) not in rendered_positions:
                to_render.append(pos)
                if len(to_render) >= render_failed_csv:
                    break

        if not to_render:
            print("All failed targets from the CSV have already been rendered! ✓")
            return

        print(f"Identified {len(to_render)} new target position(s) to render.")

        # 4. Render each chosen target position
        renderer = str(cfg.visualize.get("final_eval_renderer", "slow"))
        key = jax.random.PRNGKey(SEED)

        for idx, (tx, ty) in enumerate(to_render):
            print(f"  [{idx+1}/{len(to_render)}] Simulating rollout for target: ({tx:.2f}, {ty:.2f})")

            # Wrapper reset_fn to override target_pos right after reset
            def make_override_reset(r_fn, target_x, target_y):
                def override_reset(k):
                    s = r_fn(k)
                    return s.replace(target_pos=jnp.array([target_x, target_y], dtype=jnp.float32))
                return override_reset

            custom_reset = make_override_reset(reset, tx, ty)

            # Run 1 evaluation episode
            (all_states, all_rewards, all_metrics,
             _, _, _, _, _, _) = _evaluate(
                model=model,
                reset_fn=jax.jit(custom_reset),
                env_step_fn=jax.jit(env_step),
                obs_fn=jax.jit(compute_obs),
                reward_fn=jax.jit(compute_reward),
                cfg=cfg,
                key=key,
                num_episodes=1,
            )

            # Render video
            filename_stem = f"FAIL_target_{tx:.2f}_{ty:.2f}"
            vid_path = render_eval_video(
                ep_states=all_states[0],
                ep_rewards=all_rewards[0],
                ep_metrics=all_metrics[0],
                cfg=cfg,
                out_dir=video_dir,
                filename_stem=filename_stem,
                renderer=renderer,
            )
            print(f"    ✓ Video saved: {vid_path}")

    else:
        # ── Mode 2: Run parallel evaluation sweep (Heatmap Mode) ──────────────
        print(f"Running parallel evaluation sweep over {NUM_ENVS} environments...")
        max_steps = int(cfg.env.max_steps)
        max_force = float(cfg.env.max_force)
        hold_chain_for = int(cfg.env.get("hold_chain_for", 0))

        # Vectorized functions
        env_step_fn_v = jax.vmap(env_step)
        reset_fn_v    = jax.vmap(reset)
        obs_fn_v      = jax.vmap(compute_obs)
        reward_fn_v   = jax.vmap(compute_reward)

        # Batch step helper
        def step_batch(state, actor_h, has_succeeded):
            obs = obs_fn_v(state)
            if model.actor_memory:
                resets = jnp.logical_not(state.active)
                def _act(o, h, r):
                    h, mu, _ = model.actor(o, h, r)
                    return h, mu
                actor_h, actions = jax.vmap(_act)(obs, actor_h, resets)
            else:
                actions = jax.vmap(lambda o: model.actor(o)[0])(obs)

            actions = jnp.tanh(actions) * max_force

            old_state = state
            next_state = env_step_fn_v(state, actions)

            # Compute rewards and success check
            dummy_dones = jnp.zeros(NUM_ENVS, dtype=jnp.bool_)
            _, info = reward_fn_v(old_state, next_state, dummy_dones)

            fully_connected = info["fully_connected"] > 0.5
            new_chain_held_steps = jnp.where(
                fully_connected,
                next_state.chain_held_steps + jnp.int32(1),
                jnp.int32(0)
            )
            next_state = next_state.replace(chain_held_steps=new_chain_held_steps)

            success_achieved = new_chain_held_steps >= (hold_chain_for + 1)
            new_has_succeeded = has_succeeded | success_achieved

            return next_state, actor_h, new_has_succeeded

        # Complete batch rollout compiled with lax.scan
        @jax.jit
        def run_rollout(state, actor_h):
            def scan_body(carry, _):
                state, actor_h, has_succeeded = carry
                next_state, next_actor_h, next_has_succeeded = step_batch(state, actor_h, has_succeeded)
                return (next_state, next_actor_h, next_has_succeeded), None

            init_carry = (state, actor_h, jnp.zeros(NUM_ENVS, dtype=jnp.bool_))
            (final_state, final_actor_h, final_has_succeeded), _ = jax.lax.scan(
                scan_body, init_carry, None, length=max_steps
            )
            return final_state, final_has_succeeded

        # Initial states
        master_key = jax.random.PRNGKey(SEED)
        env_keys = jax.random.split(master_key, NUM_ENVS)
        state = reset_fn_v(env_keys)

        if model.actor_memory:
            actor_h = model.initial_actor_hidden((NUM_ENVS,))
        else:
            actor_h = None

        print("Simulating parallel episodes (XLA compiled)...")
        start_time = time.time()
        final_state, final_has_succeeded = run_rollout(state, actor_h)
        final_has_succeeded.block_until_ready()
        elapsed = time.time() - start_time
        print(f"Simulation completed in {elapsed:.2f} seconds.")

        # Compute stats
        num_success = int(jnp.sum(final_has_succeeded))
        num_fail = NUM_ENVS - num_success
        success_rate = (num_success / NUM_ENVS) * 100.0
        print(f"Results: Successes: {num_success} | Failures: {num_fail} | Success Rate: {success_rate:.2f}%")

        # ── Export failed target positions to CSV ─────────────────────────────
        failed_mask = np.array(~final_has_succeeded)
        target_positions = np.array(final_state.target_pos)

        # Handle MEM_T8 where target_pos is (E, N, 2) instead of (E, 2)
        if target_positions.ndim == 3:
            target_positions = target_positions[:, 0, :]

        failed_target_positions = target_positions[failed_mask]

        csv_path = video_dir / "failed_target_positions.csv"
        heatmap_path = video_dir / "failed_targets_heatmap.png"

        # If any of the files already exist, add a timestamp solver to avoid overwriting
        if csv_path.exists() or heatmap_path.exists():
            ts = time.strftime("%Y%m%d_%H%M%S")
            csv_path = video_dir / f"failed_target_positions_{ts}.csv"
            heatmap_path = video_dir / f"failed_targets_heatmap_{ts}.png"
            print("Default CSV or Heatmap already exists. Saving with timestamp suffix solver.")

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])
            for pos in failed_target_positions:
                writer.writerow([f"{pos[0]:.6f}", f"{pos[1]:.6f}"])

        print(f"Saved {len(failed_target_positions)} failed target position(s) to: {csv_path}")

        # ── Draw and Overlay Heatmap ──────────────────────────────────────────
        print("Rendering blueprint and overlaying heatmap...")
        map_path = _resolve_map_path(map_name)
        if not map_path.exists():
            print(f"ERROR: Map definition not found at {map_path}")
            sys.exit(1)

        with open(map_path, "r") as f:
            map_data = yaml.safe_load(f)

        try:
            map_def = MapDefinition.load(map_path, cell_size=1.0)
        except Exception as e:
            print(f"Warning: Could not load MapDefinition: {e}")
            map_def = None

        # Re-use render_png with state=None and show_spawns=False for clean map layout
        background_img = render_png(
            data=map_data,
            map_def=map_def,
            state=None,
            show_zones=SHOW_SPAWN_ZONES,
            show_spawns=False,
            scale=SCALE,
        )

        overlay = background_img.copy()
        height = float(map_data["height"])

        for pos in failed_target_positions:
            tx, ty = pos[0], pos[1]
            px = int(tx * SCALE)
            py = int((height - ty) * SCALE)
            cv2.circle(overlay, (px, py), HEATMAP_DOT_RADIUS, (68, 68, 239), -1, cv2.LINE_AA)

        heatmap_img = cv2.addWeighted(overlay, HEATMAP_ALPHA, background_img, 1.0 - HEATMAP_ALPHA, 0)

        # Draw run description on heatmap
        info_str = f"Run: {run_dir.name} | Failures: {num_fail}/{NUM_ENVS} | Success Rate: {success_rate:.1f}%"
        cv2.putText(heatmap_img, info_str, (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.45, (55, 41, 31), 1, cv2.LINE_AA)

        cv2.imwrite(str(heatmap_path), heatmap_img)
        print(f"Saved failed targets heatmap to: {heatmap_path}")


if __name__ == "__main__":
    main()
