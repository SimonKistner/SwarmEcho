"""
scripts/test_physics.py
========================
Physics sanity check — runs a random-action rollout and saves a video.

This is the Step-3 gate: verify the physics visually BEFORE any RL.

Usage
-----
    # Full 200-step rollout → outputs/videos/physics_test.mp4
    uv run python scripts/test_physics.py

    # Custom length / config overrides:
    uv run python scripts/test_physics.py --steps 400 env.num_agents=12

    # Dry-run: init + 1 step, print VRAM stats, exit without video:
    uv run python scripts/test_physics.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure the repo root is on sys.path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp

from core.config import load_config, validate_config
from env.physics import make_env_fns
from env.rewards import make_reward_fn


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="SwarmEcho physics sanity check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--steps", type=int, default=200,
        help="Number of physics steps to roll out (default: 200)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="PRNG seed (default: 0)",
    )
    parser.add_argument(
        "--fps", type=int, default=20,
        help="Output video frame rate (default: 20)",
    )
    parser.add_argument(
        "--output", type=str, default="outputs/videos/physics_test.mp4",
        help="Output video path",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Init + 1 step only, print VRAM/shape stats, then exit",
    )
    # Any remaining args are OmegaConf overrides (e.g. env.num_agents=12)
    return parser.parse_known_args()


# ---------------------------------------------------------------------------
# VRAM stats helper
# ---------------------------------------------------------------------------

def _print_device_stats() -> None:
    devices = jax.local_devices()
    for dev in devices:
        stats = dev.memory_stats()
        if stats:
            used_mb  = stats.get("bytes_in_use", 0) / 1e6
            limit_mb = stats.get("bytes_limit", 0) / 1e6
            print(f"  [{dev}]  {used_mb:.1f} MB used / {limit_mb:.1f} MB available")
        else:
            print(f"  [{dev}]  memory stats not available")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args, extra_args = _parse_args()

    # -----------------------------------------------------------------------
    # Config (merges CLI overrides like env.num_agents=12)
    # -----------------------------------------------------------------------
    cfg = load_config(cli_overrides=bool(extra_args))
    validate_config(cfg)

    N = cfg.env.num_agents

    print("\n-- SwarmEcho Physics Test ------------------------------")
    print(f"  Devices     : {jax.devices()}")
    print(f"  num_agents  : {N}")
    print(f"  steps       : {args.steps}")
    print(f"  seed        : {args.seed}")
    print("────────────────────────────────────────────────────────\n")

    # -----------------------------------------------------------------------
    # Build env functions
    # -----------------------------------------------------------------------
    env_step, reset, _, (W, H, occ_grid) = make_env_fns(cfg)

    # JIT-compile the functions we need
    reset_jit = jax.jit(reset)
    step_jit  = jax.jit(env_step)

    key = jax.random.PRNGKey(args.seed)

    # -----------------------------------------------------------------------
    # DRY-RUN MODE
    # -----------------------------------------------------------------------
    if args.dry_run:
        print("=== DRY-RUN MODE — no video will be saved ===\n")

        state = reset_jit(key)
        state.pos.block_until_ready()   # force compilation now

        print("Initial state shapes:")
        print(f"  pos           : {state.pos.shape}  dtype={state.pos.dtype}")
        print(f"  vel           : {state.vel.shape}  dtype={state.vel.dtype}")
        print(f"  coverage_grid : {state.coverage_grid.shape}  dtype={state.coverage_grid.dtype}")
        print(f"  step          : {state.step.shape}  dtype={state.step.dtype}")

        actions = jax.random.uniform(
            jax.random.PRNGKey(1), (N, 2),
            minval=-cfg.env.max_force, maxval=cfg.env.max_force,
        )
        state2 = step_jit(state, actions)
        state2.pos.block_until_ready()

        print(f"\nAfter 1 step:")
        print(f"  pos[0]  : {state2.pos[0]}")
        print(f"  vel[0]  : {state2.vel[0]}")
        print(f"  covered : {int(state2.coverage_grid.sum())} / {G*G} cells")

        print("\nDevice memory after init + 1 step:")
        _print_device_stats()

        # Rough per-env state size estimate
        state_bytes = (
            N * 2 * 2 +    # pos + vel   (N, 2) each × 4 bytes
            2 + 2 +         # base_pos, target_pos
            G * G           # coverage_grid (bool = 1 byte)
        )
        state_bytes_f32 = state_bytes * 4
        print(f"\nEstimated state size per env : ~{state_bytes_f32 / 1024:.1f} KB")
        print(f"For {cfg.training.num_envs} envs               : ~{state_bytes_f32 * cfg.training.num_envs / 1e6:.1f} MB")
        print(f"Rollout buffer ({cfg.training.num_steps} steps) : ~{state_bytes_f32 * cfg.training.num_envs * cfg.training.num_steps / 1e6:.1f} MB")
        print("\nDry-run complete ✓")
        return

    # -----------------------------------------------------------------------
    # FULL ROLLOUT via lax.scan
    # -----------------------------------------------------------------------
    print("Step 1/3: Initialising state...")
    state = reset_jit(key)
    state.pos.block_until_ready()

    compute_reward = make_reward_fn(cfg)

    def scan_step(carry_state, scan_key):
        """One scan body: split key → random action → env step → compute reward → collect."""
        scan_key, subkey = jax.random.split(scan_key)
        actions = jax.random.uniform(
            subkey, (N, 2),
            minval=-cfg.env.max_force,
            maxval=cfg.env.max_force,
        )
        next_state = env_step(carry_state, actions)
        rew, _ = compute_reward(carry_state, next_state, jnp.bool_(False))
        # Output both the new state (carry) and a snapshot of state & reward (ys)
        return next_state, (next_state, rew)

    # Generate one key per step — scan iterates over the xs (keys)
    step_keys = jax.random.split(key, args.steps)

    print(f"Step 2/3: Running {args.steps}-step scan rollout on GPU...")
    t0 = time.perf_counter()
    scan_fn = jax.jit(lambda s, ks: jax.lax.scan(scan_step, s, ks))
    _final_state, (trajectory, rewards_traj) = scan_fn(state, step_keys)
    trajectory.pos.block_until_ready()
    elapsed = time.perf_counter() - t0

    steps_per_sec = args.steps / elapsed
    print(f"  Done in {elapsed:.2f}s  ({steps_per_sec:.0f} steps/sec)")
    print(f"  trajectory.pos shape : {trajectory.pos.shape}")
    print(f"  Final active         : {_final_state.active}")
    print(f"  Final target_known   : {_final_state.target_known}")
    # Cell size is hardcoded to 1.0 in physics.py
    grid_cells = int(W) * int(H)
    print(f"  Final coverage       : {int(_final_state.coverage_grid.sum())} / {grid_cells} cells "
          f"({100*_final_state.coverage_grid.sum()/grid_cells:.1f}%)")

    # -----------------------------------------------------------------------
    # Render
    # -----------------------------------------------------------------------
    print(f"\nStep 3/3: Rendering video...")

    # Import here so JAX init isn't delayed at the top of the file
    from visualize.renderer import render_video

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parents[1] / output_path

    saved_path = render_video(
        trajectory=trajectory,
        cfg=cfg,
        filename=output_path,
        fps=args.fps,
        rewards=rewards_traj,
    )

    print(f"\n✓ Physics test complete!")
    print(f"  Video saved → {saved_path}")
    print(f"\nOpen the video and verify:")
    print(f"  □ Drones move, bounce off walls")
    print(f"  □ Comm links (dashed) appear when agents are close")
    print(f"  □ Visual-radius halos visible around each drone")
    print(f"  □ Coverage overlay grows throughout the rollout")
    print(f"  □ Drone colours reflect connectivity to base/target")


if __name__ == "__main__":
    main()


