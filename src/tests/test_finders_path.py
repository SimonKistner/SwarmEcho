"""Finder-path diagnostic harness for M02.

This is intentionally a runnable debugging script, not a pytest test.  It prints
all state needed to inspect why a target sighting would publish a bad global
finder path.

Examples:
    uv run python src/tests/test_finders_path.py --steps 20 --action random
    uv run python src/tests/test_finders_path.py --place-agent-at-target
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from core.config import load_config, validate_config, MAP_DIR, compute_obs_dim, compute_action_dim
from env.maps import MapDefinition
from env.physics import make_env_fns
from env.raycast import dda_raycast
from env.observations import make_obs_fns
from models.mappo import MAPPOModel


def _load_m02_cfg(config_path: Path | None = None):
    cfg = load_config(
        config_path=config_path or Path("src/curriculum_config/levels/M02_small_grid_maze.yaml"),
        cli_overrides=True,
        overrides=[
            "training.warn_vram_limit=false",
            "training.abort_on_vram_limit=false",
            "training.num_minibatches=16",
        ],
    )
    OmegaConf.set_readonly(cfg, False)
    cfg.training.warn_vram_limit = False
    cfg.training.abort_on_vram_limit = False
    OmegaConf.set_readonly(cfg, True)
    validate_config(cfg)
    return cfg



def _resolve_checkpoint(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    # Accept Windows-style paths pasted from the user's workstation.
    normalized = path_str.replace("\\", "/")
    if len(normalized) >= 2 and normalized[1] == ":":
        drive = normalized[0].lower()
        normalized = f"/mnt/{drive}" + normalized[2:]
    return Path(normalized).expanduser().resolve()


def _load_policy(checkpoint_path: Path, cfg, obs_dim: int, act_dim: int):
    from flax import nnx
    import orbax.checkpoint as ocp

    model = MAPPOModel(
        obs_dim=obs_dim,
        act_dim=act_dim,
        num_agents=int(cfg.env.num_agents),
        hidden_dim=int(cfg.network.hidden_dim),
        num_layers=int(cfg.network.num_layers),
        actor_num_layers=int(cfg.network.actor_num_layers),
        critic_type=str(cfg.network.critic_type),
        actor_memory=bool(cfg.network.get("actor_memory", False)),
        critic_memory=bool(cfg.network.get("critic_memory", False)),
        rngs=nnx.Rngs(0),
        memory_comm_enabled=bool(cfg.network.get("memory_comm_enabled", False)),
        memory_comm_gradient_mode=str(cfg.network.get("memory_comm_gradient_mode", "rial")),
        memory_comm_every_k_steps=int(cfg.network.get("memory_comm_every_k_steps", 5)),
        memory_comm_num_heads=int(cfg.network.get("memory_comm_num_heads", 4)),
        memory_comm_merge=str(cfg.network.get("memory_comm_merge", "residual")),
        memory_comm_attention_mode=str(cfg.network.get("memory_comm_attention_mode", "attend_global_learned_query")),
    )
    _, empty_state = nnx.split(model)
    restored_state = ocp.Checkpointer(ocp.StandardCheckpointHandler()).restore(
        str(checkpoint_path),
        args=ocp.args.StandardRestore(empty_state),
    )
    nnx.update(model, restored_state)
    return model


def _policy_actions(model, obs_fn, state, actor_h, base_memory, base_memory_valid, t: int, max_force: float):
    obs = obs_fn(state)
    if model.actor_memory:
        resets = jnp.logical_not(state.active)
        if model.memory_comm_enabled:
            n_eval = obs.shape[0]
            step_share = (int(model.memory_comm_every_k_steps) <= 1) or ((t % int(model.memory_comm_every_k_steps)) == 0)
            comm_mask = (
                state.adj_matrix[:n_eval, :n_eval]
                if state.adj_matrix.shape[-1]
                else jnp.zeros((n_eval, n_eval), dtype=bool)
            ) & jnp.asarray(step_share)
            base_mask = (
                (
                    state.adj_matrix[:n_eval, n_eval]
                    if state.adj_matrix.shape[-1]
                    else jnp.zeros((n_eval,), dtype=bool)
                )
                & ~state.target_known[:n_eval]
                & base_memory_valid
                & jnp.asarray(step_share)
            )
            actor_h, raw_actions, _ = model.actor.__call_team__(
                obs, actor_h, resets, comm_mask, state.active, base_memory, base_mask
            )
            connected_to_base = (
                state.adj_matrix[:n_eval, n_eval]
                if state.adj_matrix.shape[-1]
                else jnp.zeros((n_eval,), dtype=bool)
            )
            reporters = state.target_known & connected_to_base & state.active
            first_idx = jnp.argmax(reporters.astype(jnp.int32), axis=-1)
            has_reporter = jnp.any(reporters)
            reported_memory = actor_h[first_idx]
            should_store = has_reporter & ~base_memory_valid
            base_memory = jnp.where(should_store, reported_memory, base_memory)
            base_memory_valid = base_memory_valid | should_store
        else:
            actor_h, raw_actions, _ = jax.vmap(lambda o, h, r: model.actor(o, h, r))(obs, actor_h, resets)
    else:
        raw_actions = jax.vmap(lambda o: model.actor(o)[0])(obs)
    return jnp.tanh(raw_actions) * max_force, actor_h, base_memory, base_memory_valid

def _maze_helpers(map_def: MapDefinition):
    maze_cols = int(map_def.maze_cell_cols or 1)
    maze_rows = int(map_def.maze_cell_rows or 1)
    cell_w = float(map_def.width) / maze_cols
    cell_h = float(map_def.height) / maze_rows
    base_cell = np.array(
        [
            int(np.clip(np.floor((float(map_def.width) / 2.0) / cell_w), 0, maze_cols - 1)),
            int(np.clip(np.floor((float(map_def.height) / 2.0) / cell_h), 0, maze_rows - 1)),
        ],
        dtype=np.int16,
    )
    offsets = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.int16)
    start_cells = np.clip(base_cell[None, :] + offsets, [0, 0], [maze_cols - 1, maze_rows - 1])

    def cell_from_pos(pos):
        return np.array(
            [
                int(np.clip(np.floor(float(pos[0]) / cell_w), 0, maze_cols - 1)),
                int(np.clip(np.floor(float(pos[1]) / cell_h), 0, maze_rows - 1)),
            ],
            dtype=np.int16,
        )

    def is_start_cell(cell):
        return bool(np.any(np.all(start_cells == cell[None, :], axis=1)))

    return maze_cols, maze_rows, cell_w, cell_h, base_cell, start_cells, cell_from_pos, is_start_cell


def _directly_sees(state, cfg, occ_grid, cell_from_pos):
    del cell_from_pos
    target_pos = np.asarray(jax.device_get(state.target_pos))
    if target_pos.ndim == 1:
        target_pos_agents = np.repeat(target_pos[None, :], cfg.env.num_agents, axis=0)
    else:
        target_pos_agents = target_pos

    pos = np.asarray(jax.device_get(state.pos))
    active = np.asarray(jax.device_get(state.active), dtype=bool)
    sees = []
    for i in range(cfg.env.num_agents):
        dist = np.linalg.norm(pos[i] - target_pos_agents[i])
        in_range = bool(dist <= float(cfg.env.visual_radius) and active[i])
        los = bool(jax.device_get(dda_raycast(jnp.array(pos[i]), jnp.array(target_pos_agents[i]), occ_grid)))
        sees.append(in_range and los)
    return np.asarray(sees, dtype=bool), target_pos_agents


def _neighbor(a, b):
    a = np.asarray(a, dtype=np.int16)
    b = np.asarray(b, dtype=np.int16)
    return bool(abs(int(a[0]) - int(b[0])) <= 1 and abs(int(a[1]) - int(b[1])) <= 1)


def print_diagnostics(label, state, cfg, occ_grid, map_def, helpers):
    maze_cols, maze_rows, cell_w, cell_h, base_cell, start_cells, cell_from_pos, is_start_cell = helpers
    del maze_cols, maze_rows, cell_w, cell_h

    pos = np.asarray(jax.device_get(state.pos))
    target_known = np.asarray(jax.device_get(state.target_known), dtype=bool)
    active = np.asarray(jax.device_get(state.active), dtype=bool)
    path_lens = np.asarray(jax.device_get(state.finder_path_lens))
    path_active = np.asarray(jax.device_get(state.finder_path_active), dtype=bool)
    path_cells = np.asarray(jax.device_get(state.finder_path_cells))
    global_valid = bool(jax.device_get(state.finders_path_valid))
    global_len = int(jax.device_get(state.finders_path_len))
    direct, target_pos_agents = _directly_sees(state, cfg, occ_grid, cell_from_pos)
    cells = np.asarray([cell_from_pos(p) for p in pos])
    target_cells = np.asarray([cell_from_pos(t) for t in target_pos_agents])
    in_start = np.asarray([is_start_cell(c) for c in cells], dtype=bool)
    candidates = direct & (path_lens > 0)

    print("\n" + "=" * 96)
    print(label)
    print("=" * 96)
    print(f"step={int(jax.device_get(state.step))} target_pos={np.asarray(jax.device_get(state.target_pos))}")
    print(f"base_cell={base_cell.tolist()} code_start_cells={start_cells.tolist()}")
    print(f"map_target_exclude_zones={map_def.target_exclude_zones}")
    global_path = np.asarray(jax.device_get(state.finders_path))
    print(f"global_path_valid={global_valid} global_path_len={global_len} global_prefix={global_path[:min(global_len, 12)].tolist() if global_len > 0 else []}")
    print("idx active known direct candidate in_start cell tgt_cell len path_active last_cell prefix")

    for i in range(cfg.env.num_agents):
        length = int(path_lens[i])
        last_idx = max(length - 1, 0)
        last_cell = path_cells[i, last_idx].tolist() if path_cells.shape[1] else []
        prefix = path_cells[i, : min(length, 8)].tolist() if length > 0 else []
        print(
            f"{i:>3} {str(active[i]):>6} {str(target_known[i]):>5} {str(direct[i]):>6} "
            f"{str(candidates[i]):>9} {str(in_start[i]):>8} {cells[i].tolist()!s:>8} "
            f"{target_cells[i].tolist()!s:>8} {length:>3} {str(path_active[i]):>11} "
            f"{last_cell!s:>9} {prefix}"
        )

    if np.any(direct):
        old_idx = int(np.argmax(direct.astype(np.int32)))
        old_len = int(path_lens[old_idx])
        old_last = path_cells[old_idx, max(old_len - 1, 0)]
        print(
            "old_selection_by_directly_sees: "
            f"idx={old_idx} len={old_len} last={old_last.tolist()} "
            f"target_cell={target_cells[old_idx].tolist()} neighbor={_neighbor(old_last, target_cells[old_idx])}"
        )
    else:
        print("old_selection_by_directly_sees: no direct sighting")

    if np.any(candidates):
        new_idx = int(np.argmax(candidates.astype(np.int32)))
        new_len = int(path_lens[new_idx])
        new_last = path_cells[new_idx, max(new_len - 1, 0)]
        print(
            "new_selection_by_direct_and_len_gt_0: "
            f"idx={new_idx} len={new_len} last={new_last.tolist()} "
            f"target_cell={target_cells[new_idx].tolist()} neighbor={_neighbor(new_last, target_cells[new_idx])}"
        )
    else:
        print("new_selection_by_direct_and_len_gt_0: no ready finder candidate")



def _append_loop_erased_np(path, length, cell):
    valid_path = path[:length]
    matches = np.where(np.all(valid_path == cell[None, :], axis=1))[0]
    if matches.size:
        return path, int(matches[0]) + 1
    if length > 0 and np.array_equal(path[length - 1], cell):
        return path, length
    if length < path.shape[0]:
        path = path.copy()
        path[length] = cell
        return path, length + 1
    return path, length


def print_transition_diagnostics(label, before, after, cfg, occ_grid, map_def, helpers, emit: bool = True):
    """Mirror the finder-path update that env_step applies internally.

    The final EnvState intentionally keeps old per-agent path buffers once the
    global path is frozen, so inspecting only the returned state can be
    misleading. This mirrors the internal update using before.pos and after.pos.
    """
    _, _, _, _, base_cell, _, cell_from_pos, is_start_cell = helpers
    paths = np.asarray(jax.device_get(before.finder_path_cells)).copy()
    lens = np.asarray(jax.device_get(before.finder_path_lens)).astype(np.int32).copy()
    active_agents = np.asarray(jax.device_get(after.active), dtype=bool)
    known_before = np.asarray(jax.device_get(before.target_known), dtype=bool)
    prev_cells = np.asarray([cell_from_pos(p) for p in np.asarray(jax.device_get(before.pos))])
    mid_cells = np.asarray([cell_from_pos(p) for p in np.asarray(jax.device_get(after.pos))])
    in_start = np.asarray([is_start_cell(c) for c in mid_cells], dtype=bool)
    direct, target_pos_agents = _directly_sees(after, cfg, occ_grid, cell_from_pos)
    target_cells = np.asarray([cell_from_pos(t) for t in target_pos_agents])

    for i in range(cfg.env.num_agents):
        length = int(lens[i])
        was_tracking = length > 0
        now_tracking = was_tracking or ((not in_start[i]) and active_agents[i])
        reset_path = in_start[i] and was_tracking and (not known_before[i]) and (not direct[i])
        if (not was_tracking) and now_tracking:
            paths[i, 0] = base_cell
            paths[i, 1] = prev_cells[i]
            paths[i, 2] = mid_cells[i]
            length = 3
        if was_tracking and now_tracking and (not reset_path) and active_agents[i]:
            paths[i], length = _append_loop_erased_np(paths[i], length, mid_cells[i])
        if reset_path:
            paths[i] = 0
            length = 0
        lens[i] = length

    candidates = direct & (lens > 0)
    rows = []
    anomaly = False
    selected_summary = "candidate_selected_inside_env_step: none"
    if emit:
        print("\n" + "-" * 96)
        print(label)
        print("-" * 96)
        print("idx direct updated_len updated_last target_cell ready_neighbor updated_prefix")
    for i in range(cfg.env.num_agents):
        length = int(lens[i])
        last = paths[i, max(length - 1, 0)]
        row = (
            f"{i:>3} {str(direct[i]):>6} {length:>11} {last.tolist()!s:>12} "
            f"{target_cells[i].tolist()!s:>11} {str(bool(candidates[i] and _neighbor(last, target_cells[i]))):>14} "
            f"{paths[i, :min(length, 8)].tolist() if length > 0 else []}"
        )
        rows.append(row)
        if emit:
            print(row)
    if np.any(candidates):
        idx = int(np.argmax(candidates.astype(np.int32)))
        last = paths[idx, max(int(lens[idx]) - 1, 0)]
        neighbor = _neighbor(last, target_cells[idx])
        anomaly = not neighbor
        selected_summary = (
            f"candidate_selected_inside_env_step idx={idx} len={int(lens[idx])} "
            f"last={last.tolist()} target_cell={target_cells[idx].tolist()} neighbor={neighbor}"
        )
    if emit:
        print(selected_summary)
    return {"anomaly": anomaly, "selected_summary": selected_summary, "rows": rows}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--action", choices=("zero", "random", "policy"), default="zero")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path; Windows Q:\\... paths are mapped to /mnt/q/...")
    parser.add_argument("--diagnostics", choices=("all", "anomaly"), default="all", help="Use 'anomaly' to print compact steps until the warning condition is mirrored.")
    parser.add_argument("--stop-on-anomaly", action="store_true", help="Stop after printing the first mirrored warning/anomaly.")
    parser.add_argument(
        "--place-agent-at-target",
        action="store_true",
        help="Force agent 0 onto the sampled target before stepping; useful to inspect zero-length direct-sighting behavior.",
    )
    args = parser.parse_args()

    checkpoint_path = _resolve_checkpoint(args.checkpoint)
    run_config_path = None
    if checkpoint_path is not None and checkpoint_path.parent.name == "checkpoints":
        candidate = checkpoint_path.parents[1] / "config.yaml"
        if candidate.exists():
            run_config_path = candidate
    cfg = _load_m02_cfg(run_config_path)
    env_step, reset, _, (resolved_w, resolved_h, occ_grid, comm_occ_grid) = make_env_fns(cfg)
    obs_fn = None
    model = None
    actor_h = None
    base_memory = None
    base_memory_valid = jnp.bool_(False)
    if args.action == "policy":
        if checkpoint_path is None:
            raise SystemExit("--action policy requires --checkpoint <path>")
        obs_dim = compute_obs_dim(cfg)
        act_dim = compute_action_dim(cfg)
        obs_fn, _ = make_obs_fns(cfg, resolved_w, resolved_h, occ_grid, comm_occ_grid)
        model = _load_policy(checkpoint_path, cfg, obs_dim, act_dim)
    map_def = MapDefinition.load(MAP_DIR / "M02_small_grid_maze.yaml", cell_size=1.0)
    helpers = _maze_helpers(map_def)

    key = jax.random.PRNGKey(args.seed)
    state = reset(key)
    if model is not None and model.actor_memory:
        actor_h = model.initial_actor_hidden(())
        base_memory = jnp.zeros((model.hidden_dim,), dtype=jnp.float32)
    if args.place_agent_at_target:
        state = state.replace(
            pos=state.pos.at[0].set(state.target_pos),
            active=state.active.at[0].set(jnp.bool_(True)),
        )

    # JIT-compile key steps to prevent eager JAX overhead (critical for speed)
    env_step_jit = jax.jit(env_step)
    policy_step_jit = None
    if model is not None:
        @jax.jit
        def policy_step_jit(state, actor_h, base_memory, base_memory_valid, step):
            return _policy_actions(
                model, obs_fn, state, actor_h, base_memory, base_memory_valid, step, float(cfg.env.max_force)
            )

    if args.diagnostics == "all":
        print_diagnostics("initial", state, cfg, occ_grid, map_def, helpers)
    else:
        print(f"checkpoint={checkpoint_path} config={run_config_path or 'M02 defaults'} action={args.action} diagnostics=anomaly")
        print("step summary: global_valid global_len any_direct any_candidate selected_inside_env_step")

    for step in range(args.steps):
        if args.action == "policy":
            actions, actor_h, base_memory, base_memory_valid = policy_step_jit(
                state, actor_h, base_memory, base_memory_valid, step
            )
        elif args.action == "random":
            key, subkey = jax.random.split(key)
            actions = jax.random.normal(subkey, (cfg.env.num_agents, 2), dtype=jnp.float32)
        else:
            actions = jnp.zeros((cfg.env.num_agents, 2), dtype=jnp.float32)
        before = state
        state = env_step_jit(state, actions)
        if args.diagnostics == "all":
            transition = print_transition_diagnostics(f"mirrored internal finder update for env_step {step + 1}", before, state, cfg, occ_grid, map_def, helpers)
            print_diagnostics(f"after env_step {step + 1}", state, cfg, occ_grid, map_def, helpers)
        else:
            transition = print_transition_diagnostics(f"mirrored internal finder update for env_step {step + 1}", before, state, cfg, occ_grid, map_def, helpers, emit=False)
            direct, _ = _directly_sees(state, cfg, occ_grid, helpers[6])
            lens = np.asarray(jax.device_get(state.finder_path_lens))
            print(f"{step + 1:04d} {bool(jax.device_get(state.finders_path_valid))} {int(jax.device_get(state.finders_path_len))} {bool(np.any(direct))} {bool(np.any(direct & (lens > 0)))} {transition['selected_summary']}")
            if transition["anomaly"]:
                print("\nMirrored warning/anomaly detected; dumping detailed context.")
                print_diagnostics("before anomalous env_step", before, cfg, occ_grid, map_def, helpers)
                print_transition_diagnostics(f"mirrored internal finder update for env_step {step + 1}", before, state, cfg, occ_grid, map_def, helpers)
                print_diagnostics("after anomalous env_step", state, cfg, occ_grid, map_def, helpers)
                if args.stop_on_anomaly:
                    break
        if bool(jax.device_get(state.finders_path_valid)):
            print("global finder path became valid; stopping early")
            break


if __name__ == "__main__":
    main()
