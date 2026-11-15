"""Inspection 10: inject proposed poses and check actual runtime chain-hold success."""

from collections import Counter, defaultdict
from dataclasses import replace
from time import perf_counter

import numpy as np

from general_corner_simplification import Inspection as GeneralInspection
from remaining_corner_geometry import coordinates
from swarmecho.core.config import load_level


def make_group_kernel(prepared, level, agents, seed, device):
    import jax
    import jax.numpy as jnp
    from swarmecho.env.environment import make_env_fns
    from swarmecho.env.obstacles import points_inside_aabbs

    cfg = replace(level.env, num_agents=agents)
    reset, step, _, _ = make_env_fns(
        prepared.building, cfg, plan_geodesic=False,
        target_found_requires_delivery=level.reward.target_found_requires_delivery,
        allow_redundancy_reward=level.reward.allow_redundancy_reward,
        memory_comm_every_k_steps=level.network.memory_comm_every_k_steps,
        chain_reward_system=level.reward.chain_reward_system,
        delivery_gap_vesting_enabled=False,
    )
    with jax.default_device(device):
        template = reset(jax.random.PRNGKey(seed),
                         target_pos=jnp.asarray(prepared.targets[0], dtype=jnp.float32))
        base = jnp.asarray(prepared.base, dtype=jnp.float32)
        lower = jnp.asarray([prepared.building.wall_thickness_m / 2 + cfg.drone_radius] * 2 +
                            [prepared.building.tile_thickness_m / 2 + cfg.drone_radius], dtype=jnp.float32)
        upper = jnp.asarray(prepared.building.world_size_m, dtype=jnp.float32) - lower
        covered = jnp.ones_like(template.coverage, dtype=jnp.bool_)
        active = jnp.ones((agents,), dtype=jnp.bool_)
        zero_actions = jnp.zeros((agents, 3), dtype=jnp.float32)
        clock = max(0, (agents - 1) * cfg.spawn_delay)

    def evaluate_one(positions, target):
        valid = (jnp.all(jnp.isfinite(positions)) & jnp.all(positions >= lower) &
                 jnp.all(positions <= upper) & ~jnp.any(points_inside_aabbs(
                     positions, template.solid_min, template.solid_max, cfg.drone_radius)))
        state = template._replace(
            pos=positions, vel=jnp.zeros_like(positions), base_pos=base, target_pos=target,
            active=active, step=jnp.int32(clock), coverage=covered,
            directly_sees_target=jnp.zeros_like(active), is_conn_base=jnp.zeros_like(active),
            is_conn_target=jnp.zeros_like(active), target_known=jnp.zeros_like(active),
            base_target_known=jnp.bool_(False), success=jnp.bool_(False),
            fully_connected=jnp.bool_(False), chain_held_steps=jnp.int32(0),
            best_chain_held_steps=jnp.int32(0), stationary_steps=jnp.int32(0),
            done=jnp.bool_(False), idle_terminated=jnp.bool_(False),
            collided=jnp.zeros_like(active),
        )

        def hold_tick(carry, index):
            current, connected_all, collided_any, max_drift, first_connected, first_held, ticks = carry
            running = ~current.done
            next_state = jax.lax.cond(
                current.done, lambda value: value,
                lambda value: step(value._replace(coverage=covered), zero_actions), current,
            )
            connected_all &= (~running) | next_state.fully_connected
            collided_any |= running & jnp.any(next_state.collided)
            max_drift = jnp.maximum(max_drift, jnp.max(jnp.abs(next_state.pos - positions)))
            first_connected = jnp.where(index == 0, next_state.fully_connected, first_connected)
            first_held = jnp.where(index == 0, next_state.chain_held_steps, first_held)
            ticks += running.astype(jnp.int32)
            return (next_state, connected_all, collided_any, max_drift,
                    first_connected, first_held, ticks), None

        carry = (state, jnp.bool_(True), jnp.bool_(False), jnp.float32(0),
                 jnp.bool_(False), jnp.int32(0), jnp.int32(0))
        final, _ = jax.lax.scan(hold_tick, carry, jnp.arange(cfg.hold_chain_for, dtype=jnp.int32))
        state, connected_all, collided_any, drift, first_connected, first_held, ticks = final
        active_count = jnp.sum(state.active, dtype=jnp.int32)
        runtime_success = state.success & (state.chain_held_steps >= cfg.hold_chain_for)
        certified = (valid & first_connected & (first_held == 1) & connected_all & runtime_success &
                     ~collided_any & (drift == 0) & (active_count == agents) & jnp.all(state.active))
        return {
            "valid": valid, "first_connected": first_connected, "first_held": first_held,
            "connected_all": connected_all, "runtime_success": runtime_success,
            "success": state.success, "held": state.chain_held_steps,
            "certified": certified, "collision": collided_any, "drift": drift,
            "active_count": active_count, "ticks": ticks, "idle": state.idle_terminated,
            "done": state.done, "base_known": state.base_target_known,
            "final_pos": state.pos, "sees": state.directly_sees_target,
            "conn_base": state.is_conn_base, "conn_target": state.is_conn_target,
        }

    return jax.jit(jax.vmap(evaluate_one)), cfg, clock


def sample_ranges(indices):
    if not indices:
        return "none"
    ranges = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index == previous + 1:
            previous = index
        else:
            ranges.append(str(start) if start == previous else f"{start}-{previous}")
            start = previous = index
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ", ".join(ranges)


class Inspection:
    name = "runtime_chain_hold"

    def __init__(self, prepared, args):
        self.prepared, self.args = prepared, args
        self.level = load_level(args.level)
        self.simplifier = GeneralInspection(prepared, args)
        self.groups = defaultdict(list)
        self.sample = -1
        self.no_assignment = []
        self.no_assignment_causes = Counter()
        self.exit_code = 0

    def observe(self, points, target):
        self.sample += 1
        chain = self.simplifier.observe(points, target)
        if chain is None or not len(chain):
            self.no_assignment.append(self.sample)
            cause = "no roadmap/finder attachment" if points is None else "invalid rounded baseline chain"
            self.no_assignment_causes[cause] += 1
        else:
            self.groups[len(chain)].append((self.sample, target.copy(), chain.copy()))

    def report(self):
        import jax
        jax.config.update("jax_enable_x64", False)
        cfg = self.level.env
        print("\n--- Actual-runtime static chain-hold certification ---", flush=True)
        self.simplifier.report_guard()
        print("Inject proposed positions once; zero velocity/actions; hold counters begin at zero.", flush=True)
        if cfg.success_condition != "chain_held":
            raise ValueError("This static check requires the actual level's success_condition=chain_held.")
        if cfg.adapt_size.enabled or cfg.num_obstacles or self.level.reward.delivery_gap_vesting_enabled:
            raise ValueError("This check currently requires fixed population, no generated obstacles, and delivery vesting disabled.")
        if cfg.hold_chain_for < 1:
            raise ValueError("hold_chain_for must be positive.")
        geometry_fields = ("drone_radius", "obstacle_planning_clearance_m", "target_wall_buffer_fraction",
                           "comm_radius_base", "comm_radius", "visual_radius", "roadmap_approach",
                           "roadmap_node_density", "roadmap_merge_wall_end_nodes")
        if any(getattr(cfg, key) != getattr(self.prepared.cfg, key) for key in geometry_fields):
            raise ValueError("Full runtime level geometry settings differ from the simplifier settings.")
        try:
            devices = jax.devices(self.args.runtime_device)
        except RuntimeError as error:
            raise RuntimeError(f"Requested {self.args.runtime_device} is unavailable. "
                               "CPU must be explicitly selected with --runtime-device cpu.") from error
        if not devices:
            raise RuntimeError(f"No {self.args.runtime_device} device is available.")
        device = devices[0]
        print(f"JAX {jax.__version__}; device: {device.platform} / {device.device_kind}", flush=True)
        print(f"Precision: float32; hold requirement: {cfg.hold_chain_for} ticks; "
              f"idle limit: {cfg.no_movement_termination_steps}; episode limit: {cfg.max_steps}", flush=True)
        print(f"Batch size: {self.args.runtime_batch_size}; exact k drone slots per group; no padded drones.", flush=True)
        print("Coverage is fixed to full solely to omit unrelated coverage work; chain-held success is unchanged.")

        totals = Counter()
        reasons = Counter()
        failures = []
        failed_samples = list(self.no_assignment)
        summaries = []
        started = perf_counter()
        max_rounding = 0.0
        for agents, records in sorted(self.groups.items()):
            group_started = perf_counter()
            print(f"\nChecking k={agents}: {len(records):,} targets; preparing/compiling runtime kernel...", flush=True)
            with jax.default_device(device):
                kernel, group_cfg, clock = make_group_kernel(self.prepared, self.level, agents,
                                                            self.args.seed, device)
                group = Counter()
                for offset in range(0, len(records), self.args.runtime_batch_size):
                    batch = records[offset:offset + self.args.runtime_batch_size]
                    real_count = len(batch)
                    poses64 = np.stack([record[2] for record in batch])
                    targets64 = np.stack([record[1] for record in batch])
                    poses = poses64.astype(np.float32)
                    targets = targets64.astype(np.float32)
                    max_rounding = max(max_rounding, float(np.max(np.abs(poses64 - poses.astype(float)))),
                                       float(np.max(np.abs(targets64 - targets.astype(float)))))
                    padding = self.args.runtime_batch_size - real_count
                    if padding:
                        poses = np.concatenate((poses, np.repeat(poses[:1], padding, axis=0)))
                        targets = np.concatenate((targets, np.repeat(targets[:1], padding, axis=0)))
                    result = jax.device_get(kernel(jax.device_put(poses, device), jax.device_put(targets, device)))
                    for local, (sample, target64, chain64) in enumerate(batch):
                        item = {key: np.asarray(value[local]) for key, value in result.items()}
                        group["tested"] += 1
                        for key in ("first_connected", "runtime_success", "certified"):
                            group[key] += int(bool(item[key]))
                        if bool(item["certified"]):
                            continue
                        failed_samples.append(sample)
                        why = []
                        if not bool(item["valid"]):
                            why.append("invalid float32 physical placement")
                        if not bool(item["first_connected"]):
                            why.append("first runtime tick not fully connected")
                        if int(item["first_held"]) != 1:
                            why.append("first runtime hold counter not one")
                        if not bool(item["connected_all"]):
                            why.append("connectivity failed during hold window")
                        if not bool(item["runtime_success"]):
                            why.append("runtime chain-hold success not reached")
                        if bool(item["collision"]):
                            why.append("runtime collision")
                        if float(item["drift"]) != 0:
                            why.append("runtime changed injected positions")
                        if int(item["active_count"]) != agents:
                            why.append("active agent count mismatch")
                        reasons.update(why)
                        if len(failures) < self.args.geometry_details:
                            failures.append({"sample": sample, "agents": agents, "target64": target64,
                                             "chain64": chain64, "injected": poses[local].copy(),
                                             "target32": targets[local].copy(), "result": item, "why": why})
                totals.update(group)
                summaries.append((agents, len(records), group, clock))
            print(f"k={agents}: connected on first tick {group['first_connected']:,}/{len(records):,}; "
                  f"chain-hold success {group['runtime_success']:,}/{len(records):,}; "
                  f"strictly certified {group['certified']:,}/{len(records):,}; "
                  f"{perf_counter() - group_started:.2f}s", flush=True)

        print("\n Agents | Targets | First connected | Runtime hold success | Strictly certified")
        print("--------+---------+-----------------+----------------------+-------------------")
        for agents, count, group, clock in summaries:
            print(f"{agents:7d} | {count:7,d} | {group['first_connected']:15,d} | "
                  f"{group['runtime_success']:20,d} | {group['certified']:17,d}")
        certified = totals["certified"]
        print(f"\nCertified assignments: {certified:,}/{self.args.targets:,} "
              f"({100 * certified / self.args.targets:.2f}%)")
        print(f"Runtime hold successes: {totals['runtime_success']:,}/{totals['tested']:,} tested assignments")
        print(f"Targets with no proposed assignment: {len(self.no_assignment):,}")
        for cause, count in self.no_assignment_causes.items():
            print(f"  {cause}: {count:,}")
        print(f"Maximum coordinate rounding (poses and targets): {max_rounding * 1e6:.6f} micrometres")
        print("Activation clocks by k: " + ", ".join(f"{agents}:{clock}" for agents, _, _, clock in summaries))
        elapsed = perf_counter() - started
        print(f"Runtime checking time, including first compilation: {elapsed:.2f}s")
        if self.no_assignment:
            reasons["no proposed assignment"] += len(self.no_assignment)
        print("\nFailure reasons (may overlap):")
        for reason, count in reasons.most_common():
            print(f"  {reason}: {count:,}")
        if not reasons:
            print("  none")
        print("Failed zero-based sample indices: " + sample_ranges(sorted(failed_samples)))
        print(f"\n--- Failed assignment examples: up to {self.args.geometry_details} ---")
        for example in failures:
            item = example["result"]
            print(f"Sample {example['sample']}; k={example['agents']}; reasons=" + "; ".join(example["why"]))
            print(f"  Target float64={coordinates(example['target64'])}; "
                  f"runtime float32={coordinates(example['target32'])}")
            print(f"  First hold={int(item['first_held'])}; final hold={int(item['held'])}; "
                  f"success={bool(item['success'])}; done={bool(item['done'])}; "
                  f"idle={bool(item['idle'])}; executed ticks={int(item['ticks'])}; "
                  f"max drift={float(item['drift']):.9g}m")
            print(f"  Runtime sees={item['sees'].tolist()}; conn_base={item['conn_base'].tolist()}; "
                  f"conn_target={item['conn_target'].tolist()}")
            for index, (original, injected, final) in enumerate(zip(
                    example["chain64"], example["injected"], item["final_pos"]), 1):
                print(f"  Drone {index}: proposed64={coordinates(original)}")
                print(f"    injected32={coordinates(injected)}; final32={coordinates(final)}")
        if not failures:
            print("  none")
        self.exit_code = int(certified != self.args.targets)
        # Batch runners consume the actual counters, never parsed console text.
        self.summary = {
            "totals": totals.copy(), "reasons": reasons.copy(),
            "no_assignment": len(self.no_assignment),
            "groups": [(agents, count, group.copy()) for agents, count, group, _ in summaries],
            "failed_samples": sorted(failed_samples), "elapsed_s": elapsed,
        }
        print(f"\nStatus: {'all_sampled_assignments_certified' if not self.exit_code else 'runtime_assignment_failures'}")
        print("Certification uses the actual runtime step/success fields at exactly the proposed k, with no pose repinning.")
        print("This confirms placed sampled assignments only; navigation, robustness and learned-policy success are untested.")


if __name__ == "__main__":
    from run import main
    main(default_tests=[Inspection.name])
