"""Compare production JAX queries to the reference and certify their actual poses."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from dataclasses import replace

import numpy as np

from paths import prepare_paths, building_roadmap
from reporting import capture_console
from run_maps import numbered_maps, runtime_table
from runtime_chain_hold import Inspection


def reset_smoke(prepared, args, device):
    """Exercise actual filtered reset with each adaptive-spawn/population combination."""
    import jax
    import jax.numpy as jnp
    from swarmecho.core.config import load_level
    from swarmecho.env.environment import make_env_fns
    from swarmecho.env.relay import make_calculator
    from swarmecho.training.adaptive_spawn import AdaptiveTargetSpawnController

    level = load_level(args.level)
    graph = building_roadmap(prepared.building, prepared.cfg, corner_bonus_m=0.)
    cfg = replace(level.env, num_agents=2)
    calculate = make_calculator(cfg, graph, prepared.building)
    verify = jax.jit(jax.vmap(lambda base, target: calculate(base, target)))
    print("\n--- Actual reset compatibility: budget 2, four adaptive combinations ---", flush=True)
    for weighted in (False, True):
        for adaptive in (False, True):
            config = replace(cfg, adapt_size=replace(cfg.adapt_size, enabled=adaptive,
                                                     initial_agents=min(cfg.adapt_size.initial_agents, 2)))
            reset, _, _, _ = make_env_fns(prepared.building, config, dynamic_max_geo_distance=True,
                spawn_pair_max_attempts=level.training.spawn_pair_max_attempts,
                target_found_requires_delivery=level.reward.target_found_requires_delivery,
                memory_comm_every_k_steps=level.network.memory_comm_every_k_steps,
                chain_reward_system=level.reward.chain_reward_system)
            weights = None
            if weighted:
                controller = AdaptiveTargetSpawnController(prepared.building, config,
                    level.training.adaptive_target_spawn, feasibility_budget=2)
                discovery_keys = jax.random.split(jax.random.PRNGKey(args.seed + 200 + int(adaptive)), 4)
                discovery = jax.jit(jax.vmap(reset))(discovery_keys)
                if np.any(np.asarray(jax.device_get(discovery.spawn_pair_attempts)) == 0):
                    raise RuntimeError("Uniform startup reset failed before adaptive eligibility initialization.")
                controller.initialize_eligibility(jax.device_get(discovery.target_pos))
                controller.rates = np.linspace(0., .9, len(controller.cells))
                controller.advance(1, 0)
                weights = jnp.asarray(controller.box_weights)
            keys = jax.random.split(jax.random.PRNGKey(args.seed + 100 + int(weighted) * 2 + int(adaptive)), 4)
            started = perf_counter()
            states = jax.jit(jax.vmap(lambda key: reset(key, target_spawn_weights=weights)))(keys)
            counts, valid = jax.device_get(verify(states.base_pos, states.target_pos))
            attempts = np.asarray(jax.device_get(states.spawn_pair_attempts))
            if np.any(attempts == 0) or not np.all(valid & (counts <= 2)):
                raise RuntimeError(f"Filtered reset failed with weighted={weighted}, adaptive={adaptive}.")
            if adaptive and not np.all(np.asarray(jax.device_get(states.adaptive.zone_valid))):
                raise RuntimeError("Reference map has an invalid decommission zone.")
            print(f"adaptive spawn={weighted}; adaptive size={adaptive}: 4/4 certified pairs <=2; "
                  f"attempts={attempts.tolist()}; {perf_counter() - started:.2f}s including compilation", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", nargs="+")
    parser.add_argument("--targets", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=456789)
    parser.add_argument("--level", default="G01_static")
    parser.add_argument("--runtime-batch-size", type=int, default=16)
    parser.add_argument("--geometry-details", type=int, default=5)
    parser.add_argument("--reset-smoke", action="store_true",
                        help="Also exercise real reset with adaptive spawning and adaptive size on the first map.")
    args = parser.parse_args()
    if args.targets < 1 or args.runtime_batch_size < 1 or args.geometry_details < 0:
        parser.error("Targets/batch size must be positive; details must be nonnegative.")
    args.runtime_device = "gpu"
    names = args.maps or numbered_maps()
    if not names:
        parser.error("No selected maps.")
    results = Path(__file__).resolve().parent / "results"
    results.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    archive = results / f"{stamp}_13_production_queries.txt"
    print(f"Transcript: {archive}")
    runs, regressions = [], 0
    with archive.open("x", encoding="utf-8", newline="\n") as handle:
        _, capture = capture_console(handle)
        with capture:
            try:
                import jax
                import jax.numpy as jnp
                from swarmecho.env.relay import make_calculator
                jax.config.update("jax_enable_x64", False)
                device = jax.devices("gpu")[0]
                print(f"Production JAX query / CPU reference / independent real hold; device={device.device_kind}")
                print(f"Maps={names}; targets/map={args.targets}; seed={args.seed}; level={args.level}")
                for map_index, name in enumerate(names):
                    args.map = name
                    prepared = prepare_paths(args)
                    inspection = Inspection(prepared, argparse.Namespace(**vars(args)))
                    if inspection.level.env.adapt_size.enabled:
                        # The placement certificate deliberately keeps exactly k
                        # agents alive. Population protocol is checked separately
                        # by --reset-smoke, using the unmodified level settings.
                        inspection.level = replace(inspection.level, env=replace(inspection.level.env,
                            adapt_size=replace(inspection.level.env.adapt_size, enabled=False,
                                               reward_by_reference_enabled=False)))
                        print("Hold certificate: fixed population at exactly k; adaptive reset is checked separately by --reset-smoke.")
                    for points, target in prepared.iter_targets():
                        inspection.observe(points, target)
                    prepared.report_routes(args.geometry_details)
                    reference = np.zeros(args.targets, dtype=int)
                    for agents, records in inspection.groups.items():
                        for sample, _, _ in records:
                            reference[sample] = agents
                    graph = building_roadmap(prepared.building, prepared.cfg, corner_bonus_m=0.)
                    with jax.default_device(device):
                        query = make_calculator(prepared.cfg, graph, prepared.building, return_witness=True)
                        kernel = jax.jit(jax.vmap(lambda target: query(jnp.asarray(prepared.base, jnp.float32), target)))
                        inspection.groups = defaultdict(list)
                        inspection.no_assignment = []
                        inspection.no_assignment_causes = Counter()
                        predicted = np.zeros(args.targets, dtype=int)
                        started = perf_counter()
                        for offset in range(0, args.targets, args.runtime_batch_size):
                            targets = prepared.targets[offset:offset + args.runtime_batch_size]
                            real = len(targets)
                            if real < args.runtime_batch_size:
                                targets = np.concatenate((targets, np.repeat(targets[:1], args.runtime_batch_size - real, axis=0)))
                            routes, counts, valid = jax.device_get(kernel(jnp.asarray(targets, jnp.float32)))
                            for local in range(real):
                                sample = offset + local
                                if valid[local]:
                                    agents = int(counts[local])
                                    predicted[sample] = agents
                                    inspection.groups[agents].append((sample, prepared.targets[sample].copy(),
                                                                     np.asarray(routes[local, 1:agents + 1], dtype=float)))
                                else:
                                    inspection.no_assignment.append(sample)
                                    inspection.no_assignment_causes["production query unresolved"] += 1
                        print(f"\n{name}: production queries including first compilation: {perf_counter() - started:.2f}s", flush=True)
                        # Repeat queries without compilation to separate execution cost.
                        warm_started = perf_counter()
                        probe = prepared.targets[:min(args.targets, args.runtime_batch_size)]
                        if len(probe) < args.runtime_batch_size:
                            probe = np.concatenate((probe, np.repeat(probe[:1], args.runtime_batch_size - len(probe), axis=0)))
                        jax.block_until_ready(kernel(jnp.asarray(probe, jnp.float32)))
                        print(f"Warm batch ({args.runtime_batch_size} queries): {perf_counter() - warm_started:.6f}s")
                    both = (reference > 0) & (predicted > 0)
                    lost = (reference > 0) & (predicted == 0)
                    regressions += int(lost.sum())
                    print(f"Assignments: CPU={np.count_nonzero(reference):,}; JAX={np.count_nonzero(predicted):,}; "
                          f"CPU-valid lost={lost.sum():,}; newly recovered={np.sum((reference == 0) & (predicted > 0)):,}")
                    print(f"Counts among common assignments: equal={np.sum(both & (reference == predicted)):,}; "
                          f"JAX fewer={np.sum(both & (predicted < reference)):,}; JAX more={np.sum(both & (predicted > reference)):,}")
                    for budget in range(2, max(7, int(predicted.max(initial=0))) + 1):
                        print(f"  Budget {budget}: eligible CPU={np.sum((reference > 0) & (reference <= budget)):,}; "
                              f"JAX={np.sum((predicted > 0) & (predicted <= budget)):,}")
                    print("The following guard counters describe the CPU reference; the actual checked poses are production JAX witnesses.")
                    inspection.report()
                    if args.reset_smoke and map_index == 0:
                        with jax.default_device(device):
                            reset_smoke(prepared, args, device)
                    runs.append((name, inspection))
                    jax.clear_caches()
                runtime_table(runs)
                print(f"CPU-valid assignments lost by production query: {regressions:,}")
                print("Unresolved pairs are rejected, never accepted as certified; known unresolved CPU pairs do not fail the port gate.")
                bad = sum(inspection.summary["totals"]["tested"] - inspection.summary["totals"]["certified"]
                          for _, inspection in runs)
                print("Production port gate: " + ("FAILED" if regressions or bad else "passed; unresolved pairs remain rejected"))
            except Exception as error:
                print(f"\nIntegration validation error: {type(error).__name__}: {error}", flush=True)
                raise
    if regressions or any(inspection.summary["totals"]["certified"] != inspection.summary["totals"]["tested"]
                          for _, inspection in runs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
