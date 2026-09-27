"""Run only the random-building spawn-pair check used at training startup.

Use --manifest with a training run's random_buildings.json to replay its exact
map order even if the pool has since gained more maps. Round 1 uses the same
per-environment JAX keys as training's initial reset. Later rounds draw new
targets for a stress check; they do not replay a training rollout.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Match training's quiet JAX startup before importing JAX.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_CPP_MIN_VLOG_LEVEL", "0")
os.environ.setdefault("GLOG_minloglevel", "3")

import jax
import jax.numpy as jnp
import numpy as np

from swarmecho.core.config import load_level
from swarmecho.core.terminal import terminal_print
from swarmecho.curriculum_config.maps.random_building_bank.generate_bank import (
    _read_metadata,
    find_compatible_pool,
    load_records,
    required_settings,
    select_maps,
)
from swarmecho.env.building_bank import prepare_bank_from_records
from swarmecho.env.environment import make_autoreset_fns


def _load_selection(level, manifest_path: Path | None, pool_path: Path | None):
    training = level.training
    eval_count = (level.evaluation.random_eval_envs if level.evaluation.random_eval
                  and level.evaluation.random_eval_maps is None else 0)
    required = training.num_envs + eval_count

    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "pool" not in manifest:
            raise ValueError(f"{manifest_path} has no bank pool path; use a bank-backed training manifest.")
        pool = Path(manifest["pool"]).resolve()
        choices = manifest.get("maps")
        if not isinstance(choices, list) or len(choices) != training.num_envs:
            raise ValueError("Manifest map count differs from training.num_envs.")
    elif pool_path is not None:
        pool = pool_path.resolve()
        choices = None
    else:
        found = find_compatible_pool(level, required_count=required)
        if found is None:
            raise ValueError("No compatible, sufficiently large pool exists. Build one first or pass --manifest.")
        pool, metadata = found
        choices, _ = select_maps(pool, metadata, training.seed, training.num_envs, eval_count)

    metadata = _read_metadata(pool)
    if metadata is None:
        raise ValueError(f"Missing or invalid pool metadata: {pool / 'metadata.json'}")
    if metadata["settings"] != required_settings(level):
        raise ValueError("Pool roadmap or geometry settings differ from the level. Supply the same --level and --set overrides used for training.")
    if choices is None:
        if metadata["count"] < required:
            raise ValueError(f"Pool has {metadata['count']} maps but training needs {required} including evaluation.")
        choices, _ = select_maps(pool, metadata, training.seed, training.num_envs, eval_count)
    known = {item["id"]: item for item in metadata["maps"]}
    for item in choices:
        if item.get("id") not in known or item.get("seed") != known[item["id"]]["seed"]:
            raise ValueError(f"Manifest map is absent or changed in the pool: {item!r}")
    return pool, metadata, choices


def _base_story_summary(record, cell_size):
    centres_z = (record["base_lower"][:, 2] + record["base_upper"][:, 2]) / 2
    stories = np.floor(centres_z / cell_size).astype(int)
    volumes = record["base_volumes"]
    total = float(volumes.sum())
    return ", ".join(
        f"story {story + 1}: {int(np.count_nonzero(stories == story))} boxes, "
        f"{100 * float(volumes[stories == story].sum()) / total:.1f}% of sampled volume"
        for story in np.unique(stories) if total > 0
    )


def check(level, *, manifest_path=None, pool_path=None, rounds=1, max_report=20):
    if not level.random_buildings.enabled or not level.random_buildings.load_maps_from_bank:
        raise ValueError("This checker requires random_buildings.enabled and load_maps_from_bank to be true.")
    if rounds < 1 or max_report < 0:
        raise ValueError("--rounds must be positive and --max-report must be nonnegative.")

    pool, metadata, choices = _load_selection(level, manifest_path, pool_path)
    training, cfg = level.training, level.env
    threshold = (training.minimum_geodesic_separation_multiplier * cfg.comm_radius
                 if training.minimum_geodesic_separation else 0.0)
    terminal_print(f"Pool: {pool}; selected {len(choices)} training maps from {metadata['count']}")
    skip_targets = (training.skip_on_no_pair_found and training.randomize_base
                    and training.minimum_geodesic_separation)
    terminal_print(f"Spawn rule: {training.spawn_pair_max_attempts} base attempts per target; "
                   f"separation >= {threshold:g} m; corner bonus {cfg.roadmap_corner_bonus_m:g} m; "
                   f"randomized base={training.randomize_base}; skip exhausted target={skip_targets}")
    records = load_records(pool, choices, require_corner=bool(metadata["settings"]["roadmap"]["corner_bonus_m"]))
    bank = prepare_bank_from_records(records, level.building, cfg, log=terminal_print)
    reset, _, _, _ = make_autoreset_fns(
        level.building, cfg, level.reward,
        randomize_base=training.randomize_base,
        minimum_geodesic_separation=training.minimum_geodesic_separation,
        minimum_geodesic_separation_multiplier=training.minimum_geodesic_separation_multiplier,
        skip_on_no_pair_found=training.skip_on_no_pair_found,
        spawn_pair_max_attempts=training.spawn_pair_max_attempts,
        building_bank=bank,
    )

    @jax.jit
    def reset_batch(keys, map_ids):
        states = jax.vmap(lambda key, index: reset(key, map_id=index))(keys, map_ids)
        return (states.spawn_pair_attempts, states.spawn_pair_target_retries,
                states.target_pos, states.base_pos)

    failures = 0
    reported = 0
    for round_index in range(rounds):
        root_key = jax.random.PRNGKey(training.seed + 1)
        if round_index:
            root_key = jax.random.fold_in(root_key, round_index)
        keys = jax.random.split(root_key, training.num_envs)
        attempts_all = []
        target_retries_all = []
        round_failures = 0
        for start in range(0, training.num_envs, 32):
            stop = min(start + 32, training.num_envs)
            indices = jnp.minimum(jnp.arange(start, start + 32), training.num_envs - 1)
            batch_keys = keys[indices]
            attempts, target_retries, targets, bases = reset_batch(batch_keys, indices)
            attempts, target_retries, targets, bases = (np.asarray(value)[:stop - start]
                                                        for value in (attempts, target_retries, targets, bases))
            attempts_all.extend(attempts.tolist())
            target_retries_all.extend(target_retries.tolist())
            for local in np.flatnonzero(attempts == 0):
                index = start + int(local)
                round_failures += 1
                if reported < max_report:
                    item, record = choices[index], records[index]
                    target = targets[local]
                    story = min(int(target[2] // level.building.cell_size_m) + 1,
                                level.random_buildings.stories)
                    terminal_print(f"[FAIL] round={round_index + 1} env={index} map={item['id']} "
                                   f"seed={item['seed']} target={target.tolist()} story={story} "
                                   f"last_rejected_base={bases[local].tolist()}")
                    terminal_print(f"       base sampling: {_base_story_summary(record, level.building.cell_size_m)}")
                    terminal_print(f"       map file: {pool / 'maps' / (item['id'] + '.yaml')}")
                    reported += 1
            if stop == training.num_envs or stop % 512 == 0:
                terminal_print(f"Round {round_index + 1}: checked {stop}/{training.num_envs}; failures {round_failures}")
        successful = np.asarray(attempts_all, dtype=np.int32)
        positive = successful[successful > 0]
        target_retries = np.asarray(target_retries_all, dtype=np.int32)
        terminal_print(f"Round {round_index + 1}: {round_failures}/{training.num_envs} failed; "
                       f"successful attempts median={np.median(positive):.0f}, "
                       f"p95={np.percentile(positive, 95):.0f}, max={positive.max()}; "
                       f"skipped targets total={target_retries.sum()}, "
                       f"resets with skips={np.count_nonzero(target_retries)}/{training.num_envs}, "
                       f"max skips/reset={target_retries.max()}" if len(positive)
                       else f"Round {round_index + 1}: all {training.num_envs} failed")
        failures += round_failures
    terminal_print(f"Total: {failures} failed resets across {rounds} round(s). "
                   f"{'Training would raise the spawn-pair error.' if failures else 'No spawn-pair error observed.'}")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", default="B02b_random_buildings_deliver")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Repeat the exact level overrides passed to training")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--manifest", type=Path, help="Training run's random_buildings.json for its exact map order")
    source.add_argument("--pool", type=Path, help="Select maps from this compatible pool")
    parser.add_argument("--rounds", type=int, default=1,
                        help="1 replays the initial reset; additional rounds stress new targets")
    parser.add_argument("--max-report", type=int, default=20, help="Maximum individual failures to print")
    args = parser.parse_args()
    try:
        failures = check(load_level(args.level, args.set), manifest_path=args.manifest,
                         pool_path=args.pool, rounds=args.rounds, max_report=args.max_report)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Spawn-pair checker setup failed: {exc}\n")
    if failures:
        parser.exit(1)


if __name__ == "__main__":
    main()
