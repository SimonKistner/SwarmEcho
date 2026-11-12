"""Historical single-map dataset assembly using the shared rollout engine."""
import time
import numpy as np
from geo_experiments.shared.rollout import prepare_policy, make_collector
from geo_experiments.shared.common import sample_count, resolve_path
from geo_experiments.shared.artifacts import add_group, archive, group, save_data, update_run
from geo_experiments.shared.console import terminal_print as print
from geo_experiments.archive.target_location.config import SOURCES


def collect_dataset(model, level, config, run_dir, identity):
    import jax
    import jax.numpy as jnp
    from swarmecho.training.train import _validate_evaluation_pairs
    from geo_experiments.audit_observations import Audit

    d = config["dataset"]
    collect = make_collector(level)
    audit_config = config["audit"]
    audit = Audit() if audit_config["enabled"] and audit_config.get("mode", "full") == "full" else None
    audit_target = min(audit_config["samples"], sample_count(d)) if audit is not None else 0
    collect_history = make_collector(level, history_stride=audit_config["every"]) if audit is not None else None
    chunks, evaluation_chunks = [], []
    requested = sample_count(d)
    attempted = accepted = succeeded = missing = delivered = 0
    started = time.monotonic()
    while accepted < requested and attempted < d["max_episodes"]:
        auditing = audit is not None and audit.samples < audit_target
        count = min(d["eval_batch_size"], audit_config["batch_size"] if auditing else d["eval_batch_size"],
                    d["max_episodes"] - attempted)
        ids = np.arange(attempted, attempted + count, dtype=np.uint32)
        keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(d["seed"]), i))(jnp.asarray(ids))
        print(f"[collect] episodes {attempted + 1}–{attempted + count}; matched successes {accepted}/{requested}", flush=True)
        batch = jax.device_get((collect_history if auditing else collect)(model, keys))
        _validate_evaluation_pairs(level, batch["spawn_pair_attempts"])
        eligible = batch["delivered"] if config.get("protocol", {}).get("eligibility") == "delivered" else batch["success"]
        mask = eligible & batch["complete_sample"]
        needed = requested - accepted
        selected = np.flatnonzero(mask)[:needed]
        if auditing:
            for row in selected[:audit_target - audit.samples]:
                audit.add(batch, int(row), int(ids[row]), level.building.cell_size_m)
        chunks.append({**{name: np.asarray(value)[selected] for name, value in batch.items()
                          if not name.startswith("history_")},
                       "episode_id": ids[selected]})
        evaluation_chunks.append({"episode_id": ids, **{name: np.asarray(batch[name]) for name in
                                  ("success", "delivered", "target_required_agents", "complete_sample", "episode_length", "target_position")}})
        succeeded += int(batch["success"].sum())
        delivered += int(batch["delivered"].sum())
        missing += int((batch["success"] & ~batch["complete_sample"]).sum())
        attempted += count
        accepted += len(selected)
        update_run(run_dir, progress={"stage": "collection", "attempted": attempted, "accepted": accepted,
                   "successful_episodes": succeeded, "successful_missing_snapshot": missing,
                   "requested": requested, "elapsed_seconds": time.monotonic() - started})
    dataset = {name: np.concatenate([c[name] for c in chunks]) for name in chunks[0]}
    episodes = {name: np.concatenate([c[name] for c in evaluation_chunks]) for name in evaluation_chunks[0]}
    metadata = {"num_samples": accepted,
                "attempted_episodes": attempted, "successful_episodes": succeeded,
                "delivered_episodes": delivered,
                "eligibility": config.get("protocol", {}).get("eligibility", "success"),
                "success_rate": succeeded / attempted, "successful_missing_snapshot": missing,
                "collection_seconds": time.monotonic() - started,
                "input_dimensions": {s: int(dataset[s].shape[1]) for s in SOURCES},
                "message_layout": "signature followed by value",
                "finder_rule": "first direct visual sighting; smallest agent index breaks simultaneous ties",
                "snapshot_rule": "updated actor GRU/current emitted token on first communication tick after discovery",
                "base_rule": "first current emitted token stored by base, potentially from a relay"}
    audit_result = None
    if audit_config["enabled"] and audit_config.get("mode") == "conditional" and accepted == requested:
        from geo_experiments.shared.diagnostics.auditing import audit_targets
        temporary = {}
        temporary["geometry/cell_size_m"] = np.asarray(level.building.cell_size_m)
        add_group(temporary, "dataset", {**dataset, "map_id": np.zeros(accepted, dtype=np.int32)})
        summary = audit_targets(model, [level], config, temporary)
        audit_result = (summary, {**group(temporary, "audit"),
                        **{"pretest/" + k: v for k, v in group(temporary, "audit_pretest").items()},
                        **{"collisions/" + k: v for k, v in group(temporary, "audit_collisions").items()}})
    if audit is not None:
        summary = audit.summary()
        summary.update(settings={"samples": audit_target, "every": audit_config["every"],
                                 "batch_size": min(d["eval_batch_size"], audit_config["batch_size"])},
                       source={"kind": "integrated", "identity": identity,
                               "scope": "first matched episodes of this dataset"})
        audit_result = (summary, audit.arrays())
        print(f"[audit] {audit.samples} samples; {summary['distinct_grus']} exact GRUs; "
              f"{summary['mixed_cell_groups']} mixed-cell groups", flush=True)
    if accepted != requested:
        arrays = {}
        add_group(arrays, "dataset", dataset)
        add_group(arrays, "episodes", episodes)
        if audit_result is not None:
            add_group(arrays, "audit", audit_result[1])
        save_data(run_dir, arrays)
        update_run(run_dir, collection=metadata, ambiguity_audit=audit_result[0] if audit_result else None)
        raise RuntimeError(f"Collected only {accepted}/{requested} matched successful episodes in {attempted} attempts. "
                           "Partial data saved in data.npz; increase max_episodes or inspect policy success.")
    return dataset, metadata, episodes, audit_result


def reuse_dataset(path, config, run_dir, identity):
    import json
    from geo_experiments.shared.cache import snapshot_identity_matches

    path = resolve_path(path)
    manifest = json.loads((path.parent / "run.json").read_text(encoding="utf-8"))
    if not snapshot_identity_matches(manifest["provenance"]["identity"], identity):
        raise ValueError("Cached dataset checkpoint/level/map/collection identity does not match this run.")
    arrays = archive(path)
    dataset, episodes = group(arrays, "dataset"), group(arrays, "episodes")
    if len(dataset["episode_id"]) != sample_count(config["dataset"]):
        raise ValueError("Cached dataset size must match train + validation + test samples.")
    eligible = dataset["delivered"] if config.get("protocol", {}).get("eligibility") == "delivered" else dataset["success"]
    if not np.all(eligible & dataset["complete_sample"]):
        raise ValueError("Cached dataset contains failed episodes or missing event snapshots.")
    metadata = dict(manifest["collection"])
    metadata["reused_from"] = str(path)
    audit_result = None
    if config["audit"]["enabled"]:
        summary = manifest.get("ambiguity_audit")
        wanted = min(config["audit"]["samples"], sample_count(config["dataset"]))
        if config["audit"].get("mode") == "conditional":
            if summary and summary.get("format") == "swarmecho-target-audit/v2":
                from geo_experiments.shared.diagnostics.auditing import audit_is_current
                if not audit_is_current(summary, config, ("map_specific",)):
                    raise ValueError("Cached collection has a different source-specific audit pipeline/settings.")
                return dataset, metadata, episodes, (summary, group(arrays, "audit"))
            if (not summary or summary.get("format") != "swarmecho-conditional-audit/v1"
                    or summary["pretest"]["samples"] != wanted
                    or manifest["config"]["audit"] != config["audit"]):
                raise ValueError("Cached collection has no matching conditional latent screen.")
            return dataset, metadata, episodes, (summary, group(arrays, "audit"))
        if (not summary or summary.get("source", {}).get("kind") != "integrated"
                or summary.get("samples") != wanted
                or summary.get("settings", {}).get("every") != config["audit"]["every"]
                or not group(arrays, "audit")):
            raise ValueError("Cached data has no matching integrated ambiguity audit. "
                             "Collect fresh data with dataset.path=null, or set audit.enabled=false.")
        audit_result = (summary, group(arrays, "audit"))
    return dataset, metadata, episodes, audit_result
