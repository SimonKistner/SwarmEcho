"""Cheap exact-latent screen; full input histories only when the screen warrants it."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from geo_experiments.artifacts import add_group, group
from geo_experiments.audit_observations import STAGES, cross_cell_pairs, digest
from geo_experiments.common import SOURCES
from geo_experiments.common import fingerprint
from geo_experiments.collisions import analyse_collisions, ambiguous_rows, locations, vector_groups
from geo_experiments.console import terminal_print as print

FORMAT = "swarmecho-target-audit/v2"


def screen_latents(dataset, settings, seed):
    count = min(settings["samples"], len(dataset["finder_gru"]))
    indices = np.sort(np.random.default_rng(seed).choice(len(dataset["finder_gru"]), count, replace=False))
    inputs = {}
    for source in SOURCES:
        counts = Counter(digest(row) for row in dataset[source][indices])
        repeated = sum(n for n in counts.values() if n > 1)
        inputs[source] = {"distinct": len(counts), "same_latent_samples": repeated,
                          "same_latent_fraction": repeated / count if count else 0}
    return {"format": "swarmecho-latent-screen/v1", "samples": count, "inputs": inputs,
            "definition": "Share of sampled episodes belonging to an exact duplicate latent group (including its first member)",
            "threshold": settings["duplicate_threshold"]}, indices


def conflict_summary(records, decoder):
    """Interpret the same replayed vectors under each readout's information contract."""
    inputs = {}
    for source, original_groups in records.items():
        groups = defaultdict(list)
        for latent, rows in original_groups.items():
            for cell, hashes in rows:
                # Experts are routed externally: cross-map equality cannot make
                # two labels ambiguous for a single expert. Shared-coordinate
                # readouts do not require the building identity in their label.
                key = (cell[0], latent) if decoder == "map_specific" else latent
                label = cell[1:] if decoder == "shared_cell" else cell
                groups[key].append((label, hashes))
        pairs = mixed = samples = 0
        matches = dict.fromkeys(STAGES, 0)
        for rows in groups.values():
            n = cross_cell_pairs(Counter(cell for cell, _ in rows))
            pairs += n
            mixed += int(n > 0)
            samples += len(rows) if n else 0
            if source != "base_message":
                for stage, name in enumerate(STAGES):
                    clusters = defaultdict(Counter)
                    for cell, hashes in rows:
                        clusters[hashes[stage]][cell] += 1
                    matches[name] += sum(cross_cell_pairs(c) for c in clusters.values())
        inputs[source] = {"distinct_states": len(groups), "mixed_cell_groups": mixed,
                          "mixed_cell_samples": samples, "cross_cell_pairs": pairs,
                          "history_matches": matches if source != "base_message" else None}
    return inputs


def pipeline_identity(config, decoders):
    decoders = ("map_specific" if d == "legacy" else d for d in decoders)
    return {"format": FORMAT, "settings": dict(config["audit"]), "decoders": sorted(set(decoders)),
            "sources": {name: fingerprint(Path(__file__).with_name(name))
                        for name in ("auditing.py", "collisions.py", "audit_observations.py")}}


def audit_is_current(summary, config, decoders):
    if not config["audit"]["enabled"]:
        return summary is None
    return bool(summary and summary.get("pipeline") == pipeline_identity(config, decoders)
                and summary.get("history_status") != "pending")


def plan_audit(config, arrays, *, decoders=None):
    """All source routing and snapshot analysis runs on already collected arrays."""
    settings = config["audit"]
    if not settings["enabled"]:
        return None, np.empty(0, dtype=np.int32)
    for key in list(arrays):
        if key.startswith(("audit/", "audit_pretest/", "audit_collisions/")):
            del arrays[key]
    decoders = tuple(decoders or (config.get("protocol", {}).get("decoder", "map_specific"),))
    decoders = tuple("map_specific" if d == "legacy" else d for d in decoders)
    dataset = group(arrays, "dataset")
    screen, indices = screen_latents(dataset, settings, config["dataset"]["split_seed"])
    add_group(arrays, "audit_pretest", {"sample_indices": indices})
    cell_size = float(arrays["geometry/cell_size_m"])
    location = locations(dataset, cell_size)
    analyses = {}
    candidates = np.empty(0, dtype=np.int32)
    for source in SOURCES:
        row = screen["inputs"][source]
        triggered = settings["mode"] == "full" or (row["same_latent_samples"] > 0
                    and row["same_latent_fraction"] >= settings["duplicate_threshold"])
        row["triggered"] = triggered
        if not triggered:
            continue
        analysis, members = analyse_collisions(dataset, source, cell_size)
        analyses[source] = analysis
        add_group(arrays, f"audit_collisions/{source}", members)
        print(f"[audit] {source}: {analysis['repeated_samples']:,}/{analysis['samples']:,} samples in "
              f"{analysis['repeated_groups']:,} repeated-vector groups (saved data)")
        if source == "finder_gru":
            # Only ambiguous GRU groups found by the screen justify input-history
            # investigation. Packet collisions never route to finder histories.
            candidates = ambiguous_rows(vector_groups(dataset[source], indices), location, decoders)
    add_group(arrays, "audit_pretest", {"finder_history_indices": candidates})
    result = {"format": FORMAT, "pipeline": pipeline_identity(config, decoders), "pretest": screen,
              "snapshot_analyses": analyses, "history": None,
              "map_names": arrays.get("geometry/map_names", np.asarray([config["policy"]["level"]])).tolist(),
              "history_status": "pending" if len(candidates) else "not_required",
              "history_requested_samples": len(candidates)}
    if not len(candidates):
        print("[audit] finder observation histories skipped: no threshold-triggering finder-GRU groups with conflicting labels")
    return result, candidates


def audit_targets(model, levels, config, arrays, *, decoders=None):
    result, indices = plan_audit(config, arrays, decoders=decoders)
    if result is None or not len(indices):
        return result
    if model is None or levels is None:
        raise ValueError("Finder-GRU conflicts require history replay in the policy runtime; snapshot-only audits do not.")
    settings = config["audit"]
    dataset = group(arrays, "dataset")
    result["history"] = investigate_finder(model, levels, config, arrays, indices)
    result["history_status"] = "complete"
    return result


def investigate_finder(model, levels, config, arrays, indices):
    """Replay just the conflicting finder cohort, stopping at finder capture."""
    settings = config["audit"]
    dataset = group(arrays, "dataset")

    import jax
    from geo_experiments.collection import make_collector
    from geo_experiments.target_collection import episode_keys
    # Replay only this diagnostic subset, never another probe-training collection.
    # Recompute groups from replay features and disclose exact reproduction rates.
    records = {"finder_gru": defaultdict(list)}
    fingerprints, reproduced = [], {"finder_gru": 0}
    skipped = 0
    for m, level in enumerate(levels):
        map_ids = dataset.get("map_id", np.zeros(len(dataset["finder_gru"]), dtype=np.int32))
        rows = indices[map_ids[indices] == m]
        if not len(rows):
            continue
        collect = make_collector(level, history_stride=settings["every"], stop_after_finder=True)
        for start in range(0, len(rows), settings["batch_size"]):
            selected = rows[start:start + settings["batch_size"]]
            ids = dataset["episode_id"][selected]
            if config.get("protocol", {}).get("decoder", "legacy") == "legacy":
                import jax.numpy as jnp
                keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(config["dataset"]["seed"]), i))(jnp.asarray(ids))
            else:
                keys = episode_keys(config["dataset"]["seed"], m, ids)
            batch = jax.device_get(collect(model, keys))
            for local, original in enumerate(selected):
                if int(batch["finder_step"][local]) < 0:
                    skipped += 1
                    continue
                valid = batch["history_steps"][local] >= 0
                steps, flags = batch["history_steps"][local, valid], batch["history_flags"][local, valid]
                raw, encoded, context = (batch[k][local, valid] for k in ("history_obs", "history_encoded", "history_context"))
                hashes = [digest(steps, flags, values) for values in (raw, encoded, np.concatenate((encoded, context), axis=-1))]
                cell = (m, *np.floor(batch["target_position"][local] / level.building.cell_size_m).astype(int).tolist())
                fingerprints.append((int(original), hashes))
                reproduced["finder_gru"] += int(np.array_equal(batch["finder_gru"][local], dataset["finder_gru"][original]))
                records["finder_gru"][digest(batch["finder_gru"][local])].append((cell, hashes))
            del batch
            print(f"[audit] finder input histories: map {m + 1}/{len(levels)}, "
                  f"{min(start + settings['batch_size'], len(rows))}/{len(rows)} conflicting finder samples")
    by_decoder = {decoder: conflict_summary(records, decoder) for decoder in ("map_specific", "map_cell", "shared_cell")}
    result = dict(status="full", reason="finder_conflicts", samples=len(fingerprints), attempted_samples=len(indices), skipped_missing_snapshots=skipped,
                  inputs=by_decoder["map_cell"], by_decoder=by_decoder, exact_replayed_snapshots=reproduced,
                  every=settings["every"], scope="Conflicting finder-GRU groups only; replay stops at finder capture.")
    add_group(arrays, "audit", {"sample_indices": np.asarray([r[0] for r in fingerprints]),
              "history_sha256": np.asarray([[list(h) for h in row[1]] for row in fingerprints], dtype=np.uint8)})
    return result


def footer_note(summary):
    if summary and summary.get("format") == FORMAT:
        from geo_experiments.common import SOURCE_LABELS
        screen = summary["pretest"]
        rates = "; ".join(f"{SOURCE_LABELS[s]} {100 * v['same_latent_fraction']:.3f}% same latents" for s, v in screen["inputs"].items())
        histories = ("Finder observation histories were unnecessary." if summary["history_status"] == "not_required"
                     else f"Finder history analysis: {summary['history_status']} ({summary['history_requested_samples']:,} selected episodes).")
        return f"Latent pretest: {rates} on {screen['samples']:,} samples; threshold {100 * screen['threshold']:.3f}%. Triggered packet analyses use all retained snapshots. {histories} Exact equality only; low duplication does not rule out approximate ambiguity."
    if not summary or summary.get("format") != "swarmecho-conditional-audit/v1":
        return ""
    from geo_experiments.common import SOURCE_LABELS
    screen = summary["pretest"]
    rates = "; ".join(f"{SOURCE_LABELS[s]} {100 * v['same_latent_fraction']:.3f}% same latents" for s, v in screen["inputs"].items())
    suffix = ("Detailed analysis skipped (all inputs below the threshold)." if summary["status"] == "screen_only"
              else "Detailed analysis forced by configuration." if summary.get("reason") == "forced" else "Detailed analysis triggered.")
    return f"Latent pretest: {rates} on {screen['samples']:,} samples; threshold {100 * screen['threshold']:.3f}%. {suffix} Exact equality only; low duplication does not rule out approximate ambiguity."


def main():
    """Refresh a stopped suite collection's audit without loading JAX or a policy."""
    import json
    import html
    from geo_experiments.artifacts import archive, save_data, update_run
    from geo_experiments.common import ROOT, resolve_path
    from geo_experiments.collision_report import snapshot_sections
    from geo_experiments.report_layout import document

    if len(sys.argv) != 2 or not sys.argv[1].startswith("run="):
        raise ValueError("Use run=<suite collection folder> to analyse saved snapshots without policy replay.")
    run_dir = resolve_path(sys.argv[1].split("=", 1)[1])
    if not run_dir.is_relative_to((ROOT / "geo_experiments").resolve()):
        raise ValueError("Audit output must remain inside geo_experiments/.")
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != "target_collection" or manifest.get("progress", {}).get("stage") not in ("collected", "complete"):
        raise ValueError("A completed primary target collection is required; incomplete quotas cannot be audited as a complete dataset.")
    arrays = archive(run_dir / "data.npz")
    suite_file = run_dir.parent.parent / "run.json"
    plan = json.loads(suite_file.read_text(encoding="utf-8")).get("execution_plan", []) if suite_file.is_file() else []
    decoders = sorted({j["config"]["protocol"]["decoder"] for j in plan if j["collection_key"] == run_dir.name})
    summary, indices = plan_audit(manifest["config"], arrays, decoders=decoders or None)
    save_data(run_dir, arrays)
    complete = not len(indices)
    update_run(run_dir, ambiguity_audit=summary, progress={"stage": "complete" if complete else "collected"})
    views = decoders or [manifest["config"]["protocol"]["decoder"]]
    note = footer_note(summary)
    body = ('<header><div class="eyebrow">Geo experiments · saved collection diagnostics</div>'
            f'<h1>{html.escape(manifest["config"]["run"]["name"])}</h1>'
            '<p class="note">Actual collected snapshots; no policy replay or MLP fitting is performed by this command.</p></header>')
    for decoder in views:
        body += '<h2 style="margin-top:34px">' + html.escape(decoder) + '</h2>' + snapshot_sections(summary, decoder, summary.get("map_names", []) if summary else [])
    body += f'<footer><p class="note">{html.escape(note)}</p></footer>'
    temporary = run_dir / ".audit.tmp.html"
    temporary.write_text(document("Saved collection audit", body), encoding="utf-8")
    temporary.replace(run_dir / "audit.html")
    print(f"[audit] saved {run_dir / 'audit.html'}")
    if len(indices):
        print(f"[audit] {len(indices):,} ambiguous finder samples need policy replay; resume the suite in WSL.")


if __name__ == "__main__":
    main()
