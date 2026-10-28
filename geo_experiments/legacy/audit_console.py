"""Audit tables and import of the previously supplied console-only audit."""
from __future__ import annotations

import html
import re
from pathlib import Path

from geo_experiments.audit_common import STAGES
from geo_experiments.artifacts import update_run
from geo_experiments.common import fingerprint
from geo_experiments.dashboard import numeric_cell


def parse_console(text):
    # Accept both historical bare output and the training-style clock prefixes.
    text = re.sub(r"(?m)^\[(?:\d{4}-\d{2}-\d{2} )?\d{2}:\d{2}:\d{2}\]\s?", "", text)
    def required(pattern):
        match = re.search(pattern, text, re.MULTILINE)
        if not match:
            raise ValueError(f"Audit console is missing a required result: {pattern}")
        return match

    sample = required(r"^Samples: (\d+); exact distinct finder GRUs: (\d+)")
    mixed = required(r"^Mixed-target-cell GRU groups: (\d+); episodes in them: (\d+)")
    pairs = required(r"^Different-cell episode pairs within identical GRUs: (\d+)")
    settings = required(r"^\[audit\] samples=(\d+); every=(\d+); batch=(\d+); seed=(\d+)")
    identity = dict(re.findall(r"^\[identity\] (\w+)=([0-9a-f]{64})$", text, re.MULTILINE))
    result = {"format": "swarmecho-ambiguity-audit/v1", "status": "complete",
              "samples": int(sample[1]), "distinct_grus": int(sample[2]),
              "mixed_cell_groups": int(mixed[1]), "mixed_cell_samples": int(mixed[2]),
              "cross_cell_pairs": int(pairs[1]), "exact": {}, "references": {},
              "settings": {"samples": int(settings[1]), "every": int(settings[2]), "batch_size": int(settings[3])},
              "source": {"kind": "imported_console", "identity": identity, "seed": int(settings[4]),
                         "scope": "separate console audit, not the complete probe dataset"}}
    timestamp = re.search(r"^\[audit\] (\d{4}-\d{2}-\d{2}T\S+)$", text, re.MULTILINE)
    if timestamp:
        result["source"]["recorded_at"] = timestamp[1]
    for title in STAGES:
        match = required(r"^\s+" + re.escape(title) + r": (\d+)/(\d+) .*EXACT")
        count, total = int(match[1]), int(match[2])
        if total != result["cross_cell_pairs"] or not 0 <= count <= total:
            raise ValueError("Inconsistent cross-cell pair counts in audit console.")
        result["exact"][title] = {"matched_pairs": count, "total_pairs": total,
                                  "rate": count / total if total else None}
    reference = stage = None
    number = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    for line in text.splitlines():
        head = re.match(r"\s*(All references|Different-cell references): (\d+) time-aligned comparisons; (\d+) excluded", line)
        newer = re.match(r"\s*(all|cross_cell): (\d+) comparisons; excluded times=(\d+), unavailable reference=(\d+)", line)
        if head or newer:
            match = head or newer
            kind = "all" if match[1] in ("All references", "all") else "cross_cell"
            reference = {"count": int(match[2]), "excluded_times": int(match[3]),
                         "excluded_memory": int(newer[4]) if newer else 0, "stages": {}}
            result["references"][kind] = reference
            stage = None
            continue
        if reference is None:
            continue
        flag = re.search(r"Identical active/reset flags: " + number + "%", line)
        if flag:
            reference["flags_equal_rate"] = float(flag[1]) / 100
        for title in STAGES:
            match = re.match(r"\s*" + re.escape(title) + r": equal components " + number
                             + r"%; fully equal steps " + number + "%", line)
            if match:
                stage = {"equal_components_rate": float(match[1]) / 100,
                         "equal_steps_rate": float(match[2]) / 100, "thresholds": []}
                reference["stages"][title] = stage
                break
        if stage is None:
            continue
        values = re.search(r"max-abs(?: difference)? median/p90/max:?\s*" + number + r"\s*/\s*"
                           + number + r"\s*/\s*" + number + r"; median RMS " + number, line)
        if values:
            stage.update(zip(("max_abs_median", "max_abs_p90", "max_abs_max", "rms_median"),
                             (float(v) for v in values.groups())))
        threshold = re.search(r"Entire history(?: max-abs)? <= " + number + r": (\d+)(?:/(\d+))?", line)
        if threshold:
            count = int(threshold[2])
            total = reference["count"]
            if not 0 <= count <= total or threshold[3] and int(threshold[3]) != total:
                raise ValueError("Inconsistent audit reference counts.")
            separation = re.search(r"target separation median/max(?: metres:)?\s*" + number + r"/" + number, line)
            stage["thresholds"].append({"atol": float(threshold[1]), "count": count,
                "rate": count / total if total else None,
                "target_separation_median_m": float(separation[1]) if separation else None,
                "target_separation_max_m": float(separation[2]) if separation else None})
    all_max = re.search(r"Maximum target separation \(including same-cell pairs\): " + number, text)
    cross = re.search(r"Mixed-cell identical-input-history clusters: (\d+); maximum cross-cell target separation: " + number, text)
    newer = re.search(r"Identical GRU-input histories: maximum target separation (\S+) m; cross-cell (\S+) m", text)
    result["target_separation"] = {
        "max_any_m": float(all_max[1]) if all_max else float(newer[1]) if newer and newer[1] != "None" else None,
        "max_cross_cell_m": float(cross[2]) if cross else float(newer[2]) if newer and newer[2] != "None" else None,
        "mixed_cell_clusters": int(cross[1]) if cross else None, "examples": []}
    collection = re.search(r"^\[collection\] successful=(\d+)/(\d+) .*successful missing snapshot=(\d+)", text, re.MULTILINE)
    if collection:
        result["source"]["collection"] = {"successful_episodes": int(collection[1]),
            "attempted_episodes": int(collection[2]), "successful_missing_snapshot": int(collection[3])}
    if result["samples"] != result["settings"]["samples"]:
        raise ValueError("Console audit did not finish its requested sample count.")
    return result


def attach_console_audit(run_dir, path, manifest):
    path = Path(path).resolve()
    text = path.read_text(encoding="utf-8")
    result = parse_console(text)
    expected = manifest["provenance"]["identity"]
    identity = result["source"]["identity"]
    for key in ("checkpoint_sha256", "map_sha256"):
        if identity.get(key) != expected.get(key):
            raise ValueError(f"Cannot attach audit from a different checkpoint/map: {key}")
    result["source"].update(console_sha256=fingerprint(path), console=text,
                            differing_identity_fields=[key for key in ("level_sha256", "runtime_sha256", "collector_sha256")
                                                       if identity.get(key) != expected.get(key)])
    update_run(run_dir, ambiguity_audit=result)
    manifest["ambiguity_audit"] = result
    return result


def audit_section(summary):
    if summary and summary.get("format") in ("swarmecho-conditional-audit/v1", "swarmecho-target-audit/v2"):
        from geo_experiments.target_report import detailed_audit
        return detailed_audit(summary)
    if not summary:
        return '<section><h2>Ambiguity audit</h2><p class="note">No audit was collected for this run.</p></section>'
    if summary.get("format") == "swarmecho-own-cell-audit/v1":
        return own_audit_section(summary)
    parts = ['<section><h2>Ambiguity audit</h2>']
    imported = summary.get("source", {}).get("kind") == "imported_console"
    every = summary.get("settings", {}).get("every")
    scope = ("Imported from the separate console run: same checkpoint/map, its own episode collection. "
             "Level/runtime versions may differ; this is a reported summary, not recovered histories."
             if imported else "First matched episodes from this run's collection; no separate evaluation rollout.")
    parts.append(f'<p class="note">{scope} {summary["samples"]:,} samples; every {every} step(s), through finder capture. '
                 'Exact counts compare all different-cell pairs within identical GRUs. Approximate similarity compares each episode with its group’s first episode.</p>')
    version = f"prefix-history/v1/every={every}"

    def table(name, headers, rows):
        parts.append(f'<h3>{html.escape(name)}</h3><div class="table-scroll"><table class="audit-table" data-compare-table="{html.escape(name, quote=True)}"><thead><tr>')
        parts.extend(f'<th>{html.escape(h)}</th>' for h in headers)
        parts.append('</tr></thead><tbody>')
        for label, cells in rows:
            parts.append(f'<tr><td>{html.escape(label)}</td>' + ''.join(cells) + '</tr>')
        parts.append('</tbody></table></div>')

    rows = []
    for label, value, unit, decimals, suffix in (
        ("Audited episodes", summary.get("samples"), "episodes", 0, ""),
        ("Exact distinct finder GRUs", summary.get("distinct_grus"), "states", 0, ""),
        ("Groups with conflicting target cells", summary.get("mixed_cell_groups"), "groups", 0, ""),
        ("Episodes in conflicting-cell groups", summary.get("mixed_cell_samples"), "episodes", 0, ""),
        ("Share in conflicting-cell groups", 100 * summary["mixed_cell_samples"] / summary["samples"], "percent", 1, "%"),
        ("Different-cell pairs within identical GRUs", summary.get("cross_cell_pairs"), "pairs", 0, ""),
        ("Maximum target separation: identical GRU-input histories", summary.get("target_separation", {}).get("max_any_m"), "metres", 3, " m"),
        ("Maximum cross-cell target separation", summary.get("target_separation", {}).get("max_cross_cell_m"), "metres", 3, " m")):
        rows.append((label, [numeric_cell(label, "Value", value, unit=unit, decimals=decimals, suffix=suffix)]))
    table("Snapshot ambiguity", ("Metric", "Value"), rows)
    rows = []
    for title in STAGES:
        row = summary.get("exact", {}).get(title, {})
        rate = row.get("rate")
        rows.append((title, [numeric_cell(title, "Matching different-cell pairs", row.get("matched_pairs"), unit="pairs", version=version),
                            numeric_cell(title, "Exact history match", 100 * rate if rate is not None else None,
                                         unit="percent", decimals=1, suffix="%", version=version)]))
    table("Exact history matches", ("Input history", "Matching different-cell pairs", "Exact match"), rows)
    for kind, name in (("cross_cell", "Reference similarity — different target cells"),
                       ("all", "Reference similarity — all episodes")):
        reference = summary.get("references", {}).get(kind)
        if reference is None:
            continue
        if kind == "all":
            parts.append('<details class="audit-detail"><summary>All-episode reference details</summary>')
        rows = []
        for title in STAGES:
            row = reference.get("stages", {}).get(title, {})
            exact = next((t for t in row.get("thresholds", []) if t.get("atol") == 0), {})
            near = next((t for t in row.get("thresholds", []) if t.get("atol") == .001), {})
            values = []
            for column, value, unit, decimals, suffix in (
                ("Exact history", exact.get("rate"), "percent", 1, "%"),
                ("Within 0.001", near.get("rate"), "percent", 1, "%"),
                ("Maximum difference", row.get("max_abs_max"), "component", 6, ""),
                ("Median RMS", row.get("rms_median"), "component", 6, ""),
                ("Median target separation: exact histories", exact.get("target_separation_median_m"), "metres", 3, " m")):
                values.append(numeric_cell(title, column, 100 * value if unit == "percent" and value is not None else value,
                                          unit=unit, decimals=decimals, suffix=suffix, version=version))
            rows.append((title, values))
        table(name, ("Input history", "Exact history", "Within 0.001", "Max difference", "Median RMS", "Median separation"), rows)
        parts.append(f'<p class="note">{reference.get("count", 0):,} reference comparisons; '
                     f'{reference.get("excluded_times", 0):,} excluded for unequal sampled times; '
                     f'{reference.get("excluded_memory", 0):,} nonidentical reference histories unavailable under the memory limit. '
                     'Absolute tolerances; maximum difference spans every sampled component. Metres are target separation, not route error.</p>')
        if kind == "all":
            parts.append('</details>')
    parts.append('<p class="note">No conflicting pairs gives N/A match rates. More unique vectors alone do not establish better target decoding. '
                 'With sampling intervals greater than one, matching recorded steps does not establish equality at skipped steps.</p></section>')
    return ''.join(parts)


def own_audit_section(summary):
    from geo_experiments.common import SOURCE_LABELS

    parts = ['<section><h2>Ambiguity audit</h2>',
             f'<p class="note">{summary["samples"]:,} retained snapshots selected with a fixed seed across all splits. '
             'All active actor inputs were fingerprinted every step through each snapshot, with lifetime resets. '
             'Same-cell repetition is redundancy; different cells within an identical GRU or message are ambiguity.</p>',
             '<div class="table-scroll"><table class="audit-table" data-compare-table="Own-cell snapshot ambiguity">'
             '<thead><tr><th>Input</th><th>Distinct states</th><th>Conflicting groups</th><th>Snapshots in those groups</th>'
             '<th>Different-cell pairs</th></tr></thead><tbody>']
    for source, row in summary["inputs"].items():
        label = SOURCE_LABELS[source]
        parts.append(f'<tr><td>{html.escape(label)}</td>')
        for key, column, unit in (("distinct_states", "Distinct states", "states"),
                                  ("mixed_cell_groups", "Conflicting groups", "groups"),
                                  ("mixed_cell_samples", "Snapshots in conflicting groups", "snapshots"),
                                  ("cross_cell_pairs", "Different-cell pairs", "pairs")):
            parts.append(numeric_cell(label, column, row[key], unit=unit, version="own-cell-audit/v1"))
        parts.append('</tr>')
    parts.append('</tbody></table></div><h3>Exact history matches within different-cell pairs</h3>'
                 '<div class="table-scroll"><table class="audit-table" data-compare-table="Own-cell exact history matches">'
                 '<thead><tr><th>Input / history</th><th>Matching pairs</th><th>Share of conflicting pairs</th></tr></thead><tbody>')
    for source, row in summary["inputs"].items():
        for stage, count in row["history_matches"].items():
            label = f"{SOURCE_LABELS[source]} / {stage}"
            parts.append(f'<tr><td>{html.escape(label)}</td>'
                         + numeric_cell(label, "Matching pairs", count, unit="pairs", version="own-cell-history/v1")
                         + numeric_cell(label, "Share of conflicting pairs", 100 * count / row["cross_cell_pairs"]
                                        if row["cross_cell_pairs"] else None, unit="percent", decimals=1,
                                        suffix="%", version="own-cell-history/v1") + '</tr>')
    parts.append('</tbody></table></div><p class="note">Matching combined-input histories indicate observational aliasing '
                 'for those conflicting pairs; different histories with the same state indicate representation collisions. '
                 'No conflicting pairs gives N/A shares. Exact SHA-256 fingerprints treat signed zero equally; '
                 'this audit does not test approximate similarity. Full floating histories are discarded after filtering.</p></section>')
    return ''.join(parts)
