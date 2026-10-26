"""Task-aware collision summaries and expandable episode-property profiles."""
from __future__ import annotations

import html

from geo_experiments.common import SOURCE_LABELS
from geo_experiments.dashboard import numeric_cell


def _distribution(rows, name, map_names):
    parts = []
    for row in rows:
        value = row["value"]
        if name == "Map":
            value = map_names[int(value)] if int(value) < len(map_names) else f"Map {int(value) + 1}"
        elif name == "Map + target cell":
            m, *xyz = value
            map_name = map_names[int(m)] if int(m) < len(map_names) else f"Map {int(m) + 1}"
            value = f"{map_name}: {tuple(xyz)}"
        elif isinstance(value, list):
            value = str(tuple(value))
        elif isinstance(value, bool):
            value = "Yes" if value else "No"
        elif name == "Target-required agents" and value == 0:
            value = "Not estimated"
        elif name.startswith("Active agents") and value < 0:
            value = "Not recorded"
        parts.append(f"{html.escape(str(value))}: {row['samples']:,}")
    return '; '.join(parts)


def snapshot_sections(summary, decoder, map_names=()):
    if not summary or summary.get("format") != "swarmecho-target-audit/v2":
        return ""
    decoder = "map_specific" if decoder == "legacy" else decoder
    scope = {"map_specific": "Map identity selects the expert; label conflicts are counted within each map.",
             "map_cell": "A conflicting label means a different map or target XYZ cell.",
             "shared_cell": "A conflicting label means different target XYZ coordinates; map identity alone is not a conflict."}[decoder]
    parts = []
    for source, analysis in summary["snapshot_analyses"].items():
        title = {"base_message": "Base-message collision analysis", "finder_message": "Finder-message collision analysis",
                 "finder_gru": "Finder-GRU collision analysis"}[source]
        task = analysis["by_decoder"][decoder]
        n = analysis["samples"]
        parts.append(f'<section><h2>{title}</h2><p class="note">'
            f'Exact equality of the complete saved {html.escape(SOURCE_LABELS[source].lower())} vector; '
            f'all {n:,} retained episodes across training, validation and evaluation are analysed. '
            'These are saved-data diagnostics, not held-out MLP scores. No episode replay is needed for this section. '
            f'{scope}</p>')
        parts.append(f'<p><strong>{analysis["repeated_samples"]:,} / {n:,} samples '
            f'({100 * analysis["repeated_fraction"]:.3f}%)</strong> belong to '
            f'{analysis["repeated_groups"]:,} repeated-vector groups. '
            f'<strong>{task["conflicting_samples"]:,} samples</strong> belong to '
            f'{task["conflicting_groups"]:,} groups with conflicting target labels.</p>'
            '<p class="note">An affected sample is not necessarily an incorrect prediction. Repeated vectors with one target label are redundancy; '
            'different labels for the same readout input establish ambiguity for that label. Different vectors do not guarantee their target information differs.</p>')
        parts.append('<div class="table-scroll"><table class="audit-table" data-compare-table="Collision scope"><thead><tr><th>Metric</th><th>Count</th><th>Share of all samples</th></tr></thead><tbody>')
        for label, value, unit, share in (
            ("Retained episodes", n, "episodes", 100.0),
            ("Distinct vectors", analysis["distinct_vectors"], "states", None),
            ("Repeated-vector groups", analysis["repeated_groups"], "groups", None),
            ("Samples in repeated-vector groups", analysis["repeated_samples"], "episodes", 100 * analysis["repeated_fraction"]),
            ("Groups with conflicting target labels", task["conflicting_groups"], "groups", None),
            ("Samples in conflicting-label groups", task["conflicting_samples"], "episodes", 100 * task["conflicting_samples"] / n if n else None)):
            version = f"snapshot-collisions/{source}/{decoder}/v1"
            parts.append(f'<tr><td>{label}</td>' + numeric_cell(label, "Count", value, unit=unit, version=version)
                         + numeric_cell(label, "Share of all samples", share, unit="percent", decimals=3, suffix="%", version=version) + '</tr>')
        parts.append('</tbody></table></div><h3>What stays the same inside repeated-vector groups?</h3>'
            '<div class="table-scroll"><table class="audit-table" data-compare-table="Episode similarities"><thead><tr>'
            '<th>Episode property</th><th>Homogeneous groups / known groups</th><th>Matching pairs within groups</th><th>Full-dataset pair baseline</th></tr></thead><tbody>')
        for name, row in analysis["similarities"].items():
            known = row["known_groups"]
            version = f"snapshot-similarities/{source}/v1"
            parts.append(f'<tr><td>{html.escape(name)}</td><td>{row["same_groups"]:,} / {known:,}</td>'
                + numeric_cell(name, "Matching pairs within groups", 100 * row["pair_agreement"] if row["pair_agreement"] is not None else None,
                               unit="percent", decimals=1, suffix="%", version=version)
                + numeric_cell(name, "Full-dataset pair baseline", 100 * row["baseline_pair_agreement"] if row["baseline_pair_agreement"] is not None else None,
                               unit="percent", decimals=1, suffix="%", version=version) + '</tr>')
        for name in analysis["missing_properties"]:
            parts.append(f'<tr><td>{html.escape(name)}</td><td colspan="3" class="muted">Not recorded in this collection</td></tr>')
        parts.append('</tbody></table></div><p class="note">Homogeneous means every episode in that group agrees. Pair agreement weights larger groups by their number of episode pairs. '
            'The baseline draws pairs from the full retained collection, so map/population mix can explain part of a difference. '
            'Unknown required-agent counts and missing capture-time populations are excluded from agreement calculations. Agent indices identify slots, not the number of active agents.</p>')
        if source != "finder_gru":
            parts.append('<p class="note">Repeated groups with every finder GRU distinct: '
                f'{analysis["upstream"]["finder_gru"]:,}/{analysis["repeated_groups"]:,}; '
                f'with every finder message distinct: {analysis["upstream"]["finder_message"]:,}/{analysis["repeated_groups"]:,}. '
                'This describes snapshot differences; it does not locate a causal loss along a relay path.</p>')
        parts.append('<details><summary>Inspect individual repeated-vector groups</summary>')
        for item in analysis["groups"]:
            parts.append(f'<details style="margin:14px 0"><summary>Group {item["id"]} · {item["samples"]:,} samples</summary>'
                         '<div class="table-scroll"><table class="audit-table"><tbody>')
            for name, values in item["properties"].items():
                parts.append(f'<tr><td>{html.escape(name)}</td><td>{_distribution(values, name, map_names)}</td></tr>')
            for key, count in item["upstream_distinct"].items():
                parts.append(f'<tr><td>Distinct {SOURCE_LABELS[key]}</td><td>{count:,} / {item["samples"]:,}</td></tr>')
            for name, values in item.get("timing", {}).items():
                parts.append(f'<tr><td>{html.escape(name)}</td><td>{values["min"]:g}–{values["max"]:g}; mean {values["mean"]:.1f}</td></tr>')
            position = item["target_position"]
            parts.append('<tr><td>Target position extent (absolute XYZ metres)</td><td>'
                f'{html.escape(str(position["min_xyz_m"]))} to {html.escape(str(position["max_xyz_m"]))}</td></tr>'
                f'<tr><td>RMS target spread around group centroid</td><td>{position["rms_spread_m"]:.3f} m</td></tr>'
                '</tbody></table></div></details>')
        parts.append('</details></section>')
    return ''.join(parts)
