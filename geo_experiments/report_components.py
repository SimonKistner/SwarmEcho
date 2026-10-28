"""Shared report palette, legends, error histograms and result tables."""
from __future__ import annotations
import html
import math
import numpy as np
from geo_experiments.common import SOURCES, SOURCE_LABELS, TASKS
from geo_experiments.dashboard import numeric_cell

# One palette throughout tables, learning curves, comparisons and matrices.
BACKGROUND, PANEL, TEXT, MUTED, GRID = "#0b1220", "#111c2e", "#e4edf7", "#9aacc3", "#2b3b53"
COLORS = {"finder_gru": "#68b9f2", "finder_message": "#f3bb65", "base_message": "#6ed1b5"}
COLORS.update(agent_gru=COLORS["finder_gru"], agent_message=COLORS["finder_message"])
TASK_LABELS = {"target_cell": "Target cell", "target_storey": "Target storey", "own_cell": "Own cell"}


def _legend(items):
    return '<div class="legend">' + "".join(
        f'<span><i style="background:{color}"></i>{html.escape(label)}</span>' for label, color in items
    ) + "</div>"


def _histogram(errors, bins, color, unit, measure="count", count_limit=1):
    if errors is None:
        return '<span class="muted">No evaluation samples</span>'
    if not len(errors):
        return '<span class="perfect">All predictions correct</span>'
    width, left, top, bottom = 430, 48, 12, 92
    limit = 100 if measure == "share" else count_limit
    step = (width - left - 10) / len(bins)
    bars = []
    for i, value in enumerate(bins):
        count = int((errors == value).sum())
        percentage = 100 * count / len(errors)
        height = (bottom - top) * (percentage if measure == "share" else count) / limit
        x = left + i * step
        label = "Wrong map" if value == -1 else str(int(value)) if np.isfinite(value) else "No route"
        bars.append(f'<g><title>{label}{"" if value == -1 else " " + unit + " off"}: {count:,} wrong predictions '
                    f'({percentage:.1f}% of errors, pooled across seeds)</title>'
                    f'<rect x="{x + step * .15:.2f}" y="{bottom - height:.2f}" '
                    f'width="{step * .7:.2f}" height="{height:.2f}" rx="2" fill="{color}"/>'
                    f'<text x="{x + step / 2:.2f}" y="106" text-anchor="middle">{label}</text></g>')
    lines = ""
    for fraction in (0, .5, 1):
        y = bottom - fraction * (bottom - top)
        tick = f"{fraction * limit:.0f}%" if measure == "share" else f"{fraction * limit:,.0f}"
        lines += (f'<line x1="{left}" x2="{width - 10}" y1="{y}" y2="{y}" stroke="{GRID}"/>'
                  f'<text x="{left - 5}" y="{y + 3}" text-anchor="end">{tick}</text>')
    axis_label = "% of incorrect predictions" if measure == "share" else "Incorrect prediction count"
    return (f'<svg class="histogram" viewBox="0 0 {width} 127" role="img" '
            f'aria-label="Distribution of {unit} off among incorrect predictions">'
            f'<g fill="{MUTED}" font-family="system-ui,sans-serif" font-size="10">'
            + lines + "".join(bars) + f'<text x="{(left + width) / 2}" y="122" text-anchor="middle">'
            f'{unit} off · {axis_label}</text></g></svg>')


def _table(records, floors, tasks=TASKS, sources=SOURCES, *, detail_groups=None, metric_version="heldout-accuracy/v1"):
    parts = ['<div class="table-scroll"><table class="results"><thead><tr>'
             '<th>Probe / input</th><th>Accuracy</th><th>Incorrect predictions</th></tr></thead><tbody>']
    for task in tasks:
        all_errors = np.concatenate([records[task, source, "overall"]["errors"] for source in sources])
        finite = all_errors[np.isfinite(all_errors) & (all_errors > 0)]
        maximum = (int(floors.max() - floors.min()) if task == "target_storey"
                   else int(finite.max()) if len(finite) else 1)
        bins = list(range(1, max(maximum, 1) + 1))
        if (all_errors == -1).any():
            bins.insert(0, -1)
        if np.isinf(all_errors).any():
            bins.append(np.inf)
        unit = "storeys" if task == "target_storey" else "cells"
        peak = max((int((records[task, source, "overall"]["errors"] == value).sum())
                    for source in sources for value in bins), default=0)
        magnitude = 10 ** math.floor(math.log10(max(peak, 1)))
        count_limit = max(2, math.ceil(peak / (2 * magnitude)) * 2 * magnitude)
        for source in sources:
            record = records[task, source, "overall"]
            color = COLORS[source]

            def cells(row, scope):
                tooltip = f'Initialization standard deviation: {100 * row["std"]:.2f} percentage points'
                key = f"{TASK_LABELS[task]} / {SOURCE_LABELS[source]} / {scope}"
                return (numeric_cell(key, "Accuracy", 100 * row["accuracy"] if row["n"] else None,
                                     unit="percent", decimals=1, suffix="%", css="accuracy", title=tooltip,
                                     version=metric_version) + '<td>'
                        + "".join(f'<div class="error-histogram-view" data-measure="{measure}"'
                                  + (" hidden" if measure == "share" else "") + ">"
                                  + _histogram(row["errors"] if row["n"] else None, bins, color, unit,
                                               measure, count_limit) + "</div>"
                                  for measure in ("count", "share")) + "</td>")

            parts.append(f'<tr class="overall"><td><strong>{TASK_LABELS[task]}</strong>'
                         f'<span class="source"><i style="background:{color}"></i>{SOURCE_LABELS[source]}</span></td>'
                         + cells(record, "Overall") + "</tr>")
            parts.append('<tr class="detail-row"><td colspan="3"><details><summary>'
                         + (('Target-required agents, storeys and maps' if any(k.startswith('map-') for k, _ in detail_groups)
                             else 'Target-required agents and storeys') if detail_groups else 'Per-storey details') + '</summary>'
                         '<div class="secondary"><table><tbody>')
            for key, title in (detail_groups or [(f"storey-{floor}", f"Storey {floor}") for floor in floors]):
                parts.append(f'<tr><td>{html.escape(title)}</td>' + cells(records[task, source, key], title) + "</tr>")
            parts.append("</tbody></table></div></details></td></tr>")
    return "".join(parts) + "</tbody></table></div>"


