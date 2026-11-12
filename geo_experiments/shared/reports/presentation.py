"""Location-task labels, source colors and categorical result tables."""
import html
import math
import numpy as np
from geo_experiments.shared.axes import SOURCES, SOURCE_LABELS, TASKS, source_label
from geo_experiments.shared.dashboard import numeric_cell
from geo_experiments.shared.report_components import _histogram

COLORS = {"finder_gru": "#68b9f2", "finder_message": "#f3bb65", "base_message": "#6ed1b5"}
COLORS.update(agent_gru=COLORS["finder_gru"], agent_message=COLORS["finder_message"])
TASK_LABELS = {"target_cell": "Target cell", "target_storey": "Target storey", "own_cell": "Own cell"}


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
            color = COLORS.get(source, "#68b9f2")

            def cells(row, scope):
                tooltip = f'Initialization standard deviation: {100 * row["std"]:.2f} percentage points'
                key = f"{TASK_LABELS[task]} / {source_label(source)} / {scope}"
                return (numeric_cell(key, "Accuracy", 100 * row["accuracy"] if row["n"] else None,
                                     unit="percent", decimals=1, suffix="%", css="accuracy", title=tooltip,
                                     version=metric_version) + '<td>'
                        + "".join(f'<div class="error-histogram-view" data-measure="{measure}"'
                                  + (" hidden" if measure == "share" else "") + ">"
                                  + _histogram(row["errors"] if row["n"] else None, bins, color, unit,
                                               measure, count_limit) + "</div>"
                                  for measure in ("count", "share")) + "</td>")

            parts.append(f'<tr class="overall"><td><strong>{TASK_LABELS[task]}</strong>'
                         f'<span class="source"><i style="background:{color}"></i>{source_label(source)}</span></td>'
                         + cells(record, "Overall") + "</tr>")
            parts.append('<tr class="detail-row"><td colspan="3"><details><summary>'
                         + (('Target-required agents, storeys and maps' if any(k.startswith('map-') for k, _ in detail_groups)
                             else 'Target-required agents and storeys') if detail_groups else 'Per-storey details') + '</summary>'
                         '<div class="secondary"><table><tbody>')
            for key, title in (detail_groups or [(f"storey-{floor}", f"Storey {floor}") for floor in floors]):
                parts.append(f'<tr><td>{html.escape(title)}</td>' + cells(records[task, source, key], title) + "</tr>")
            parts.append("</tbody></table></div></details></td></tr>")
    return "".join(parts) + "</tbody></table></div>"


