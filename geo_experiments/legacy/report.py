"""Self-contained dark report from saved probe data; no policy runtime required."""
from __future__ import annotations

import base64
import html
import io
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from geo_experiments.artifacts import group, prune_legacy, save_data, update_run
from geo_experiments.common import SOURCES, SOURCE_LABELS, TASKS, fingerprint, resolve_path, experiment_axes
from geo_experiments.geometry import prediction_errors
from geo_experiments.dashboard import numeric_cell, report_controls, decorate_report, build_dashboard
from geo_experiments.legacy.audit_console import audit_section
from geo_experiments.console import terminal_print as print

from geo_experiments.report_components import (BACKGROUND, PANEL, TEXT, MUTED, GRID, COLORS, TASK_LABELS, _legend, _histogram, _table)


def _results(arrays, config):
    tasks, sources = experiment_axes(config)
    graph, labels, splits = (group(arrays, key) for key in ("geometry", "labels", "splits"))
    test = splits["test"]
    floors = np.unique(graph["cells"][:, 2] + 1)
    masks = {"overall": np.ones(len(test), dtype=bool)}
    required = arrays.get("dataset/target_required_agents")
    if required is not None and "own_cell" not in labels:
        masks.update({f"required-{int(k)}": required[test] == k for k in np.unique(required)})
    masks.update({f"storey-{floor}": labels["actual_storey"][test] == floor for floor in floors})
    records = {}
    for task in tasks:
        actual = labels[task][test]
        for source in sources:
            predicted = np.stack([arrays[f"probes/{task}/{source}/seed_{seed}/predicted"]
                                  for seed in config["probe"]["seeds"]])
            if predicted.shape != (len(config["probe"]["seeds"]), len(actual)):
                raise ValueError(f"Prediction rows do not match held-out split: {task}/{source}")
            errors = np.stack([prediction_errors(task, p, test, labels, graph) for p in predicted])
            correct = predicted == actual[None, :]
            for name, mask in masks.items():
                accuracy = correct[:, mask].mean(axis=1) if mask.any() else np.full(len(predicted), np.nan)
                wrong = errors[:, mask][~correct[:, mask]]
                finite = wrong[np.isfinite(wrong)]
                if np.any((finite <= 0) | (finite != np.floor(finite))):
                    raise ValueError("Incorrect predictions must have positive, discrete error.")
                records[task, source, name] = {
                    "accuracy": float(accuracy.mean()),
                    "std": float(accuracy.std(ddof=1)) if len(accuracy) > 1 else 0.0,
                    "errors": wrong, "n": int(mask.sum()),
                }
    return graph, labels, splits, floors, records


def _distribution_counts(arrays, graph, labels, splits):
    if "own_cell" in labels:
        count = len(graph["cells"])
        all_counts = group(arrays, "coverage")
        raw = all_counts["candidate_cell_counts"]
        return {"outcomes": [("Before filtering", "#9aacc3", raw),
                              ("Retained samples", "#68b9f2", np.bincount(labels["own_cell"], minlength=count))],
                "splits": [(name, color, np.bincount(labels["own_cell"][splits[key]], minlength=count))
                           for key, name, color in (("train", "Training", "#68b9f2"),
                                                    ("validation", "Validation", "#f3bb65"),
                                                    ("test", "Evaluation", "#6ed1b5"))]}
    lookup = {tuple(cell): i for i, cell in enumerate(graph["cells"])}
    episodes = group(arrays, "episodes")
    if not episodes:
        raise ValueError("All attempted episodes are required to compare spawns and successes.")
    cells = np.floor(episodes["target_position"] / float(graph["cell_size_m"])).astype(int)
    try:
        ids = np.asarray([lookup[tuple(cell)] for cell in cells])
    except KeyError as exc:
        raise ValueError(f"Attempted target outside interior cells: {exc}") from exc
    count = len(graph["cells"])
    outcomes = [
        ("All spawned targets", "#9aacc3", np.bincount(ids, minlength=count)),
        ("Successful missions", "#6ed1b5", np.bincount(ids[episodes["success"]], minlength=count)),
        ("Collected samples", "#68b9f2", np.bincount(labels["target_cell"], minlength=count)),
    ]
    partition = [(name, color, np.bincount(labels["target_cell"][splits[key]], minlength=count))
                 for key, name, color in (("train", "Training", "#68b9f2"),
                                          ("validation", "Validation", "#f3bb65"),
                                          ("test", "Evaluation", "#6ed1b5"))]
    return {"outcomes": outcomes, "splits": partition}


def generate_legacy_report(run_dir, manifest, arrays, legacy=False, dpi=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import PercentFormatter

    run_dir = Path(run_dir).resolve()
    config = manifest["config"]
    tasks, sources = experiment_axes(config)
    own = "own_cell" in tasks
    dpi = dpi or manifest.get("report", {}).get("dpi", config["report"]["dpi"])
    graph, labels, splits, floors, records = _results(arrays, config)
    plt.rcParams.update({
        "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
        "text.color": TEXT, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 10, "axes.titlesize": 13,
        "axes.titlepad": 17, "axes.spines.top": False, "axes.spines.right": False,
    })

    def embed(figure, caption):
        stream = io.BytesIO()
        figure.savefig(stream, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        encoded = base64.b64encode(stream.getvalue()).decode("ascii")
        return f'<img class="chart" src="data:image/png;base64,{encoded}" alt="{html.escape(caption)}">'

    distribution_parts = []
    distributions = _distribution_counts(arrays, graph, labels, splits)
    cell_names = [",".join(f"{int(v):+d}" for v in cell) for cell in labels["cell_offsets"]]
    for category, series in distributions.items():
        legend = _legend([(f"{name} · {int(values.sum()):,}", color) for name, color, values in series])
        for measure in ("share", "count"):
            fig, axes = plt.subplots(1, 2, figsize=(16, 5.3), width_ratios=(1, 2.7), layout="constrained")
            for axis, by_floor in zip(axes, (True, False), strict=True):
                x = np.arange(len(floors) if by_floor else len(graph["cells"]))
                for i, (name, color, values) in enumerate(series):
                    values = np.asarray([values[graph["cells"][:, 2] + 1 == floor].sum() for floor in floors]) if by_floor else values
                    heights = 100 * values / values.sum() if measure == "share" else values
                    axis.bar(x + (i - (len(series) - 1) / 2) * .25, heights, width=.24, color=color, zorder=3)
                axis.set_xticks(x, [str(floor) for floor in floors] if by_floor else cell_names,
                                rotation=0 if by_floor else 90, fontsize=10 if by_floor else 8)
                axis.set_title(("Agent" if own else "Target") + (" storeys" if by_floor else " cells"))
                axis.set_xlabel("Building storey" if by_floor else "Cell XYZ offset from base · X, then Y, then storey")
                axis.set_ylabel("Share of each group (%)" if measure == "share" else "Number of samples" if own else "Number of episodes")
                axis.grid(axis="y", alpha=.7, zorder=0)
                axis.set_axisbelow(True)
                axis.set_ylim(bottom=0)
            hidden = "" if (category, measure) == ("outcomes", "count") else " hidden"
            distribution_parts.append(f'<div class="distribution-view" data-category="{category}" '
                                      f'data-measure="{measure}"{hidden}>' + legend + embed(fig, "Sample distribution by storey and cell") + "</div>")

    source_legend = _legend([(SOURCE_LABELS[s], COLORS[s]) for s in sources])
    required = arrays.get("dataset/target_required_agents") if not own else None
    required_values = np.unique(required) if required is not None else []
    from geo_experiments.target_report import required_label
    detail_groups = ([(f"required-{int(k)}", required_label(int(k))) for k in required_values]
                     + [(f"storey-{floor}", f"Storey {floor}") for floor in floors]) if len(required_values) else None
    groups = ["overall", *(f"required-{int(k)}" for k in required_values), *(f"storey-{floor}" for floor in floors)]
    fig, axes = plt.subplots(1, len(tasks), figsize=(7.5 * len(tasks), 4.8), layout="constrained", squeeze=False)
    for task, axis in zip(tasks, axes[0], strict=True):
        x = np.arange(len(groups), dtype=float)
        if len(required_values):
            x[1 + len(required_values):] += .9
            axis.axvline(x[1 + len(required_values)] - .9, color=GRID, linestyle="--")
        for i, source in enumerate(sources):
            rows = [records[task, source, name] for name in groups]
            axis.bar(x + (i - (len(sources) - 1) / 2) * .23, [100 * row["accuracy"] for row in rows], width=.22,
                     yerr=[100 * row["std"] for row in rows], color=COLORS[source],
                     error_kw={"ecolor": TEXT, "elinewidth": 1, "capsize": 3}, zorder=3)
        axis.set_title(TASK_LABELS[task])
        axis.set_xticks(x, ["Overall", *(required_label(int(k)) for k in required_values), *(f"Storey {floor}" for floor in floors)])
        axis.set_ylabel("Exact prediction accuracy (%)")
        axis.set_ylim(0, 100)
        axis.grid(axis="y", alpha=.7, zorder=0)
        axis.set_axisbelow(True)
    comparison = embed(fig, "Accuracy comparison, overall and by actual target storey")

    histories = { (task, source, seed): arrays[f"probes/{task}/{source}/seed_{seed}/history"]
                  for task in tasks for source in sources for seed in config["probe"]["seeds"] }
    loss_max = max(h[:, 3].max() for h in histories.values())
    epochs_max = max(h[-1, 0] for h in histories.values())
    fig, axes = plt.subplots(len(tasks), 2, figsize=(15, 3.5 * len(tasks)), layout="constrained", squeeze=False)
    for row, task in enumerate(tasks):
        for source in sources:
            for seed in config["probe"]["seeds"]:
                history = histories[task, source, seed]
                for column, metric in enumerate((3, 4)):
                    axes[row, column].plot(history[:, 0], history[:, metric], color=COLORS[source], alpha=.7, linewidth=1.3)
        for column, metric in enumerate(("Cross-entropy loss", "Accuracy")):
            axis = axes[row, column]
            axis.set_title(f"{TASK_LABELS[task]} · validation {metric.lower()}")
            axis.set_xlabel("Epoch")
            axis.set_ylabel(metric)
            axis.set_xlim(1, max(epochs_max, 2))
            axis.set_ylim(0, loss_max * 1.08 if column == 0 else 1)
            if column == 1:
                axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
            axis.grid(alpha=.5)
    learning = embed(fig, "Validation learning curves for every initialization seed")

    confusion = []
    for task in tasks:
        fig, axes = plt.subplots(1, len(sources), figsize=(5.33 * len(sources), 5.5), layout="constrained", squeeze=False)
        cell_task = task != "target_storey"
        class_names = cell_names if cell_task else [f"{int(v):+d}" for v in labels["storey_offsets"]]
        true = labels[task][splits["test"]]
        for source, axis in zip(sources, axes[0], strict=True):
            counts = np.zeros((len(class_names), len(class_names)), dtype=int)
            for seed in config["probe"]["seeds"]:
                predicted = arrays[f"probes/{task}/{source}/seed_{seed}/predicted"]
                np.add.at(counts, (true, predicted), 1)
            total = counts.sum(axis=1, keepdims=True)
            values = counts / np.maximum(total, 1)
            values = np.ma.array(values, mask=np.broadcast_to(total == 0, values.shape))
            cmap = LinearSegmentedColormap.from_list(source, [PANEL, COLORS[source]])
            cmap.set_bad(GRID)
            image = axis.imshow(values, vmin=0, vmax=1, cmap=cmap, interpolation="nearest")
            axis.set_title(SOURCE_LABELS[source], color=COLORS[source])
            axis.set_xticks(np.arange(len(class_names)), class_names, rotation=90, fontsize=6 if cell_task else 11)
            axis.set_yticks(np.arange(len(class_names)), class_names, fontsize=6 if cell_task else 11)
            axis.set_xlabel("Predicted cell" if cell_task else "Predicted storey offset")
            axis.set_ylabel("Actual cell" if cell_task else "Actual storey offset")
            fig.colorbar(image, ax=axis, shrink=.75, label="Share of actual class", format=PercentFormatter(xmax=1, decimals=0))
        confusion.append(f'<h3>{TASK_LABELS[task]}</h3>' + embed(fig, f"{TASK_LABELS[task]} confusion matrices"))

    collection = manifest["collection"]
    counts = "".join(f'<span class="pill"><b>{len(splits[key]):,}</b> {name}</span>' for key, name in
                     (("train", "training"), ("validation", "validation"), ("test", "evaluation")))
    name = html.escape(config["run"]["name"])
    level = html.escape(config["policy"]["level"])
    seeds = len(config["probe"]["seeds"])
    filtering_section = ""
    if own:
        collection_note = (f"{collection['num_samples']:,} retained agent snapshots from {collection['attempted_episodes']:,} episodes, "
                           f"regardless of outcome. Every {collection['sampling_steps']} steps ({collection['sampling_seconds']:.2f} s). "
                           f"{collection['duplicate_samples']:,} duplicates and {collection['cross_split_duplicates']:,} cross-split histories excluded.")
        distribution_options = '<option value="outcomes">Before filtering &amp; retained samples</option>'
        measure_count = "Sample counts"
        distribution_note = ("Shares normalize each group separately. Candidate counts include repeated visits before filtering; "
                             "retained counts reflect exact-history deduplication. These are visited agent cells, not uniformly sampled target positions.")
        if collection.get("uniform_cell_buckets"):
            collection_note += " Uniform per-cell buckets in every split."
            distribution_note = ("Shares normalize each group separately. Before-filter counts show natural visits including duplicates; "
                                 "retained samples fill uniform eligible-cell buckets after history filtering. Counts per eligible cell differ by at most one "
                                 "in each split and overall. This equal-cell evaluation distribution differs from older visitation-weighted runs.")
        excluded_cells = collection.get("excluded_cells", [])
        if excluded_cells:
            excluded_text = ", ".join(str(tuple(cell)) for cell in excluded_cells)
            collection_note += (f" Overall results cover {collection['eligible_cell_count']} of "
                                f"{collection['total_cell_count']} actual cells. Excluded absolute grid XYZ: {excluded_text}.")
            distribution_note += (" Excluded cells stay visible in candidate coverage but have no retained samples. "
                                  "The prediction head and distance graph still include the full building.")
        scope_note = "actual agent storey"
        rows = []
        metrics = [("Candidate snapshots inside cells", "candidate_samples"),
                           ("Excluded outside interior cells", "outside_interior_samples"),
                           ("Removed repeated history/cell pairs", "duplicate_samples"),
                           ("Excluded histories in earlier splits", "cross_split_duplicates"),
                           ("Unused after pool quota reached", "quota_unused_samples"),
                           ("Candidates before target discovery", "candidate_before_found"),
                           ("Candidates after target discovery", "candidate_after_found"),
                           ("Retained before target discovery", "retained_before_found"),
                           ("Retained after target discovery", "retained_after_found")]
        if "full_bucket_rejections" in collection:
            metrics.insert(4, ("Rejected because cell bucket was full", "full_bucket_rejections"))
        if "excluded_cell_samples" in collection:
            metrics.insert(1, ("Rejected because actual cell was excluded", "excluded_cell_samples"))
        for title, key in metrics:
            rows.append(f'<tr><td>{title}</td>' + numeric_cell(title, "Snapshots", collection[key],
                        unit="snapshots", version="own-cell-filter/v1") + '</tr>')
        for pool, count in collection["pool_episode_counts"].items():
            title = f"{pool.capitalize()} pool episodes"
            rows.append(f'<tr><td>{title}</td>' + numeric_cell(title, "Episodes", count,
                        unit="episodes", version="own-cell-filter/v1") + '</tr>')
        filtering_section = ('<section><h2>Collection and filtering</h2><div class="table-scroll">'
                             '<table class="audit-table"><thead><tr><th>Metric</th><th>Count</th></tr></thead><tbody>'
                             + ''.join(rows) + '</tbody></table></div><p class="note">Each episode belongs to one pool. '
                             'Exact combined-input histories retained in an earlier pool cannot enter later pools. '
                             'Within a pool, one history/cell pair is kept, preserving conflicting labels. '
                             'Candidate counts include surplus final-batch snapshots; excluded duplicate counts refer '
                             'to candidates considered before reaching each quota. Evaluation therefore tests new '
                             'retained histories, rather than raw time-weighted visitation.</p></section>')
    else:
        collection_note = (f"{collection['successful_episodes']:,} successful missions / {collection['attempted_episodes']:,} attempted "
                           f"({100 * collection['success_rate']:.1f}%). {collection['successful_missing_snapshot']:,} successful missions missing a snapshot.")
        distribution_options = '<option value="outcomes">Spawns, successes &amp; collected samples</option>'
        measure_count = "Episode counts"
        distribution_note = ("Share normalizes each group separately to 100%, so changes in shape reveal selection by mission success. "
                             "All interior cells are shown, including cells with no targets. Spawn exclusions and available free space affect the expected distribution. "
                             "Collected samples stop at the requested total; all missions in the final batch are included in spawns and successes.")
        scope_note = "actual target storey"
    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark"><title>Geo diagnostics · {name}</title><style>
:root{{color-scheme:dark;--bg:{BACKGROUND};--panel:{PANEL};--text:{TEXT};--muted:{MUTED};--line:{GRID};}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1480px;margin:auto;padding:36px 44px 70px}}header{{padding-bottom:24px;border-bottom:1px solid var(--line)}}
.eyebrow{{color:#68b9f2;font-size:12px;letter-spacing:.14em;text-transform:uppercase}}h1{{font-size:32px;margin:8px 0 12px;line-height:1.2}}
h2{{font-size:22px;margin:0 0 12px}}h3{{font-size:17px;margin:20px 0 8px}}p{{margin:8px 0 16px}}section{{margin-top:34px}}
.muted,.note{{color:var(--muted)}}.note{{font-size:13px;max-width:1050px}}.pills{{display:flex;gap:10px;flex-wrap:wrap;margin-top:20px}}
.pill{{background:var(--panel);border:1px solid var(--line);padding:8px 14px;border-radius:8px;color:var(--muted)}}.pill b{{color:var(--text);margin-right:5px}}
.controls{{display:flex;gap:18px;flex-wrap:wrap;align-items:center;margin-bottom:14px}}label{{color:var(--muted);font-size:13px}}
select{{margin-left:7px;background:var(--panel);color:var(--text);border:1px solid #425571;padding:7px 10px;border-radius:6px}}
.legend{{display:flex;gap:22px;flex-wrap:wrap;margin:10px 0 16px;color:var(--muted);font-size:13px}}.legend span{{display:flex;gap:7px;align-items:center}}
.legend i,.source i{{display:inline-block;width:10px;height:10px;border-radius:3px;flex-shrink:0}}
.chart{{display:block;width:100%;height:auto;border-radius:10px;background:var(--panel)}}[hidden]{{display:none!important}}
.table-scroll{{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--panel)}}table{{border-collapse:collapse;width:100%;min-width:660px;table-layout:fixed}}
th{{text-align:left;color:var(--muted);font-size:12px;font-weight:600;background:#18263c;padding:13px 20px}}th:first-child{{width:31%}}th:nth-child(2){{width:16%}}
td{{padding:12px 20px;vertical-align:middle}}.overall td{{border-top:1px solid var(--line);padding-top:20px}}.overall:first-child td{{border-top:0}}
.source{{display:flex;gap:8px;align-items:center;margin-top:7px;font-size:13px;color:var(--muted)}}.accuracy{{font-size:25px;font-weight:650;font-variant-numeric:tabular-nums}}
.histogram{{display:block;width:100%;height:auto;max-width:490px}}.perfect{{color:#6ed1b5;font-size:13px}}
.detail-row>td{{padding-top:0;padding-bottom:14px}}summary{{cursor:pointer;color:var(--muted);font-size:12px;width:fit-content}}summary:hover{{color:#68b9f2}}
.secondary{{background:#142138;border-left:2px solid #425571;border-radius:0 6px 6px 0;margin:12px 0 3px;padding:5px 0}}
.secondary table{{min-width:620px}}.secondary td:first-child{{width:31%;color:var(--muted);font-size:13px}}.secondary td:nth-child(2){{width:16%;font-size:20px}}
.secondary tr+tr td{{border-top:1px solid var(--line)}}
.report-controls{{margin-top:16px}}a{{color:#68b9f2}}.metric-delta{{font-size:13px;font-weight:500;color:#a7c5ee;white-space:nowrap}}
.metric-delta.positive{{color:#6ed1b5}}.metric-delta.negative{{color:#f0838a}}
.audit-table{{min-width:750px;table-layout:auto}}.audit-table th:first-child{{width:34%}}.audit-table th:nth-child(2){{width:auto}}
.audit-table td{{font-variant-numeric:tabular-nums;border-top:1px solid var(--line)}}.audit-detail{{margin-top:20px}}
@media(max-width:800px){{main{{padding:24px 14px 40px}}h1{{font-size:26px}}td,th{{padding-left:12px;padding-right:12px}}.legend{{gap:12px}}}}
</style></head><body><main><header><div class="eyebrow">Geo experiments · frozen-policy diagnostics</div>
<h1>{name}</h1><p class="muted">{level} · {'own-cell' if own else 'target'} localization · {seeds} initialization seeds per probe/input</p>
<p class="note">{collection_note}</p>
<div class="pills">{counts}</div>{report_controls(run_dir)}</header>
<section><h2>Sample distributions</h2><div class="controls">
<label>Compare<select id="distribution-category" onchange="updateDistribution()">{distribution_options}<option value="splits">Training, validation &amp; evaluation</option></select></label>
<label>Measure<select id="distribution-measure" onchange="updateDistribution()"><option value="count">{measure_count}</option><option value="share">Share (%)</option></select></label></div>
{''.join(distribution_parts)}
<p class="note">{distribution_note}</p></section>
<section><h2>Held-out results</h2>{_table(records, floors, tasks, sources, detail_groups=detail_groups)}
<p class="note">Accuracy is the mean across initialization seeds. The Measure selector above also controls these histograms.
They include only incorrect predictions, pooled across seeds: counts are prediction counts, and shares sum to 100% of errors.
Count scales are shared across inputs and storey details for each probe; hover a bar for its count and share.
Cell error counts the shortest valid neighbouring-cell route through openings,
respecting solid walls. {'Storey error is the number of storeys off. ' if not own else ''}Per-storey details group by the {scope_note}.</p></section>
{audit_section(manifest.get("ambiguity_audit"))}
{filtering_section}
<section><h2>Overall and per-storey accuracy</h2>{source_legend}{comparison}
<p class="note">Error bars show standard deviation across initialization seeds, using the same held-out episodes for every fit.</p></section>
<section><h2>Validation learning curves</h2>{source_legend}{learning}
<p class="note">Each line is one decoder initialization; colours identify the input feature source. An epoch is one pass through probe training examples. These curves train the small MLP, while the drone policy stays frozen.
Accuracy is the share of exact correct labels, so higher is better. Cross-entropy loss measures probability assigned to the true label, so lower is better; confident mistakes can raise loss even when accuracy stays flat.
Normally loss falls and accuracy rises before levelling off. Early stopping selects validation-loss improvements under min_delta and stops after the configured patience. Evaluation episodes are used only after this selection.</p></section>
<section><h2>Confusion matrices</h2><p class="note">Rows are actual classes; columns are predictions. Each row sums to 100%, pooled across seeds.
Colours use the same 0–100% scale; grey rows have no evaluation samples. Cell coordinates and storey offsets are relative to the base.</p>{''.join(confusion)}</section>
</main><script>function updateDistribution(){{const category=document.getElementById('distribution-category').value;
const measure=document.getElementById('distribution-measure').value;document.querySelectorAll('.distribution-view').forEach(view=>{{
view.hidden=view.dataset.category!==category||view.dataset.measure!==measure;}});
document.querySelectorAll('.error-histogram-view').forEach(view=>{{view.hidden=view.dataset.measure!==measure;}});}}</script></body></html>'''
    if not own:
        # Fresh single-map collections can show required-agent counts and success
        # highlights. Historical archives lack those fields; never invent counts.
        if all(k in arrays for k in ("episodes/target_required_agents", "episodes/delivered", "dataset/target_required_agents")):
            from geo_experiments.target_report import distributions
            from geo_experiments.report_layout import distribution_panel, DISTRIBUTION_STYLE
            chart_graph = {**graph, "map_names": np.asarray([config["policy"]["level"]])}
            requirements = np.unique(arrays["episodes/target_required_agents"])
            replacement = distribution_panel(distributions(arrays, chart_graph, labels, splits, floors, requirements), single_map=True,
                                             eligibility=config.get("protocol", {}).get("eligibility", "success"))
            begin = document.index('<section><h2>Sample distributions</h2>')
            end = document.index('</section>', begin) + len('</section>')
            document = document[:begin] + replacement + document[end:]
            document = document.replace('</head>', f'<style>{DISTRIBUTION_STYLE}</style></head>')
            document = document.replace('<h2>Overall and per-storey accuracy</h2>', '<h2>Overall accuracy</h2>')
            script = document.rfind('<script>function updateDistribution()')
            if script >= 0:
                end = document.index('</script>', script) + len('</script>')
                document = document[:script] + document[end:]
        begin = document.index('<section><h2>Validation learning curves</h2>')
        confusion_begin = document.index('<section><h2>Confusion matrices</h2>', begin)
        end = document.index('</section>', confusion_begin) + len('</section>')
        document = document[:begin] + document[confusion_begin:end] + document[begin:confusion_begin] + document[end:]
        from geo_experiments.auditing import footer_note
        note = footer_note(manifest.get("ambiguity_audit"))
        if note:
            document = document.replace('</main>', f'<footer><p class="note">{html.escape(note)}</p></footer></main>')
    document = decorate_report(document, run_dir, manifest)
    # Save the archive/manifest before deleting any original generated files.
    if legacy:
        save_data(run_dir, arrays)
        update_run(run_dir, **{k: v for k, v in manifest.items() if k != "format"})
    temporary = run_dir / ".report.tmp.html"
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(run_dir / "report.html")
    update_run(run_dir, report={"dpi": dpi, "renderer_sha256": fingerprint(__file__)})
    if legacy:
        prune_legacy(run_dir)
    build_dashboard()
    print(f"[report] {run_dir / 'report.html'}", flush=True)

