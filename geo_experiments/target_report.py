"""Configuration-aware target reports for shared coordinates, experts and map+cell."""
from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import numpy as np

from geo_experiments.artifacts import group, update_run
from geo_experiments.auditing import footer_note
from geo_experiments.common import SOURCES, SOURCE_LABELS, experiment_axes, fingerprint
from geo_experiments.dashboard import build_dashboard, decorate_report, numeric_cell, report_controls
from geo_experiments.labels import DECODERS, cell_errors, display_predictions
from geo_experiments.report_components import COLORS, GRID, MUTED, PANEL, TEXT, TASK_LABELS, _legend, _table
from geo_experiments.report_layout import distribution_panel, document
from geo_experiments.console import terminal_print as print


def required_label(value):
    return f"{value} required agent{'s' if value != 1 else ''}" if value > 0 else "Not estimated"


def results(arrays, config):
    graph, labels, splits = (group(arrays, key) for key in ("geometry", "labels", "splits"))
    test = splits["test"]
    floors = np.unique(graph["cells"][:, 2] + 1)
    requirements = np.unique(group(arrays, "episodes")["target_required_agents"])
    detail = [(f"required-{int(k)}", required_label(int(k))) for k in requirements]
    detail += [(f"storey-{int(f)}", f"Storey {f}") for f in floors]
    show_maps = not (len(graph["map_names"]) >= 20 and config["policy"].get("map_source") == "validation")
    if show_maps:
        detail += [(f"map-{m}", str(name)) for m, name in enumerate(graph["map_names"])]
    masks = {"overall": np.ones(len(test), dtype=bool)}
    masks.update({f"required-{int(k)}": labels["required_agents"][test] == k for k in requirements})
    masks.update({f"storey-{int(f)}": labels["actual_storey"][test] == f for f in floors})
    if show_maps:
        masks.update({f"map-{m}": labels["map_id"][test] == m for m in range(len(graph["map_names"]))})
    records, diagnostics = {}, {}
    decoder = config["protocol"]["decoder"]
    tasks, sources = experiment_axes(config)
    for task in tasks:
        actual = labels["coordinate_cell" if task == "target_cell" and decoder == "shared_cell" else task][test]
        for source in sources:
            predicted = np.stack([arrays[f"probes/{task}/{source}/seed_{s}/predicted"] for s in config["probe"]["seeds"]])
            if predicted.shape != (len(config["probe"]["seeds"]), len(test)):
                raise ValueError(f"Held-out prediction shape mismatch: {task}/{source}")
            correct = predicted == actual[None]
            if task == "target_cell":
                decoded = np.stack([display_predictions(p, labels, graph, test, decoder) for p in predicted])
                errors = np.stack([cell_errors(p, labels[task][test], graph) for p in decoded])
                if decoder == "map_cell":
                    diagnostics[source] = {
                        "Map identity": float((graph["cell_map_id"][predicted] == labels["map_id"][test][None]).mean()),
                        "Coordinate only (ignores map identity)": float((graph["cell_coordinate_id"][predicted] == labels["coordinate_cell"][test][None]).mean())}
            else:
                errors = np.abs(labels["storey_offsets"][predicted] - labels["storey_offsets"][actual][None])
            for name, mask in masks.items():
                accuracy = correct[:, mask].mean(axis=1) if mask.any() else np.full(len(predicted), np.nan)
                records[task, source, name] = {"accuracy": float(accuracy.mean()),
                    "std": float(accuracy.std(ddof=1)) if len(accuracy) > 1 else 0.0,
                    "errors": errors[:, mask][~correct[:, mask]], "n": int(mask.sum())}
    return graph, labels, splits, floors, requirements, detail, records, diagnostics


def distributions(arrays, graph, labels, splits, floors, requirements):
    episodes, dataset = (group(arrays, k) for k in ("episodes", "dataset"))
    single_map = len(graph["map_names"]) == 1
    names = {"storey": [str(int(v)) for v in floors], "required": [required_label(int(v)) for v in requirements]}
    if single_map:
        names["cell"] = [",".join(map(str, c)) for c in graph["cells"]]
    output = {"labels": names, "outcomes": [], "splits": []}

    def series(name, color, positions, required, success, ids=None):
        cells = np.floor(positions / float(graph["cell_size_m"])).astype(int)
        row = {"name": name, "color": color, "total": len(cells)}
        for kind, values, categories in (("storey", cells[:, 2] + 1, floors), ("required", required, requirements)):
            row[kind] = [int((values == v).sum()) for v in categories]
            row["success_" + kind] = [int(((values == v) & success).sum()) for v in categories]
        if single_map:
            row["cell"] = [int(np.all(cells == c, axis=1).sum()) for c in graph["cells"]]
            row["success_cell"] = [int((np.all(cells == c, axis=1) & success).sum()) for c in graph["cells"]]
        return row

    for name, color, mask in (("All spawned targets", "#9aacc3", np.ones(len(episodes["success"]), dtype=bool)),
                              ("Delivered to base", "#6ed1b5", episodes["delivered"])):
        output["outcomes"].append(series(name, color, episodes["target_position"][mask],
              episodes["target_required_agents"][mask], episodes["success"][mask]))
    output["outcomes"].append(series("Collected samples", "#68b9f2", dataset["target_position"],
            dataset["target_required_agents"], dataset["success"]))
    for split, name, color in (("train", "Training", "#68b9f2"), ("validation", "Validation", "#f3bb65"), ("test", "Evaluation", "#6ed1b5")):
        rows = splits[split]
        output["splits"].append(series(name, color, dataset["target_position"][rows],
                dataset["target_required_agents"][rows], dataset["success"][rows]))
    return output


def detailed_audit(summary, decoder="map_cell"):
    if summary and summary.get("format") == "swarmecho-target-audit/v2":
        from geo_experiments.collision_report import snapshot_sections
        sections = snapshot_sections(summary, decoder, summary.get("map_names", []))
        if summary.get("history"):
            sections += detailed_audit(summary["history"], decoder).replace('<h2>Ambiguity audit</h2>', '<h2>Finder observation histories</h2>')
        return sections
    if not summary or summary.get("status") != "full":
        return ""
    rows = []
    inputs = summary.get("by_decoder", {}).get(decoder, summary["inputs"])
    for source, data in inputs.items():
        title = SOURCE_LABELS[source]
        values = [numeric_cell(title, column, data[key], unit=unit, version=f"ambiguity/{decoder}/v1") for column, key, unit in
                 (("Distinct input groups", "distinct_states", "groups"), ("Conflicting groups", "mixed_cell_groups", "groups"),
                  ("Conflicting-label pairs", "cross_cell_pairs", "pairs"))]
        rows.append(f'<tr><td>{title}</td>' + ''.join(values) + '</tr>')
    history_rows = []
    for source, data in inputs.items():
        for stage, count in (data["history_matches"] or {}).items():
            title = f"{SOURCE_LABELS[source]} / {stage}"
            history_rows.append(f'<tr><td>{html.escape(title)}</td>' + numeric_cell(title, "Identical-history conflicting pairs", count, unit="pairs", version=f"ambiguity/{decoder}/v1") + '</tr>')
    fidelity = '; '.join(f"{SOURCE_LABELS[s]} {n:,}/{summary['samples']:,}" for s, n in summary["exact_replayed_snapshots"].items())
    scope = {"map_cell": "Conflicts mean different map + absolute-cell labels within an exact latent group.",
             "shared_cell": "Conflicts mean different absolute-cell coordinates within an exact latent group; different maps with the same coordinate are not conflicting labels.",
             "map_specific": "Groups are conditioned on the expert's map identity. Only different cells within the same map and exact latent group conflict; cross-map equality is not expert ambiguity."}[decoder]
    reason = ('Configuration forced full history analysis of the diagnostic subset. ' if summary.get("reason") == "forced"
              else 'The cheap screen triggered a replay of its diagnostic subset with full input histories. ')
    return ('<section><h2>Ambiguity audit</h2><p class="note">' + reason +
            f'{scope} Groups below are recomputed from replayed features. '
            f'Exact reproduction of the original snapshots: {fidelity}. Different batch shapes can alter floating-point results. '
            f'{summary.get("skipped_missing_snapshots", 0):,} replayed episodes lacked complete snapshots and were excluded. '
            f'Histories sampled every {summary["every"]} step(s); base-sender histories are unavailable. '
            'Identical sampled histories do not establish equality at skipped steps.</p>'
            '<div class="table-scroll"><table class="audit-table"><thead><tr><th>Input</th><th>Distinct input groups</th><th>Conflicting groups</th><th>Conflicting-label pairs</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table></div><h3>Input histories within conflicting pairs</h3>'
            '<div class="table-scroll"><table class="audit-table"><thead><tr><th>Input/history</th><th>Identical-history conflicting pairs</th></tr></thead><tbody>'
            + ''.join(history_rows) + '</tbody></table></div></section>')


def generate_target_report(run_dir, manifest, arrays, dpi=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import PercentFormatter

    run_dir = Path(run_dir)
    config = manifest["config"]
    dpi = dpi or config["report"]["dpi"]
    tasks, sources = experiment_axes(config)
    decoder = config["protocol"]["decoder"]
    graph, labels, splits, floors, requirements, detail, records, diagnostics = results(arrays, config)
    plt.rcParams.update({"figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
        "text.color": TEXT, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    def embed(fig, caption, css="chart"):
        stream = io.BytesIO()
        fig.savefig(stream, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return f'<img class="{css}" src="data:image/png;base64,{base64.b64encode(stream.getvalue()).decode()}" alt="{html.escape(caption)}">'

    collection = manifest["collection"]
    pills = ''.join(f'<span class="pill"><b>{len(splits[s]):,}</b> {name}</span>' for s, name in
                    (("train", "training"), ("validation", "validation"), ("test", "evaluation")))
    name = html.escape(config["run"]["name"])
    condition = {"map_specific": "Map identity selects one expert; each expert predicts a cell within its map.",
                 "map_cell": "One decoder predicts both map identity and cell; exact accuracy requires both to be correct.",
                 "shared_cell": "One decoder predicts absolute XYZ cell coordinates across maps; it does not predict map identity."}[decoder]
    body = (f'<header><div class="eyebrow">Geo experiments · frozen-policy diagnostics</div><h1>{name}</h1>'
            f'<p class="muted">{DECODERS[decoder]} · {len(graph["map_names"])} maps · {len(config["probe"]["seeds"])} initialization seeds per input</p>'
            f'<p class="note">{collection["delivered_episodes"]:,} deliveries / {collection["attempted_episodes"]:,} attempted episodes; '
            f'{collection["successful_episodes"]:,} chain successes. {condition} No policy gradients or map IDs are added to the features.</p>'
            f'<div class="pills">{pills}</div>{report_controls(run_dir)}</header>')
    if config["policy"].get("map_source") == "validation":
        body += ('<p class="note">These frozen buildings were held out from policy training. '
                 'Fresh probe episodes are collected on them, with each building represented in training, '
                 'validation and evaluation. This measures coordinate decoding on policy-held-out buildings; '
                 'it does not measure transfer to buildings excluded from decoder training. '
                 'Per-map evaluation counts are small; use the pooled results for the main interpretation.</p>')
    body += distribution_panel(distributions(arrays, graph, labels, splits, floors, requirements),
                               single_map=len(graph["map_names"]) == 1, eligibility=collection["eligibility"])
    body += '<section><h2>Held-out results</h2>' + _table(records, floors, tasks, sources, detail_groups=detail, metric_version="suite-cell-accuracy/v1")
    detail_order = 'target-required agents, then actual target storey' + (', then map' if any(k.startswith('map-') for k, _ in detail) else '')
    body += (f'<p class="note">Accuracy averages decoder initialization seeds over the same held-out episodes. Details group by {detail_order}. '
             'Histograms contain incorrect predictions only, pooled across seeds; the Measure selector also switches their counts/shares. '
             'Cell distance uses the true map’s obstacle-aware cell graph. Wrong-map predictions have their own bucket and are never assigned a fabricated route distance. '
             'No route includes disconnected or invalid predicted cells. Map identity is supplied for expert selection and for evaluating shared coordinates, but must be inferred by the map+cell decoder.</p>')
    if diagnostics:
        body += '<div class="table-scroll"><table class="audit-table"><thead><tr><th>Input / diagnostic</th><th>Accuracy</th></tr></thead><tbody>'
        for source, values in diagnostics.items():
            for metric, value in values.items():
                title = f"{SOURCE_LABELS[source]} / {metric}"
                body += f'<tr><td>{title}</td>' + numeric_cell(title, "Accuracy", value * 100, unit="percent", decimals=1, suffix="%") + '</tr>'
        body += '</tbody></table></div><p class="note">Coordinate-only accuracy ignores which map was predicted; it is a diagnostic, not the joint task’s headline accuracy.</p>'
    body += '</section>'
    legend = _legend([(SOURCE_LABELS[s], COLORS[s]) for s in sources])

    # Required-agent categories precede a visually separated storey block.
    groups = ["overall", *(f"required-{int(k)}" for k in requirements), *(f"storey-{int(f)}" for f in floors)]
    captions = ["Overall", *(required_label(int(k)).replace(" required", "") for k in requirements), *(f"Storey {f}" for f in floors)]
    x = np.arange(len(groups), dtype=float)
    x[1 + len(requirements):] += .9
    fig, axes = plt.subplots(len(tasks), 1, figsize=(15, 4.7 * len(tasks)), layout="constrained", squeeze=False)
    for task, axis in zip(tasks, axes[:, 0], strict=True):
        for i, source in enumerate(sources):
            rows = [records[task, source, g] for g in groups]
            axis.bar(x + (i - 1) * .23, [100 * r["accuracy"] for r in rows], width=.22,
                     yerr=[100 * r["std"] for r in rows], color=COLORS[source], error_kw={"ecolor": TEXT, "elinewidth": 1, "capsize": 3}, zorder=3)
        axis.axvline(x[1 + len(requirements)] - .9, color=GRID, linestyle="--")
        axis.set_xticks(x, captions, rotation=15, ha="right")
        axis.set_title(TASK_LABELS[task])
        axis.set_ylabel("Exact accuracy (%)")
        axis.set_ylim(0, 100)
        axis.grid(axis="y", zorder=0, alpha=.7)
        axis.set_axisbelow(True)
    body += '<section><h2>Overall accuracy</h2>' + legend + embed(fig, "Accuracy by target-required agents and storey")
    body += '<p class="note">Overall is episode-weighted. Map quotas are equal up to rounding in every split. Error bars show decoder-seed standard deviations, not policy-seed uncertainty or confidence intervals. Empty categories have no bar.</p></section>'

    def matrix(axis, true, predictions, values, source, title, boundaries=None):
        count = len(values)
        counts = np.zeros((count, count), dtype=np.int64)
        for predicted in predictions:
            np.add.at(counts, (true, predicted), 1)
        total = counts.sum(axis=1, keepdims=True)
        normalized = np.ma.array(counts / np.maximum(total, 1), mask=np.broadcast_to(total == 0, counts.shape))
        cmap = LinearSegmentedColormap.from_list(source, [PANEL, COLORS[source]])
        cmap.set_bad(GRID)
        image = axis.imshow(normalized, vmin=0, vmax=1, cmap=cmap, interpolation="nearest")
        axis.set_title(title, color=COLORS[source])
        axis.set_xticks(np.arange(count), values, rotation=90, fontsize=6 if count < 40 else 5)
        axis.set_yticks(np.arange(count), values, fontsize=6 if count < 40 else 5)
        axis.set_xlabel("Predicted cell" if len(values) > 3 else "Predicted storey")
        axis.set_ylabel("Actual cell" if len(values) > 3 else "Actual storey")
        for border in boundaries or []:
            axis.axvline(border - .5, color=TEXT, linewidth=.8)
            axis.axhline(border - .5, color=TEXT, linewidth=.8)
        return image

    body += '<section><h2>Confusion matrices</h2><p class="note">Rows are actual classes, columns predictions. Each supported row sums to 100% across initialization seeds; grey rows have no held-out examples. Coordinates are absolute XYZ cells. '
    body += {"map_cell": "Both axes contain every map–cell class, grouped by map. Off-diagonal map blocks show wrong-building predictions.",
             "shared_cell": "The matrix pools the shared coordinate classes across maps; duplicating it into a map–cell grid would imply a map prediction this decoder does not make.",
             "map_specific": "Each map has its own matrix. A combined matrix would have empty off-map blocks by construction, since map identity selects the expert."}[decoder] + '</p>'
    if decoder == "map_cell":
        body += '<p class="note">' + '; '.join(f'M{m + 1} = {html.escape(str(n))}' for m, n in enumerate(graph["map_names"])) + '</p>'
    for task in tasks:
        if task == "target_cell" and decoder in ("map_cell", "map_specific"):
            for source in sources:
                predictions = [arrays[f"probes/{task}/{source}/seed_{s}/predicted"] for s in config["probe"]["seeds"]]
                if decoder == "map_cell":
                    names = [f"M{m + 1} · {','.join(map(str, cell))}" for m, cell in zip(graph["cell_map_id"], graph["cells"], strict=True)]
                    fig, axis = plt.subplots(figsize=(17, 16), layout="constrained")
                    image = matrix(axis, labels[task][splits["test"]], predictions, names, source, SOURCE_LABELS[source], list(graph["map_offsets"][1:-1]))
                    fig.colorbar(image, ax=axis, shrink=.7, format=PercentFormatter(xmax=1), label="Share of actual class")
                    content = '<div class="matrix-scroll"><div class="matrix-large">' + embed(fig, f"{SOURCE_LABELS[source]} map + cell confusion matrix") + '</div></div>'
                else:
                    count = len(graph["map_names"])
                    columns = min(3, count)
                    fig, axes = plt.subplots(int(np.ceil(count / columns)), columns, figsize=(5.5 * columns, 5.3 * int(np.ceil(count / columns))), squeeze=False, layout="constrained")
                    for m, axis in enumerate(axes.flat):
                        if m >= count:
                            axis.set_visible(False)
                            continue
                        start, stop = graph["map_offsets"][m:m + 2]
                        mask = labels["map_id"][splits["test"]] == m
                        names = [','.join(map(str, c)) for c in graph["cells"][start:stop]]
                        image = matrix(axis, labels[task][splits["test"]][mask] - start,
                                       [p[mask] - start for p in predictions], names, source, str(graph["map_names"][m]))
                    fig.colorbar(image, ax=list(axes.flat), shrink=.6, format=PercentFormatter(xmax=1), label="Share of actual class")
                    content = embed(fig, f"{SOURCE_LABELS[source]} individual-map confusion matrices")
                body += f'<details class="matrix" open><summary>{SOURCE_LABELS[source]}</summary>{content}</details>'
        else:
            values = graph["coordinate_values"] if task == "target_cell" else labels["storey_offsets"] + 1
            names = [','.join(map(str, c)) for c in values] if task == "target_cell" else list(map(str, values))
            actual = labels["coordinate_cell" if task == "target_cell" else task][splits["test"]]
            fig, axes = plt.subplots(1, len(sources), figsize=(5.3 * len(sources), 5.4), squeeze=False, layout="constrained")
            for source, axis in zip(sources, axes[0], strict=True):
                image = matrix(axis, actual, [arrays[f"probes/{task}/{source}/seed_{s}/predicted"] for s in config["probe"]["seeds"]], names, source, SOURCE_LABELS[source])
                fig.colorbar(image, ax=axis, shrink=.7, format=PercentFormatter(xmax=1))
            body += embed(fig, f"{TASK_LABELS[task]} confusion matrices")
    body += '</section>'

    body += '<section><h2>Validation learning curves</h2>' + legend
    views = list(range(len(graph["map_names"]))) if decoder == "map_specific" else [None]
    if len(views) > 1:
        body += '<div class="controls"><label>Map<select onchange="document.querySelectorAll(\'.learning-view\').forEach(v=>v.hidden=v.dataset.map!==this.value)">' + ''.join(f'<option value="{m}">{html.escape(str(graph["map_names"][m]))}</option>' for m in views) + '</select></label></div>'
    for m in views:
        fig, axes = plt.subplots(len(tasks), 2, figsize=(15, 3.5 * len(tasks)), squeeze=False, layout="constrained")
        for row, task in enumerate(tasks):
            classes = len(labels["storey_offsets"]) if task == "target_storey" else len(graph["coordinate_values"]) if decoder == "shared_cell" else len(graph["cells"]) if m is None else int(np.diff(graph["map_offsets"])[m])
            for source in sources:
                for seed in config["probe"]["seeds"]:
                    prefix = f"probes/{task}/{source}/seed_{seed}"
                    if m is not None:
                        prefix += f"/maps/{m}"
                    history = arrays[prefix + "/history"]
                    selected_epoch = arrays.get(prefix + "/best_epoch", history[np.argmin(history[:, 3]), 0])
                    best = int(np.flatnonzero(history[:, 0] == selected_epoch)[0])
                    for col, metric in enumerate((3, 4)):
                        axes[row, col].plot(history[:, 0], history[:, metric], color=COLORS[source], alpha=.65)
                        axes[row, col].scatter(history[best, 0], history[best, metric], color=COLORS[source], s=18, zorder=4)
            axes[row, 0].axhline(np.log(classes), color=MUTED, linestyle=":", alpha=.6, label="Uniform guess")
            axes[row, 0].set_ylabel("Cross-entropy loss (lower is better)")
            axes[row, 1].set_ylabel("Accuracy (higher is better)")
            axes[row, 1].set_ylim(0, 1)
            axes[row, 1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
            for axis, metric in zip(axes[row], ("loss", "accuracy"), strict=True):
                axis.set_title(f"{TASK_LABELS[task]} · validation {metric}")
                axis.set_xlabel("Epoch (one pass through probe training examples)")
                axis.grid(alpha=.4)
        body += f'<div class="learning-view" data-map="{m}"' + (' hidden' if m not in (None, 0) else '') + '>' + embed(fig, "Validation loss and accuracy during decoder fitting") + '</div>'
    body += ('<p class="note">Each line is one decoder initialization; colours identify the input feature source. These curves train the small probe, not the frozen drone policy. '
             'Accuracy counts exact correct labels. Cross-entropy measures how much probability the decoder assigns to the correct label: lower is better, and confident wrong predictions are costly. '
             'The dotted loss line is a uniform guess across the available classes; class imbalance can make other uninformed baselines stronger. '
             'Expect loss to fall and accuracy to rise before levelling off. Rising loss with flat accuracy can mean increasingly confident mistakes. '
             'Dots mark the selected epoch: the best validation-loss checkpoint under the configured min_delta improvement rule. Training stops after the configured patience without another qualifying improvement. '
             'Evaluation episodes are used only after selection. Loss magnitudes across 27-class and 135-class tasks are not directly comparable.</p></section>')
    body += detailed_audit(manifest.get("ambiguity_audit"), decoder)
    audit_note = footer_note(manifest.get("ambiguity_audit"))
    body += f'<footer><p class="note">{html.escape(audit_note)}</p><p class="note">All readouts for this policy share one collection and episode split. Frozen familiar-map decoding does not establish unknown-map transfer, map reconstruction or causal use of location information by the policy.</p></footer>'
    output = decorate_report(document(f"Geo diagnostics · {name}", body), run_dir, manifest)
    temporary = run_dir / ".report.tmp.html"
    temporary.write_text(output, encoding="utf-8")
    temporary.replace(run_dir / "report.html")
    update_run(run_dir, report={"dpi": dpi, "renderer_sha256": fingerprint(__file__)})
    build_dashboard()
    print(f"[report] {run_dir / 'report.html'}", flush=True)
