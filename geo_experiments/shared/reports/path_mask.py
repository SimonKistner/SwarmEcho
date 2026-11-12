"""Compact path scores and nested offline Plotly examples from saved arrays."""
import hashlib
import html
import re
from pathlib import Path

import numpy as np

from geo_experiments.shared.artifacts import group, update_run
from geo_experiments.shared.common import fingerprint, uses_bank_maps
from geo_experiments.shared.console import terminal_print
from geo_experiments.shared.dashboard import build_dashboard, decorate_report, numeric_cell, report_controls, safe_json
from geo_experiments.shared.path_geometry import OFFSETS
from geo_experiments.shared.report_layout import document
from geo_experiments.shared.sampling import hierarchical_weights, mask_scores


LABELS = {"finder_gru": "Finder GRU", "finder_message": "Finder message",
          "all_knowing_after_info": "All knowing · once after information",
          "all_knowing_till_end": "All knowing · until end", "base": "Base stored message"}
THRESHOLDS = (50, 75, 80, 90, 95)
DEFAULT_THRESHOLD = 80


def predictions(arrays, config, threshold=DEFAULT_THRESHOLD / 100):
    result = {}
    for source in config["definition"]["inputs"]:
        rows = group(arrays, f"inputs/{source}")
        for seed in config["probe"]["seeds"]:
            saved = group(arrays, f"probes/path/{source}/seed_{seed}")
            if not saved:
                continue
            indices = saved["row_indices"]
            episodes, agents = rows["episode"][indices], rows["agent"][indices]
            probability = saved["probabilities"]
            if probability.shape != (len(indices), 125) or not np.isfinite(probability).all():
                raise ValueError(f"Invalid saved path probabilities: {source}/seed={seed}")
            exact, similarity = prediction_scores(probability, arrays["labels/path"], episodes, threshold)
            result[source, seed] = dict(episodes=episodes, agents=agents, indices=indices,
                probability=probability, exact=exact, similarity=similarity, threshold=threshold)
    return result


def prediction_scores(probability, labels, episodes, threshold):
    # Re-score saved predictions at display thresholds without materializing
    # every repeated row's full 125-cell target array at once.
    exact, similarity = np.empty(len(episodes)), np.empty(len(episodes))
    for start in range(0, len(episodes), 4096):
        stop = start + 4096
        exact[start:stop], similarity[start:stop] = mask_scores(probability[start:stop], labels[episodes[start:stop]], threshold)
    return exact, similarity


def aggregate(records, source, selected):
    values, histograms, episode_sets, agent_counts, row_counts = [], [], [], [], []
    for (name, _seed), saved in records.items():
        if name != source:
            continue
        keep = np.isin(saved["episodes"], selected)
        if not keep.any():
            continue
        episodes, agents = saved["episodes"][keep], saved["agents"][keep]
        weights = hierarchical_weights(episodes, agents)
        exact, similarity = saved["exact"][keep], saved["similarity"][keep]
        values.append([np.sum(exact * weights) * 100, np.sum(similarity * weights) * 100])
        histograms.append(np.histogram(similarity * 100, bins=np.arange(0, 101, 10), weights=weights * (1 - exact))[0])
        episode_sets.append(set(episodes))
        agent_counts.append(len(np.unique(np.column_stack((episodes, agents)), axis=0)))
        row_counts.append(len(episodes))
    if not values:
        return None
    hist = np.mean(histograms, axis=0)
    hist = hist / hist.sum() * 100 if hist.sum() else hist
    return dict(exact=float(np.mean(values, axis=0)[0]), similarity=float(np.mean(values, axis=0)[1]),
                std=np.std(values, axis=0), hist=hist, episodes=len(episode_sets[0]),
                agents=agent_counts[0], rows=row_counts[0])


def histogram(values):
    if not np.any(values):
        return '<span class="perfect">No incorrect paths</span>'
    bars, axes = [], []
    for value in (0, 50, 100):
        y = 110 - value * .8
        axes.append(f'<line x1="40" x2="330" y1="{y}" y2="{y}" stroke="#2b3b53"/><text x="34" y="{y+4}" text-anchor="end">{value}</text>')
    for value in range(0, 101, 20):
        axes.append(f'<text x="{40+value*2.9:g}" y="127" text-anchor="middle">{value}</text>')
    for i, value in enumerate(values):
        x, height = 41 + i * 29, value * .8
        bars.append(f'<rect x="{x}" y="{110-height:.2f}" width="26" height="{height:.2f}" fill="#68b9f2"><title>Similarity {i*10}–{(i+1)*10}%: {value:.1f}% of imperfect predictions</title></rect>')
    axes.append('<line x1="40" x2="40" y1="30" y2="110" stroke="#73818d"/>')
    return ('<svg viewBox="0 0 350 153" role="img" aria-label="Path similarity distribution among imperfect predictions" style="max-width:390px;width:100%">'
            '<g fill="#9aacc3" font-size="10"><text x="40" y="14">Share of imperfect predictions (%)</text>'
            + ''.join(axes) + '<text x="185" y="147" text-anchor="middle">Path similarity (%) → better</text></g>' + ''.join(bars) + '</svg>')


def score_cells(summary, source, scope, *, threshold=DEFAULT_THRESHOLD, population="common"):
    if summary is None:
        return '<td colspan="3" class="muted">No held-out predictions available</td>'
    key = f"Path / {LABELS[source]} / {scope}"
    parts = []
    for i, (metric, value) in enumerate((("Exact-path accuracy", summary["exact"]), ("Path similarity", summary["similarity"]))):
        parts.append(numeric_cell(key, metric, value, unit="percent", decimals=1, suffix="%", version=f"path-mask-overlap/v1/threshold-{threshold}/population-{population}",
            title=f"Seed variation: {summary['std'][i]:.2f} percentage points; {summary['episodes']} episodes; {summary['agents']} episode-agent pairs; {summary['rows']} snapshots"))
    return ''.join(parts) + '<td>' + histogram(summary["hist"]) + '</td>'


def example_payload(arrays, records, episode, building=None):
    m = int(arrays["episodes/map_id"][episode])
    bounds = arrays["paths/offsets"][episode:episode + 2]
    payload = dict(id=int(episode), map=str(arrays["geometry/map_names"][m]),
        cell=float(arrays["geometry/cell_size_m"][m]),
        lower=arrays[f"maps/{m}/solid_min"], upper=arrays[f"maps/{m}/solid_max"],
        base=arrays["episodes/base_position"][episode], target=arrays["episodes/target_position"][episode],
        path=arrays["paths/points"][bounds[0]:bounds[1]], rotation=int(arrays["episodes/rotation"][episode]),
        anchor=arrays["episodes/anchor"][episode], truth=arrays["labels/path"][episode].astype(int), building=building, sources={})
    for (source, seed), saved in records.items():
        slots = np.flatnonzero(saved["episodes"] == episode)
        rows = group(arrays, f"inputs/{source}")
        entries = []
        for slot in slots:
            index = saved["indices"][slot]
            entries.append(dict(agent=int(rows["agent"][index]), step=int(rows["step"][index]),
                first=int(rows["informed_step"][index]), p=saved["probability"][slot],
                float32=saved["probability"].dtype == np.float32))
        payload["sources"].setdefault(source, {})[str(seed)] = entries
    return payload


def loss_chart(history, source, seed, best_epoch=None):
    """Labeled, bounded SVG curves that remain usable in an offline report."""
    history = np.asarray(history)
    left, right, top, bottom = 67, 577, 30, 211
    end = max(2., float(history[-1, 0]))
    maximum = max(float(history[:, [1, 3]].max()), 1e-6)
    step = 10 ** np.floor(np.log10(maximum)) / 2
    ymax = np.ceil(maximum * 1.05 / step) * step
    x = lambda epoch: left + (epoch - 1) / (end - 1) * (right - left)
    y = lambda value: bottom - value / ymax * (bottom - top)
    elements = ['<svg viewBox="0 0 600 270" class="loss-chart" role="img" aria-label="Training and validation BCE by epoch">',
                '<g fill="#9aacc3" font-size="11">']
    for value in np.linspace(0, ymax, 5):
        elements.append(f'<line x1="{left}" x2="{right}" y1="{y(value):.2f}" y2="{y(value):.2f}" stroke="#2b3b53"/>'
                        f'<text x="{left-8}" y="{y(value)+4:.2f}" text-anchor="end">{value:.3g}</text>')
    for epoch in np.unique(np.rint(np.linspace(1, end, 5)).astype(int)):
        elements.append(f'<text x="{x(epoch):.2f}" y="231" text-anchor="middle">{epoch}</text>')
    elements.append('<text x="320" y="253" text-anchor="middle">Epoch</text>'
                    '<text x="18" y="125" text-anchor="middle" transform="rotate(-90 18 125)">BCE loss · lower is better</text></g>'
                    f'<path d="M{left},{top} V{bottom} H{right}" fill="none" stroke="#73818d"/>')
    for column, color, label, legend_x in ((1, '#68b9f2', 'Training', 345), (3, '#f3bb65', 'Validation', 453)):
        points = ' '.join(f'{x(row[0]):.2f},{y(row[column]):.2f}' for row in history)
        elements.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>'
                        f'<line x1="{legend_x}" x2="{legend_x+18}" y1="14" y2="14" stroke="{color}" stroke-width="2"/>'
                        f'<text x="{legend_x+23}" y="18" fill="#9aacc3" font-size="11">{label}</text>')
    if best_epoch is not None:
        elements.append(f'<line x1="{x(best_epoch):.2f}" x2="{x(best_epoch):.2f}" y1="{top}" y2="{bottom}" stroke="#6ed1b5" stroke-dasharray="4 4"><title>Selected epoch {int(best_epoch)}</title></line>')
    return (f'<figure class="loss-card"><figcaption>{html.escape(LABELS[source])} · seed {seed}</figcaption>'
            + ''.join(elements) + '</svg></figure>')


def generate_report(run_dir, manifest, arrays, dpi=None):
    from plotly.offline import get_plotlyjs

    run_dir, config = Path(run_dir), manifest["config"]
    test = arrays["splits/test"]
    records = predictions(arrays, config)
    groups = []
    for k in np.unique(arrays["episodes/required_agents"][test]):
        title = f"Targets requiring {int(k)} agents" if k else "Required agents not estimated"
        groups.append((f"required-{k}", title, test[arrays["episodes/required_agents"][test] == k]))
    cell_sizes = arrays["geometry/cell_size_m"][arrays["episodes/map_id"]]
    floors = np.floor(arrays["episodes/target_position"][:, 2] / cell_sizes).astype(int) + 1
    groups += [(f"storey-{f}", f"Target storey {f}", test[floors[test] == f]) for f in np.unique(floors[test])]
    maps = arrays["geometry/map_names"]
    if len(maps) < 20 or not uses_bank_maps(config):
        groups += [(f"map-{m}", str(name), test[arrays["episodes/map_id"][test] == m]) for m, name in enumerate(maps)]
    # Every input shares the episode split, but event availability can differ:
    # a GRU may exist even when no message ever reaches the base. Default to the
    # intersection for a fair comparison, retaining a full-population view.
    available = [set(saved["episodes"]) for saved in records.values()]
    if set(source for source, _ in records) != set(config["definition"]["inputs"]):
        available.append(set())
    common = np.array(sorted(set.intersection(*available)), dtype=int) if available else np.array([], dtype=int)
    populations = {"common": common, "available": test}
    default_population = "common" if len(common) else "available"
    scopes = [("overall", "Overall", test), *groups]
    cells = {}
    for threshold in THRESHOLDS:
        terminal_print(f"[report] path scores at {threshold}% cell threshold", flush=True)
        scored = {}
        for key, saved in records.items():
            exact, similarity = prediction_scores(saved["probability"], arrays["labels/path"], saved["episodes"], threshold / 100)
            scored[key] = {**saved, "exact": exact, "similarity": similarity}
        for population, selected in populations.items():
            for key, title, scope in scopes:
                subset = np.intersect1d(scope, selected)
                for source in config["definition"]["inputs"]:
                    content = score_cells(aggregate(scored, source, subset), source, title,
                                          threshold=threshold, population=population)
                    attrs = (f' data-evaluation-threshold="{threshold}" data-evaluation-population="{population}"'
                             + ('' if threshold == DEFAULT_THRESHOLD and population == default_population else ' hidden'))
                    cells[threshold, population, key, source] = re.sub(r'<td\b', lambda _: '<td' + attrs, content)

    def scores(source, scope):
        return ''.join(cells[t, p, scope, source] for t in THRESHOLDS for p in populations)

    from geo_experiments.shared.reports.building import building_for_report
    buildings = {}
    examples = {}
    example_ids = {}
    for key, title, selected in groups:
        if len(selected):
            # Selection depends only on recorded episode identity, never on fit quality.
            shared = np.intersect1d(selected, common)
            episode = min(map(int, shared if len(shared) else selected), key=lambda e: hashlib.sha256(
                f"{int(arrays['episodes/map_id'][e])}/{int(arrays['episodes/episode_id'][e])}".encode()).hexdigest())
            example_ids[key] = str(episode)
            if str(episode) not in examples:
                m = int(arrays["episodes/map_id"][episode])
                if m not in buildings:
                    buildings[m] = building_for_report(arrays, m, manifest)
                examples[str(episode)] = example_payload(arrays, records, episode, buildings[m])
    rejected = manifest.get("collection", {}).get("reference_route_rejections", 0)
    rejection_note = (f'<p class="note">{rejected:,} informed sampled episodes had no finite reward-reference route and were replaced. '
                      'Their coordinates and map IDs are recorded with the collection.</p>') if rejected else ''
    body = ('<header><div class="eyebrow">Geo experiments · frozen policy</div><h1>Base-to-target path predictor</h1>'
            '<p>Exact-path accuracy counts completely matching cell selections. Path similarity rewards shared path cells and penalizes missing or extra cells; empty space earns no points.</p>'
            f'<p class="note">{len(test):,} held-out episodes · {len(maps)} maps · {len(config["probe"]["seeds"])} decoder seeds. '
            'Scores balance episodes, then agents, then snapshots.</p>'
            + rejection_note + report_controls(run_dir) + '</header><section><h2>Held-out results</h2>'
            '<div class="controls"><label>Cell probability threshold <select id="held-out-threshold">'
            + ''.join(f'<option value="{t}"{" selected" if t == DEFAULT_THRESHOLD else ""}>{t}%</option>' for t in THRESHOLDS)
            + '</select></label><label>Episode population <select id="held-out-population">'
            + f'<option value="common"{" selected" if default_population == "common" else ""}>Same {len(common):,} episodes for every input</option>'
            + f'<option value="available"{" selected" if default_population == "available" else ""}>All available episodes per input</option></select></label></div>'
            '<p class="note">A cell is selected when its probability is at least the chosen threshold. Both scores and every histogram update together from saved predictions; no retraining or automatic threshold tuning.</p>'
            '<details class="metric-explanation"><summary>How to read these results</summary>'
            '<p>Path similarity counts shared path cells and penalizes both missed path cells and extra selected cells. Correctly empty cells are ignored. '
            'For example, a true path with 5 cells and a prediction with 4 correct cells plus 1 extra cell has 80% similarity, regardless of how much empty space surrounds it.</p>'
            '<p>The histogram shows similarity among imperfect predictions only, in 10-percentage-point bins. Left means little overlap; right means a near match. '
            'Perfect matches are excluded because the exact-path accuracy already shows their share. Bar heights are the balanced percentage of imperfect predictions.</p>'
            '<p>All inputs share the same held-out episode split, but some events are absent—for example, an agent can know the target without a message ever reaching the base. '
            'The default uses only episodes available for every input and seed. The alternative keeps each input’s full available population; counts appear beside each input. '
            'Each input still uses its own eligible agents and snapshots within those episodes.</p></details>'
            '<div class="table-scroll"><table class="path-results"><thead><tr><th>Input</th><th>Exact-path accuracy</th><th>Average path similarity</th><th>Similarity of imperfect predictions</th></tr></thead><tbody>')
    for source in config["definition"]["inputs"]:
        full = aggregate(records, source, test)
        count = full["episodes"] if full else 0
        body += (f'<tr><td><strong>{html.escape(LABELS[source])}</strong><br><span class="note">{count:,} / {len(test):,} test episodes available</span></td>'
                 + scores(source, "overall") + '</tr>')
        body += '<tr><td colspan="4"><details><summary>Details · required agents, storeys and maps</summary><div class="secondary">'
        for key, title, selected in groups:
            body += f'<details class="path-group"><summary>{html.escape(title)}</summary><table><tbody><tr><td>{html.escape(LABELS[source])}</td>'
            body += scores(source, key) + '</tr></tbody></table>'
            if key in example_ids:
                body += (f'<details class="path-example" data-example="{example_ids[key]}" data-source="{source}">'
                         '<summary>Representative prediction (3D)</summary><div class="example-content"></div></details>')
            body += '</details>'
        body += '</div></details></td></tr>'
    body += '</tbody></table></div></section>'
    body += '<section><details><summary>Training and validation loss</summary><p class="note">BCE is the optimization loss; lower is better. '
    body += 'Blue: training batches. Orange: the full balanced validation split. The dashed green line marks the selected epoch. These curves are independent of the display threshold.</p><div class="loss-grid">'
    for (source, seed), saved in records.items():
        history = arrays[f"probes/path/{source}/seed_{seed}/history"]
        best_epoch = arrays.get(f"probes/path/{source}/seed_{seed}/best_epoch")
        body += loss_chart(history, source, seed, best_epoch)
    body += '</div></details></section><footer><p class="note">Single canonical shortest reference route. Cell overlap does not certify an ordered navigable route. Required-agent groups use saved sampler estimates. Example sliders affect only their example; the held-out threshold selector controls every aggregate table and histogram.</p></footer>'
    css = '<style>.path-results{min-width:1000px;table-layout:auto}.path-results th:first-child{width:23%}.path-results th:nth-child(2),.path-results th:nth-child(3){width:16%}.path-results td{border-top:1px solid var(--line)}.path-group{padding:10px}.path-group>summary{font-size:14px}.path-group table{table-layout:auto}.path-example{padding:12px}.example-controls{display:flex;gap:10px;flex-wrap:wrap;padding:12px 0}.example-controls label{display:flex;align-items:center;gap:4px}.example-controls input{max-width:160px}.path-scene{height:520px;min-height:350px}.example-status{font-size:13px;color:var(--muted)}.loss-chart{width:100%;display:block}.loss-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin-top:16px}.loss-card{margin:0;padding:14px;background:var(--panel);border:1px solid var(--line);border-radius:8px;min-width:0}.loss-card figcaption{font-size:14px}.metric-explanation{margin:12px 0 20px}.metric-explanation p{font-size:13px;color:var(--muted);max-width:1000px}@media(max-width:800px){.loss-grid{grid-template-columns:1fr}}</style>'
    output = document("Path predictor", body).replace('</head>', css + '</head>')
    payload = dict(examples=examples, labels=LABELS, offsets=OFFSETS, threshold=DEFAULT_THRESHOLD / 100)
    script = Path(__file__).with_name('path_viewer.js').read_text(encoding='utf-8')
    output = output.replace('</body>', '<script>' + get_plotlyjs() + '</script><script id="path-examples" type="application/json">' + safe_json(payload) + '</script><script>' + script + '</script></body>')
    output = decorate_report(output, run_dir, manifest)
    temporary = run_dir / '.report.tmp.html'
    temporary.write_text(output, encoding='utf-8')
    temporary.replace(run_dir / 'report.html')
    update_run(run_dir, report=dict(renderer='path_mask', renderer_sha256=fingerprint(__file__),
        viewer_sha256=fingerprint(Path(__file__).with_name('path_viewer.js')),
        display_thresholds=[t / 100 for t in THRESHOLDS], default_threshold=DEFAULT_THRESHOLD / 100,
        default_population=default_population, common_test_episodes=len(common),
        histogram_metric="path_similarity_among_imperfect_predictions"))
    build_dashboard()
    return run_dir / 'report.html'
