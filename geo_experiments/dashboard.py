"""Static report navigation and semantic table comparisons; no server or fetch required."""
from __future__ import annotations

import html
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
from urllib.parse import quote

from geo_experiments.common import ROOT, json_value


def dashboard_path():
    return ROOT / "geo_experiments" / "outputs" / "index.html"


def report_id(run_dir):
    return Path(run_dir).resolve().relative_to(ROOT / "geo_experiments").as_posix()


def numeric_cell(row, column, value, *, unit="number", decimals=0, suffix="", version="1", css="", title=""):
    """Only annotated numeric table cells participate; figures/text never do."""
    if value is None or not math.isfinite(float(value)):
        return '<td class="muted">N/A</td>'
    value = float(value)
    scientific = unit == "component"
    display = (f"{value:.3g}" if scientific else f"{value:,.{decimals}f}") + suffix
    attrs = {"data-compare-row": row, "data-compare-column": column, "data-value": repr(value),
             "data-unit": unit, "data-version": version, "data-decimals": str(decimals),
             "data-suffix": suffix, "class": css, "title": title}
    if scientific:
        attrs["data-number-format"] = "scientific"
    return '<td ' + ' '.join(f'{k}="{html.escape(str(v), quote=True)}"' for k, v in attrs.items()) + '>' + html.escape(display) + '</td>'


class TableMetrics(HTMLParser):
    """Derive section names from actual headings, independent of layout/order."""
    def __init__(self):
        super().__init__()
        self.sections, self.tables, self.metrics = [], [], []
        self.heading = None
        self.names = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "section":
            self.sections.append("")
        elif tag == "h2" and self.sections:
            self.heading = []
        elif tag == "table":
            self.tables.append(attrs.get("data-compare-table", self.tables[-1] if self.tables else ""))
        elif tag == "td" and self.sections and self.tables and "data-value" in attrs:
            required = ("data-compare-row", "data-compare-column", "data-unit", "data-version")
            if not all(k in attrs for k in required) or not self.sections[-1]:
                return
            value = float(attrs["data-value"])
            if not math.isfinite(value):
                return
            metric = {"section": self.sections[-1], "table": self.tables[-1],
                      "row": attrs["data-compare-row"], "column": attrs["data-compare-column"],
                      "unit": attrs["data-unit"], "version": attrs["data-version"], "value": value}
            key = tuple(metric[k] for k in ("section", "table", "row", "column", "unit", "version"))
            if key in self.names:
                raise ValueError(f"Duplicate comparison metric: {key}")
            self.names.add(key)
            self.metrics.append(metric)

    def handle_data(self, data):
        if self.heading is not None:
            self.heading.append(data)

    def handle_endtag(self, tag):
        if tag == "h2" and self.heading is not None:
            self.sections[-1] = ' '.join(''.join(self.heading).split())
            self.heading = None
        elif tag == "section" and self.sections:
            self.sections.pop()
        elif tag == "table" and self.tables:
            self.tables.pop()


def safe_json(value):
    return json.dumps(json_value(value), ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")


def report_controls(run_dir):
    url = quote(Path(os.path.relpath(dashboard_path(), run_dir)).as_posix(), safe="/")
    url += "#report=" + quote(report_id(run_dir), safe="")
    return (f'<div class="controls report-controls"><a id="report-browser-link" href="{html.escape(url)}" target="_top">All reports</a>'
            '<label>Compare to<select id="compare-report" disabled><option value="">No comparison</option></select></label>'
            '<span id="compare-note" class="note">Open All reports to choose a comparison.</span></div>')


def decorate_report(document, run_dir, manifest):
    parser = TableMetrics()
    parser.feed(document)
    metadata = {"format": "swarmecho-report-tables/v1", "id": report_id(run_dir),
                "name": manifest["config"]["run"]["name"], "folder": Path(run_dir).name,
                "metrics": parser.metrics}
    script = Path(__file__).with_name("dashboard.js").read_text(encoding="utf-8")
    addition = f'<script id="geo-report-data" type="application/json">{safe_json(metadata)}</script><script>{script}</script>'
    return document.replace("</body>", addition + "</body>")


class ReportMetadata(HTMLParser):
    def __init__(self):
        super().__init__()
        self.capture = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "script" and dict(attrs).get("id") == "geo-report-data":
            self.capture = True

    def handle_data(self, data):
        if self.capture:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag == "script":
            self.capture = False


def build_dashboard():
    root = ROOT / "geo_experiments"
    target = dashboard_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    reports = []
    for path in root.rglob("report.html"):
        if not path.resolve().is_relative_to(root.resolve()):
            continue
        parser = ReportMetadata()
        document = path.read_text(encoding="utf-8", errors="replace")
        parser.feed(document)
        try:
            data = json.loads(''.join(parser.parts)) if parser.parts else None
        except (ValueError, TypeError):
            data = None
        if not data or data.get("format") != "swarmecho-report-tables/v1":
            data = {"id": report_id(path.parent), "name": path.parent.name, "folder": path.parent.name, "metrics": []}
        data["document_id"] = data["id"]
        data["id"] = report_id(path.parent)
        data["folder"] = path.parent.name
        metrics = TableMetrics()
        try:
            metrics.feed(document)
            data["metrics"] = metrics.metrics
        except (ValueError, TypeError):
            # Ambiguous/malformed table metadata never breaks report navigation.
            data["metrics"] = []
        data["url"] = quote(Path(os.path.relpath(path, target.parent)).as_posix(), safe="/")
        reports.append(data)
    reports.sort(key=lambda r: (r["folder"].rsplit("_", 3)[-3:], r["id"]), reverse=True)
    script = Path(__file__).with_name("dashboard.js").read_text(encoding="utf-8")
    document = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><meta name="color-scheme" content="dark">
<title>Geo experiment reports</title><style>
*{box-sizing:border-box}body{margin:0;background:#0b1220;color:#e4edf7;font:14px/1.5 system-ui,sans-serif;display:flex;height:100vh}
aside{width:260px;flex-shrink:0;background:#111c2e;border-right:1px solid #2b3b53;padding:24px 16px;overflow:auto}
h1{font-size:20px;margin:0 0 8px}p{color:#9aacc3;font-size:12px;margin:0 0 24px}
nav{display:grid;gap:10px}button{font:inherit;color:inherit;background:#142138;border:1px solid #2b3b53;border-radius:8px;padding:12px;text-align:left;cursor:pointer;overflow-wrap:anywhere}
button:hover{border-color:#68b9f2}button[aria-current=true]{background:#1a304b;border-color:#68b9f2}button small{display:block;color:#9aacc3;font-size:11px;margin-top:6px}
iframe{border:0;flex:1;min-width:0;height:100vh;background:#0b1220}#empty{padding:30px}a{color:#68b9f2} .refresh{display:block;margin-top:24px;font-size:12px}
@media(max-width:850px){aside{width:190px;padding:16px 10px}h1{font-size:17px}}@media(max-width:550px){body{display:block}aside{width:100%;height:170px;border-right:0;border-bottom:1px solid #2b3b53}nav{display:flex;overflow:auto}button{min-width:190px}aside p,.refresh{display:none}iframe{width:100%;height:calc(100vh - 170px)}}
</style></head><body><aside><h1>Geo experiments</h1><p>Select a report. Use its comparison dropdown to show differences in matching table values.</p><nav id="reports"></nav><a class="refresh" href="" onclick="location.reload();return false">Refresh reports</a></aside>
<iframe id="report-frame" title="Experiment report" sandbox="allow-scripts allow-top-navigation-by-user-activation"></iframe><div id="empty" hidden>No reports yet.</div>'''
    document += f'<script id="geo-report-catalog" type="application/json">{safe_json(reports)}</script><script>{script}</script></body></html>'
    temporary = target.with_name(".index.tmp.html")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(target)
    return target
