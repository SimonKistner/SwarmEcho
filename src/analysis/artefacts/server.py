"""
server.py — SwarmEcho Artefacts dashboard (FastHTML + HTMX).

Launch:
    uv run python src/analysis/artefacts/server.py

Then open: http://localhost:8000
"""

from __future__ import annotations

import json
import sys
import urllib.parse
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_SRC = _HERE.parents[2]
_REPO = _HERE.parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
# Also make artefacts/ importable as a package sibling
_ARTEFACTS = _HERE.parent
if str(_ARTEFACTS) not in sys.path:
    sys.path.insert(0, str(_ARTEFACTS))

from fasthtml.common import (  # type: ignore[import]
    FastHTML,
    HTMLResponse,
    Link,
    Response,
    Script,
    StaticFiles,
    serve,
    NotStr,
)

from state import (
    CheckpointOption,
    checkpoint_options,
    cfg_get,
    discover_runs,
    eval_artifacts,
    group_runs,
    load_cfg,
    map_name_from_cfg,
    map_preview_path,
    parse_artifact_label,
    training_artifacts,
    video_status,
)
from jobs import (
    any_running,
    kill_job,
    read_queue,
    start_job,
    retry_job,
    clear_queue_history,
)
from map_renderer import find_map_yaml, load_map_data, render_and_save, validate_target
from config import load_config, css_vars

# ---------------------------------------------------------------------------
# Load layout and server configuration
# ---------------------------------------------------------------------------
_CFG_VALS = load_config()

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
STATIC_DIR = _HERE.parent / "static"

_TAB_JS = """
// Keep tab buttons in sync with HTMX navigation
function _artefactsSetTab(tab) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    var el = document.querySelector('.tab-btn[data-tab="' + tab + '"]');
    if (el) el.classList.add('active');
}
// Sync render-button enabled state whenever map-picker-section updates
document.addEventListener('htmx:afterSettle', function(evt) {
    var td = document.getElementById('target-valid');
    if (!td) return;
    var btn = document.getElementById('render-btn');
    if (btn) btn.disabled = td.dataset.valid !== 'true';
});
// Toggle advanced section
function _artefactsToggleAdv() {
    var body = document.getElementById('advanced-body');
    var btn  = document.getElementById('advanced-toggle');
    if (!body || !btn) return;
    var open = body.classList.toggle('open');
    btn.setAttribute('aria-expanded', open ? 'true' : 'false');
}
"""

app = FastHTML(
    hdrs=[
        Link(rel="stylesheet", href="/static/artefacts.css"),
        Script(src="https://unpkg.com/htmx.org@2.0.4/dist/htmx.min.js"),
        Script(_TAB_JS),
        NotStr(css_vars(_CFG_VALS)),
    ],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _sint(session: dict, key: str, default: int = 0) -> int:
    try:
        return int(session.get(key, default))
    except (TypeError, ValueError):
        return default


def _clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))


def _run(session: dict) -> Path | None:
    rd = session.get("run_dir")
    return Path(rd) if rd else None


def _ckpt(session: dict, run_dir: Path | None) -> CheckpointOption | None:
    if run_dir is None:
        return None
    cp = session.get("ckpt_path")
    if not cp:
        return None
    cp_path = Path(cp)
    cfg = load_cfg(run_dir)
    for opt in checkpoint_options(run_dir, cfg):
        if opt.path == cp_path:
            return opt
    return None


# ---------------------------------------------------------------------------
# Media helpers
# ---------------------------------------------------------------------------

def _murl(path: Path) -> str:
    return "/media?p=" + urllib.parse.quote(str(path), safe="")


def _video_el(path: Path) -> str:
    url = _murl(path)
    return (
        f'<video autoplay loop muted controls preload="metadata" style="width:100%;height:auto;max-height:inherit;display:block;object-fit:contain;">'
        f'<source src="{url}" type="video/mp4">'
        f'</video>'
    )


def _img_el(path: Path) -> str:
    return f'<img src="{_murl(path)}" loading="lazy" style="width:100%;height:auto;max-height:inherit;display:block;object-fit:contain;">'


def _empty(icon: str, msg: str, btn_html: str = "") -> str:
    return (
        f'<div class="media-empty">'
        f'<span class="ei">{icon}</span>'
        f'<span>{msg}</span>'
        f'{btn_html}'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Navigator bar
# ---------------------------------------------------------------------------

def _nav_bar(paths: list[Path], idx: int, prev_url: str, next_url: str) -> str:
    n = len(paths)
    if n == 0:
        return ""
    label = parse_artifact_label(paths[idx])
    fname = paths[idx].name
    prev_dis = "disabled" if idx <= 0 else ""
    next_dis = "disabled" if idx >= n - 1 else ""
    return (
        f'<div class="nav-bar">'
        f'<button class="nav-btn" hx-get="{prev_url}" hx-swap="outerHTML" '
        f'  hx-target="closest .media-box" {prev_dis}>◀</button>'
        f'<div class="nav-info">'
        f'  <div class="nav-counter">{idx + 1} / {n}</div>'
        f'  <div class="nav-label" title="{fname}">{label}</div>'
        f'</div>'
        f'<button class="nav-btn" hx-get="{next_url}" hx-swap="outerHTML" '
        f'  hx-target="closest .media-box" {next_dis}>▶</button>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Artifact selector cards
# ---------------------------------------------------------------------------

def _artifact_list(
    paths: list[Path],
    idx: int,
    base_url: str,
    kind: str,
    show_status: bool = False,      # only True for videos
) -> str:
    if not paths:
        return ""
    icons = {"video": "🎬", "found": "🔵", "chain": "🔴", "heatmap": "🖼"}
    icon = icons.get(kind, "📄")
    rows = []
    for i, p in enumerate(paths):
        active = "active" if i == idx else ""
        label = parse_artifact_label(p)
        sz = p.stat().st_size if p.exists() else 0
        sz_str = f"{sz // 1024} KB" if sz < 1_048_576 else f"{sz / 1_048_576:.1f} MB"
        pill = ""
        if show_status:
            st = video_status(p)
            if st == "SUCCESS":
                pill = '<span class="status-pill success">SUCCESS</span>'
            elif st == "FAIL":
                pill = '<span class="status-pill fail">FAIL</span>'
        rows.append(
            f'<button class="artifact-row {active}" '
            f'  hx-get="{base_url}?idx={i}" hx-swap="outerHTML" hx-target="closest .media-box">'
            f'  <span class="row-icon">{icon}</span>'
            f'  <span class="row-meta">'
            f'    <span class="row-title">{label}</span>'
            f'    <span class="row-sub">{p.name} · {sz_str}</span>'
            f'  </span>'
            f'  {pill}'
            f'</button>'
        )
    label_text = "Select artifact"
    return (
        f'<div class="artifact-list-label">{label_text}</div>'
        f'<div class="artifact-list">{"".join(rows)}</div>'
    )


# ---------------------------------------------------------------------------
# Media box builders
# ---------------------------------------------------------------------------

def _video_box(
    paths: list[Path],
    idx: int,
    prev_url: str,
    next_url: str,
    list_url: str,
    box_id: str = "",
    ckpt_path: str = "",
) -> str:
    id_attr = f'id="{box_id}"' if box_id else ""
    if not paths:
        create_btn = ""
        if ckpt_path:
            create_btn = (
                f'<button class="btn btn-primary" style="margin-top:8px" '
                f'  hx-post="/eval/queue-job" '
                f'  hx-vals=\'{{"job":"default_video","ckpt":"{ckpt_path}"}}\' '
                f'  hx-target="#queue-panel" hx-swap="outerHTML">▶ Create default video</button>'
            )
        body = _empty("🎬", "No videos found", create_btn)
        nav = lst = ""
    else:
        idx = _clamp(idx, 0, len(paths) - 1)
        body = f'<div class="video-wrap">{_video_el(paths[idx])}</div>'
        nav = _nav_bar(paths, idx, prev_url, next_url)
        lst = _artifact_list(paths, idx, list_url, "video", show_status=True)
    return f'<div class="media-box panel" {id_attr}>{nav}{body}{lst}</div>'


def _image_box(
    paths: list[Path],
    idx: int,
    prev_url: str,
    next_url: str,
    list_url: str,
    kind: str = "found",
    box_id: str = "",
    wrap_cls: str = "",
    ckpt_path: str = "",
    extra_middle: str = "",
) -> str:
    id_attr = f'id="{box_id}"' if box_id else ""
    if not paths:
        create_btn = ""
        if ckpt_path:
            create_btn = (
                f'<button class="btn btn-primary" style="margin-top:8px" '
                f'  hx-post="/eval/queue-job" '
                f'  hx-vals=\'{{"job":"default_heatmaps","ckpt":"{ckpt_path}"}}\' '
                f'  hx-target="#queue-panel" hx-swap="outerHTML">▶ Create heatmaps</button>'
            )
        body = _empty("🖼", "No heatmaps found", create_btn)
        nav = lst = ""
    else:
        idx = _clamp(idx, 0, len(paths) - 1)
        wc = wrap_cls if wrap_cls else "media-wrap"
        body = f'<div class="{wc}">{_img_el(paths[idx])}</div>'
        nav = _nav_bar(paths, idx, prev_url, next_url)
        # No status pills for heatmaps
        lst = _artifact_list(paths, idx, list_url, kind, show_status=False)
    return f'<div class="media-box panel" {id_attr}>{nav}{extra_middle}{body}{lst}</div>'


# ---------------------------------------------------------------------------
# Queue HTML
# ---------------------------------------------------------------------------

_STATUS_ICON = {"queued": "🕒", "running": "⏳", "finished": "✅", "crashed": "❌", "cancelled": "🚫"}


def _queue_html(eval_root: Path) -> str:
    queue = read_queue(eval_root)
    poll_int = _CFG_VALS.queue.poll_interval_seconds
    poll = (
        f' hx-trigger="every {poll_int}s" hx-get="/eval/queue/status" hx-swap="outerHTML"'
        if any_running(queue) else ""
    )
    if not queue:
        inner = '<div class="queue-empty">Queue is empty</div>'
    else:
        items = []
        for j in queue:
            status = j.get("status", "unknown")
            icon = _STATUS_ICON.get(status, "❓")
            name = j.get("name", "job")
            ts = j.get("started_at") or ""
            pid = j.get("pid", -1)
            job_id = j.get("id", "")
            log_path = j.get("log_path", "")

            action_btns = ""
            if status in ("running", "queued"):
                # Kill button
                kill_btn = (
                    f'<button class="btn-kill" title="Cancel job" '
                    f'  hx-post="/eval/kill-job" '
                    f'  hx-vals=\'{{"job_id":"{job_id}","pid":{pid}}}\' '
                    f'  hx-target="#queue-panel" hx-swap="outerHTML">✕</button>'
                )
                # Follow log button (only when there is a log path)
                follow_btn = ""
                if log_path:
                    import json as _json
                    lp_json = _json.dumps({"log_path": log_path})
                    follow_btn = (
                        f'<button class="btn-follow" title="Open log in terminal" '
                        f'  hx-post="/eval/open-log" '
                        f"  hx-vals='{lp_json}' "
                        f'  hx-swap="none">Log ↗</button>'
                    )
                action_btns = follow_btn + kill_btn
            elif status in ("crashed", "cancelled"):
                retry_btn = (
                    f'<button class="btn-retry" title="Retry job" '
                    f'  hx-post="/eval/retry-job" '
                    f'  hx-vals=\'{{"job_id":"{job_id}"}}\' '
                    f'  hx-target="#queue-panel" hx-swap="outerHTML">↺ Retry</button>'
                )
                # Allow viewing log of finished/crashed jobs too
                follow_btn = ""
                if log_path:
                    import json as _json
                    lp_json = _json.dumps({"log_path": log_path})
                    follow_btn = (
                        f'<button class="btn-follow" title="View log in terminal" '
                        f'  hx-post="/eval/open-log" '
                        f"  hx-vals='{lp_json}' "
                        f'  hx-swap="none">Log ↗</button>'
                    )
                action_btns = follow_btn + retry_btn
            elif status == "finished" and log_path:
                import json as _json
                lp_json = _json.dumps({"log_path": log_path})
                action_btns = (
                    f'<button class="btn-follow" title="View log in terminal" '
                    f'  hx-post="/eval/open-log" '
                    f"  hx-vals='{lp_json}' "
                    f'  hx-swap="none">Log ↗</button>'
                )

            items.append(
                f'<div class="q-item {status}">'
                f'  <span class="qi">{icon}</span>'
                f'  <span class="qn">{name}</span>'
                f'  <span class="qt">{ts}</span>'
                f'  <span class="q-actions">{action_btns}</span>'
                f'</div>'
            )
        inner = "\n".join(items)

    return (
        f'<div id="queue-panel" class="queue-wrap" {poll}>'
        f'  <div class="queue-title-row">'
        f'    <span class="queue-title">Job Queue</span>'
        f'    <div class="flex items-center gap-2">'
        f'      <button class="btn btn-sm" hx-post="/eval/clear-queue-history" '
        f'        hx-target="#queue-panel" hx-swap="outerHTML">🗑️ Clear History</button>'
        f'      <button class="btn btn-sm" hx-get="/eval/queue/status" '
        f'        hx-target="#queue-panel" hx-swap="outerHTML">↺ Refresh</button>'
        f'    </div>'
        f'  </div>'
        f'  <div class="queue-list">{inner}</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Map picker
# ---------------------------------------------------------------------------

def _ensure_map_preview(run_dir: Path, cfg: dict) -> Path | None:
    """
    Ensure the map preview PNG exists, rendering it inline if needed (fast, no JAX).
    Returns the path if successful, None if the map name is unknown.
    """
    map_name = map_name_from_cfg(cfg)
    if not map_name:
        return None
    preview = map_preview_path(run_dir, cfg)
    if preview.exists():
        return preview
    # Find the map YAML
    yaml_path = find_map_yaml(map_name, _REPO)
    if yaml_path is None:
        return None
    # Render inline — fast (<1s), no JAX
    ok = render_and_save(yaml_path, preview, scale=4.0)
    return preview if ok else None


def _validate_coords(cfg: dict, tx: float | None, ty: float | None) -> tuple[bool, str]:
    if tx is None or ty is None:
        return False, "click map to set target"
    map_name = map_name_from_cfg(cfg)
    map_data = load_map_data(map_name, _REPO) if map_name else None
    if map_data is not None:
        return validate_target(map_data, tx, ty)
    else:
        env_w = float(cfg_get(cfg, "env", "box_width", 0.0) or 0.0)
        env_h = float(cfg_get(cfg, "env", "box_height", 0.0) or 0.0)
        valid = (0.0 <= tx <= env_w) and (0.0 <= ty <= env_h)
        return valid, "in bounds" if valid else "out of bounds"


def _map_picker_html(run_dir: Path, cfg: dict, tx: float | None, ty: float | None) -> str:
    """
    Return the inner HTML of #map-picker-section.
    """
    map_name = map_name_from_cfg(cfg)
    env_w = float(cfg_get(cfg, "env", "box_width", 0.0) or 0.0)
    env_h = float(cfg_get(cfg, "env", "box_height", 0.0) or 0.0)

    preview = _ensure_map_preview(run_dir, cfg)

    if preview is None:
        if not map_name:
            return '<div class="map-pending">⚠ No map name found in run config.</div>'
        return (
            '<div class="map-pending">'
            f'<span>⚠ Map YAML not found for <code>{map_name}</code>. '
            'Check src/curriculum_config/maps/.</span>'
            '</div>'
        )

    if env_w <= 0 or env_h <= 0:
        return '<div class="map-pending">⚠ Map dimensions (box_width/box_height) missing from run config.</div>'

    valid, reason = _validate_coords(cfg, tx, ty)
    no_target = tx is None or ty is None
    img_url = _murl(preview)

    # ── SVG crosshair marker (only when a target is set) ─────────────────
    marker_svg = ""
    if not no_target:
        fx = float(tx) / env_w
        fy = 1.0 - float(ty) / env_h  # world Y=0 is bottom, image Y=0 is top
        cross_color = "#10b981" if valid else "#ef4444"  # green=valid, red=invalid
        marker_svg = (
            f'<svg class="map-picker-svg" viewBox="0 0 100 100" '
            f'  preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">'
            f'  <line x1="{fx*100:.3f}" y1="{max(0, fy*100-4):.3f}" '
            f'        x2="{fx*100:.3f}" y2="{min(100, fy*100+4):.3f}" '
            f'    stroke="{cross_color}" stroke-width="0.8"/>'
            f'  <line x1="{max(0, fx*100-4):.3f}" y1="{fy*100:.3f}" '
            f'        x2="{min(100, fx*100+4):.3f}" y2="{fy*100:.3f}" '
            f'    stroke="{cross_color}" stroke-width="0.8"/>'
            f'  <circle cx="{fx*100:.3f}" cy="{fy*100:.3f}" r="1.8" '
            f'    fill="none" stroke="{cross_color}" stroke-width="0.6"/>'
            f'</svg>'
        )

    # ── Inline JS: click → POST coords ───────────────────────────────────
    click_js = (
        f'function artefactsClick(e){{'
        f'  var r=e.currentTarget.querySelector("img").getBoundingClientRect();'
        f'  var fx=(e.clientX-r.left)/r.width;'
        f'  var fy=(e.clientY-r.top)/r.height;'
        f'  var x=(fx*{env_w:.4f}).toFixed(2);'
        f'  var y=((1.0-fy)*{env_h:.4f}).toFixed(2);'
        f'  var ov=document.getElementById("overrides-input");'
        f'  var overrides=ov?ov.value:"";'
        f'  htmx.ajax("POST","/eval/target",{{'
        f'    target:"#map-picker-section",swap:"innerHTML",'
        f'    values:{{x:x,y:y,overrides_input:overrides}}'
        f'  }});'
    )

    # Note that FastHTML brackets inside python raw f-strings need to be doubled correctly.
    # The end bracket of artefactsClick is wrapped in the string as:
    # f'  }});'
    # f'}}'
    click_js_formatted = click_js + f'}}'

    return (
        f'<script>{click_js_formatted}</script>'
        f'<div class="map-picker-wrap" onclick="artefactsClick(event)">'
        f'  <img class="map-picker-img" src="{img_url}" alt="Map preview">'
        f'  {marker_svg}'
        f'</div>'
    )


def _eval_coords_html(
    ckpt_str: str,
    tx: float | None,
    ty: float | None,
    valid: bool,
    reason: str,
    overrides_input: str = "",
) -> str:
    no_target = tx is None or ty is None
    val_cls = "valid" if valid else "invalid"
    val_txt = f"✓ {reason}" if valid else (f"✗ {reason}" if not no_target else "← click map")

    if not no_target:
        coord_inputs = (
            f'<div style="display:flex; gap:12px; width:100%; flex-wrap:wrap;">'
            f'  <div style="flex:1; min-width:80px; display:flex; align-items:center; gap:8px;">'
            f'    <span class="coord-label">x</span>'
            f'    <input class="coord-input" id="coord-x" name="x" type="number" step="0.5" value="{tx:.2f}" style="width:100%;"'
            f'      hx-trigger="change delay:400ms" hx-post="/eval/target"'
            f'      hx-include="#coord-x,#coord-y,#overrides-input"'
            f'      hx-target="#map-picker-section" hx-swap="innerHTML">'
            f'  </div>'
            f'  <div style="flex:1; min-width:80px; display:flex; align-items:center; gap:8px;">'
            f'    <span class="coord-label">y</span>'
            f'    <input class="coord-input" id="coord-y" name="y" type="number" step="0.5" value="{ty:.2f}" style="width:100%;"'
            f'      hx-trigger="change delay:400ms" hx-post="/eval/target"'
            f'      hx-include="#coord-x,#coord-y,#overrides-input"'
            f'      hx-target="#map-picker-section" hx-swap="innerHTML">'
            f'  </div>'
            f'</div>'
            f'<div style="margin-top:8px;">'
            f'  <span class="validity-pill {val_cls}" style="display:block; text-align:center; width:100%;">{val_txt}</span>'
            f'</div>'
        )
    else:
        coord_inputs = (
            f'<div style="margin-top:8px;">'
            f'  <span class="validity-pill invalid" style="display:block; text-align:center; width:100%;">{val_txt}</span>'
            f'</div>'
        )

    disabled_attr = "" if valid else "disabled"
    render_btn = (
        f'<button class="btn btn-primary" id="render-btn" {disabled_attr} style="width:100%; justify-content:center; padding:8px 16px; font-weight:600;"'
        f'  hx-post="/eval/queue-render"'
        f'  hx-include="#coord-x,#coord-y,#overrides-input"'
        f'  hx-vals=\'{{"ckpt":"{ckpt_str}"}}\''
        f'  hx-target="#queue-panel" hx-swap="outerHTML">'
        f'  ▶ Targeted render'
        f'</button>'
    )

    return (
        f'<div class="map-section-title">Coordinates & Controls</div>'
        f'<div style="display:flex; flex-direction:column; gap:16px; height:calc(100% - 24px); justify-content:space-between;">'
        f'  <div style="display:flex; flex-direction:column; gap:16px;">'
        f'    <div>'
        f'      <span class="coord-label" style="font-size:9px; text-transform:uppercase; color:var(--text-muted); display:block; margin-bottom:6px; letter-spacing:0.05em;">Target Coordinates</span>'
        f'      {coord_inputs}'
        f'    </div>'
        f'    <div>'
        f'      <label class="coord-label" for="overrides-input" style="font-size:9px; text-transform:uppercase; color:var(--text-muted); display:block; margin-bottom:6px; letter-spacing:0.05em;">Extra Overrides</label>'
        f'      <input class="overrides-input" id="overrides-input" name="overrides_input" type="text" '
        f'        style="margin-top:0; width:100%; padding:8px 12px;" placeholder="e.g. env.max_steps=5000" value="{overrides_input}">'
        f'    </div>'
        f'  </div>'
        f'  <div style="margin-top:auto;">'
        f'    {render_btn}'
        f'  </div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Page shell
# ---------------------------------------------------------------------------

def _page_shell(sidebar_html: str, content_html: str, active_tab: str = "train") -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>SwarmEcho Artefacts</title>
  <meta name="description" content="Training and evaluation artifact browser for SwarmEcho">
  <link rel="stylesheet" href="/static/artefacts.css">
  {css_vars(_CFG_VALS)}
</head>
<body>
<div class="shell">
  <aside class="sidebar">
    <div class="sidebar-header">
      <div class="sidebar-logo"><span class="icon">🔬</span> SwarmEcho Artefacts</div>
    </div>
    <div class="sidebar-body">{sidebar_html}</div>
  </aside>

  <div class="main">
    <nav class="tab-nav" id="tab-nav">
      <button class="tab-btn {'active' if active_tab=='train' else ''}" data-tab="train"
        hx-get="/train" hx-target="#tab-content"
        onclick="_artefactsSetTab('train')">
        📈 Training
      </button>
      <button class="tab-btn {'active' if active_tab=='eval' else ''}" data-tab="eval"
        hx-get="/eval" hx-target="#tab-content"
        onclick="_artefactsSetTab('eval')">
        🎯 Eval
      </button>
      <span style="flex:1"></span>
      <span class="htmx-indicator">Loading…</span>
    </nav>

    <div class="tab-content" id="tab-content">
      {content_html}
    </div>
  </div>
</div>
<script src="https://unpkg.com/htmx.org@2.0.4/dist/htmx.min.js"></script>
<script>{_TAB_JS}</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _sidebar_html(session: dict) -> str:
    runs = discover_runs()
    if not runs:
        return '<div class="section-label" style="padding:16px">No runs found in outputs/</div>'
    groups = group_runs(runs)
    current = session.get("run_dir", "")
    parts = []
    for group, group_runs_list in groups.items():
        parts.append(f'<div class="run-group"><div class="run-group-label">{group}</div>')
        for run in group_runs_list:
            active = "active" if str(run) == current else ""
            parts.append(
                f'<button class="run-item {active}" '
                f'  hx-post="/select-run" hx-vals=\'{{"run_dir":"{str(run)}"}}\' '
                f'  hx-target="#tab-content">'
                f'  <span class="run-dot"></span>'
                f'  <span class="run-name" title="{run.name}">{run.name}</span>'
                f'</button>'
            )
        parts.append('</div>')
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Checkpoint bar
# ---------------------------------------------------------------------------

def _ckpt_bar_html(run_dir: Path | None, cfg: dict, session: dict) -> str:
    if run_dir is None:
        return ""
    ckpts = checkpoint_options(run_dir, cfg)
    if not ckpts:
        return (
            '<div class="ckpt-bar">'
            '<span style="color:var(--text-dim);font-size:12px">No checkpoints found for this run.</span>'
            '</div>'
        )
    current = session.get("ckpt_path", "")
    if not current and ckpts:
        current = str(ckpts[0].path)
        session["ckpt_path"] = current

    opts_html = ""
    selected = None
    for opt in ckpts:
        sel = "selected" if str(opt.path) == current else ""
        if sel:
            selected = opt
        opts_html += f'<option value="{opt.path}" {sel}>{opt.label}</option>'

    badge = ""
    if selected:
        if selected.has_artifacts:
            badge = '<span class="ckpt-badge has">● artifacts</span>'
        else:
            badge = '<span class="ckpt-badge none">○ no artifacts</span>'

    return (
        f'<div class="ckpt-bar">'
        f'  <label>Checkpoint</label>'
        f'  <select class="ckpt-select" id="ckpt-select" name="ckpt_path"'
        f'    hx-post="/select-ckpt" hx-target="#tab-content" hx-include="#ckpt-select">'
        f'    {opts_html}'
        f'  </select>'
        f'  {badge}'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Training tab
# ---------------------------------------------------------------------------

def _training_content(run_dir: Path, cfg: dict, session: dict) -> str:
    arts = training_artifacts(run_dir)
    vids   = arts["videos"]
    founds = arts["found_heatmaps"]
    chains = arts["chain_heatmaps"]

    vi = _clamp(_sint(session, "train_vi"), 0, max(0, len(vids) - 1))
    fi = _clamp(_sint(session, "train_fi"), 0, max(0, len(founds) - 1))
    ci = _clamp(_sint(session, "train_ci"), 0, max(0, len(chains) - 1))

    video_box = _video_box(
        vids, vi,
        f"/train/video?idx={vi-1}", f"/train/video?idx={vi+1}",
        "/train/video", box_id="train-video-box",
    )
    found_box = _image_box(
        founds, fi,
        f"/train/found?idx={fi-1}", f"/train/found?idx={fi+1}",
        "/train/found", kind="found", box_id="train-found-box", wrap_cls="heat-wrap",
    )
    chain_box = _image_box(
        chains, ci,
        f"/train/chain?idx={ci-1}", f"/train/chain?idx={ci+1}",
        "/train/chain", kind="chain", box_id="train-chain-box", wrap_cls="heat-wrap",
    )

    return (
        f'<div class="training-layout">'
        f'  <div class="video-column">'
        f'    <div class="panel-header">Training Videos</div>'
        f'    {video_box}'
        f'  </div>'
        f'  <div class="heatmap-column">'
        f'    <div class="heatmap-item">'
        f'      <div class="panel-header">Found / Delivered</div>'
        f'      {found_box}'
        f'    </div>'
        f'    <div class="heatmap-item">'
        f'      <div class="panel-header">Chain Failure</div>'
        f'      {chain_box}'
        f'    </div>'
        f'  </div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Eval tab
# ---------------------------------------------------------------------------

def _eval_content(run_dir: Path, cfg: dict, session: dict) -> str:
    ckpt_opt = _ckpt(session, run_dir)
    if ckpt_opt is None:
        return (
            '<div class="empty-state">'
            '<span class="ei">🎯</span>'
            '<span class="et">Select a checkpoint above to view eval artifacts</span>'
            '</div>'
        )

    eval_root = ckpt_opt.artifact_root
    legacy = run_dir / "videos" / "eval"
    arts = eval_artifacts(eval_root, legacy_eval=legacy if legacy.exists() else None)
    vids   = arts["videos"]
    founds = arts["found_heatmaps"]
    chains = arts["chain_heatmaps"]

    vi = _clamp(_sint(session, "eval_vi"), 0, max(0, len(vids) - 1))
    kind = session.get("eval_hkind", "found")
    heat_paths = founds if kind == "found" else chains
    hi = _clamp(_sint(session, "eval_hi"), 0, max(0, len(heat_paths) - 1))

    ckpt_str = str(ckpt_opt.path)

    video_box = _video_box(
        vids, vi,
        f"/eval/video?idx={vi-1}", f"/eval/video?idx={vi+1}",
        "/eval/video", box_id="eval-video-box", ckpt_path=ckpt_str,
    )

    heatmap_section = _eval_heatmap_box(heat_paths, hi, kind, ckpt_str=ckpt_str)

    artifact_grid = (
        f'<div class="eval-artifact-grid" id="eval-artifact-grid" '
        f'  hx-trigger="eval-artifacts-updated from:body" '
        f'  hx-get="/eval/artifacts/reload" '
        f'  hx-swap="outerHTML">'
        f'  <div><div class="panel-header">Eval Video</div>{video_box}</div>'
        f'  <div><div class="panel-header">Heatmap</div>{heatmap_section}</div>'
        f'</div>'
    )

    # Target picker — no default coords; button disabled until user clicks map
    raw_tx = session.get("target_x")
    raw_ty = session.get("target_y")
    tx: float | None = float(raw_tx) if raw_tx is not None else None
    ty: float | None = float(raw_ty) if raw_ty is not None else None
    overrides_input = session.get("overrides_input", "")

    valid, reason = _validate_coords(cfg, tx, ty)

    eval_section_2 = (
        f'<div class="eval-section-2">'
        f'  <div class="eval-queue-col">{_queue_html(eval_root)}</div>'
        f'  <div class="eval-target-group">'
        f'    <div class="eval-map-col">'
        f'      <div class="map-section-title">Target Selection Map</div>'
        f'      <div id="map-picker-section">{_map_picker_html(run_dir, cfg, tx, ty)}</div>'
        f'    </div>'
        f'    <div class="eval-ctrl-col" id="eval-coords-column">'
        f'      {_eval_coords_html(ckpt_str, tx, ty, valid, reason, overrides_input)}'
        f'    </div>'
        f'  </div>'
        f'</div>'
    )

    return (
        artifact_grid +
        eval_section_2 +
        _advanced_html(ckpt_opt)
    )


# ---------------------------------------------------------------------------
# Advanced section
# ---------------------------------------------------------------------------

def _advanced_html(ckpt_opt: CheckpointOption) -> str:
    ckpt_str = str(ckpt_opt.path)
    return (
        f'<div class="advanced-wrap">'
        f'  <button class="advanced-btn" id="advanced-toggle" aria-expanded="false" '
        f'    onclick="_artefactsToggleAdv()">'
        f'    Advanced Tools <span class="chevron">⌄</span>'
        f'  </button>'
        f'  <div class="advanced-body" id="advanced-body">'
        f'    <div class="advanced-grid">'
        
        # Heatmap pipeline
        f'      <div class="adv-card">'
        f'        <div class="adv-card-title">Heatmap Pipeline</div>'
        f'        <div style="font-size:9px; color:var(--text-muted); margin-bottom:6px; line-height:1.2;">Options: --heatmap, --cluster, --csv, --no-csv</div>'
        f'        <input class="overrides-input" id="pipe-overrides" name="pipe_overrides" type="text" '
        f'          value="--heatmap --cluster" style="margin-top:0" placeholder="e.g. --heatmap --cluster">'
        f'        <button class="btn btn-sm w-full" style="margin-top:8px"'
        f'          hx-post="/eval/queue-job"'
        f'          hx-include="#pipe-overrides"'
        f'          hx-vals=\'{{"job":"pipeline","ckpt":"{ckpt_str}"}}\''
        f'          hx-target="#queue-panel" hx-swap="outerHTML">Queue pipeline</button>'
        f'      </div>'
        f'    </div>'  # advanced-grid
        f'  </div>'   # advanced-body
        f'</div>'     # advanced-wrap
    )


# ===========================================================================
# Routes
# ===========================================================================

@app.get("/")
def index(session):
    run_dir = _run(session)
    if run_dir is None:
        runs = discover_runs()
        if runs:
            run_dir = runs[0]
            session["run_dir"] = str(run_dir)

    cfg = load_cfg(run_dir) if run_dir else {}

    if run_dir and not session.get("ckpt_path"):
        ckpts = checkpoint_options(run_dir, cfg)
        if ckpts:
            session["ckpt_path"] = str(ckpts[0].path)

    sidebar = _sidebar_html(session)
    ckpt_bar = _ckpt_bar_html(run_dir, cfg, session)

    if run_dir:
        content = ckpt_bar + _training_content(run_dir, cfg, session)
    else:
        content = (
            '<div class="empty-state">'
            '<span class="ei">📁</span>'
            '<span class="et">No runs found</span>'
            '<span class="es">Check that outputs/ contains training runs</span>'
            '</div>'
        )

    return HTMLResponse(_page_shell(sidebar, content, active_tab="train"))


@app.post("/select-run")
def select_run(session, run_dir: str):
    session["run_dir"] = run_dir
    session.pop("ckpt_path", None)
    session.pop("target_x", None)
    session.pop("target_y", None)
    rd = Path(run_dir)
    cfg = load_cfg(rd)
    ckpts = checkpoint_options(rd, cfg)
    if ckpts:
        session["ckpt_path"] = str(ckpts[0].path)
    ckpt_bar = _ckpt_bar_html(rd, cfg, session)
    return HTMLResponse(ckpt_bar + _training_content(rd, cfg, session))


@app.post("/select-ckpt")
def select_ckpt(session, ckpt_path: str):
    session["ckpt_path"] = ckpt_path
    run_dir = _run(session)
    if run_dir is None:
        return HTMLResponse("")
    cfg = load_cfg(run_dir)
    ckpt_bar = _ckpt_bar_html(run_dir, cfg, session)
    return HTMLResponse(ckpt_bar + _eval_content(run_dir, cfg, session))


# ── Tabs ──────────────────────────────────────────────────────────────────

@app.get("/train")
def tab_train(session):
    run_dir = _run(session)
    if run_dir is None:
        return HTMLResponse('<div class="empty-state"><span class="et">Select a run</span></div>')
    cfg = load_cfg(run_dir)
    return HTMLResponse(_ckpt_bar_html(run_dir, cfg, session) + _training_content(run_dir, cfg, session))


@app.get("/eval")
def tab_eval(session):
    run_dir = _run(session)
    if run_dir is None:
        return HTMLResponse('<div class="empty-state"><span class="et">Select a run</span></div>')
    cfg = load_cfg(run_dir)
    # Ensure active tab state updates properly
    return HTMLResponse(_ckpt_bar_html(run_dir, cfg, session) + _eval_content(run_dir, cfg, session))


# ── Training media swaps ──────────────────────────────────────────────────

@app.get("/train/video")
def train_video(session, idx: int = 0):
    rd = _run(session)
    if rd is None:
        return HTMLResponse("")
    paths = training_artifacts(rd)["videos"]
    idx = _clamp(idx, 0, max(0, len(paths) - 1))
    session["train_vi"] = idx
    return HTMLResponse(_video_box(
        paths, idx,
        f"/train/video?idx={idx-1}", f"/train/video?idx={idx+1}",
        "/train/video", box_id="train-video-box",
    ))


@app.get("/train/found")
def train_found(session, idx: int = 0):
    rd = _run(session)
    if rd is None:
        return HTMLResponse("")
    paths = training_artifacts(rd)["found_heatmaps"]
    idx = _clamp(idx, 0, max(0, len(paths) - 1))
    session["train_fi"] = idx
    return HTMLResponse(_image_box(
        paths, idx,
        f"/train/found?idx={idx-1}", f"/train/found?idx={idx+1}",
        "/train/found", kind="found", box_id="train-found-box", wrap_cls="heat-wrap",
    ))


@app.get("/train/chain")
def train_chain(session, idx: int = 0):
    rd = _run(session)
    if rd is None:
        return HTMLResponse("")
    paths = training_artifacts(rd)["chain_heatmaps"]
    idx = _clamp(idx, 0, max(0, len(paths) - 1))
    session["train_ci"] = idx
    return HTMLResponse(_image_box(
        paths, idx,
        f"/train/chain?idx={idx-1}", f"/train/chain?idx={idx+1}",
        "/train/chain", kind="chain", box_id="train-chain-box", wrap_cls="heat-wrap",
    ))


# ── Eval media swaps ──────────────────────────────────────────────────────

@app.get("/eval/video")
def eval_video(session, idx: int = 0):
    rd = _run(session)
    opt = _ckpt(session, rd)
    if rd is None or opt is None:
        return HTMLResponse("")
    paths = eval_artifacts(opt.artifact_root)["videos"]
    idx = _clamp(idx, 0, max(0, len(paths) - 1))
    session["eval_vi"] = idx
    return HTMLResponse(_video_box(
        paths, idx,
        f"/eval/video?idx={idx-1}", f"/eval/video?idx={idx+1}",
        "/eval/video", box_id="eval-video-box", ckpt_path=str(opt.path),
    ))


@app.get("/eval/heatmap")
def eval_heatmap(session, idx: int = 0):
    rd = _run(session)
    opt = _ckpt(session, rd)
    if rd is None or opt is None:
        return HTMLResponse("")
    arts = eval_artifacts(opt.artifact_root)
    kind = session.get("eval_hkind", "found")
    paths = arts["found_heatmaps"] if kind == "found" else arts["chain_heatmaps"]
    idx = _clamp(idx, 0, max(0, len(paths) - 1))
    session["eval_hi"] = idx
    return HTMLResponse(_eval_heatmap_box(paths, idx, kind, ckpt_str=str(opt.path)))


@app.post("/eval/hkind")
def eval_hkind(session, kind: str = "found"):
    session["eval_hkind"] = kind
    session["eval_hi"] = 0
    rd = _run(session)
    opt = _ckpt(session, rd)
    if rd is None or opt is None:
        return HTMLResponse("")
    arts = eval_artifacts(opt.artifact_root)
    paths = arts["found_heatmaps"] if kind == "found" else arts["chain_heatmaps"]
    return HTMLResponse(_eval_heatmap_box(paths, 0, kind, ckpt_str=str(opt.path)))


def _eval_heatmap_box(paths: list[Path], idx: int, kind: str, ckpt_str: str = "") -> str:
    fa = "active" if kind == "found" else ""
    ca = "active" if kind == "chain" else ""
    toggle = (
        f'<div class="kind-toggle">'
        f'  <button class="kind-btn {fa}" hx-post="/eval/hkind" hx-vals=\'{{"kind":"found"}}\' '
        f'    hx-target="#eval-heatmap-box" hx-swap="outerHTML">Found / Delivered</button>'
        f'  <button class="kind-btn {ca}" hx-post="/eval/hkind" hx-vals=\'{{"kind":"chain"}}\' '
        f'    hx-target="#eval-heatmap-box" hx-swap="outerHTML">Chain</button>'
        f'</div>'
    )
    return _image_box(
        paths, idx,
        f"/eval/heatmap?idx={idx-1}", f"/eval/heatmap?idx={idx+1}",
        "/eval/heatmap", kind=kind, box_id="eval-heatmap-box", ckpt_path=ckpt_str, extra_middle=toggle,
    )


# ── Map target ────────────────────────────────────────────────────────────

@app.post("/eval/target")
def eval_target(
    session,
    x: float = 0.0,
    y: float = 0.0,
    overrides_input: str = "",
):
    session["target_x"] = x
    session["target_y"] = y
    session["overrides_input"] = overrides_input

    rd = _run(session)
    if rd is None:
        return HTMLResponse("")

    cfg = load_cfg(rd)
    opt = _ckpt(session, rd)
    ckpt_str = str(opt.path) if opt else ""

    valid, reason = _validate_coords(cfg, x, y)

    map_html = _map_picker_html(rd, cfg, x, y)
    coords_html = _eval_coords_html(ckpt_str, x, y, valid, reason, overrides_input)

    response_html = f'{map_html}\n<div id="eval-coords-column" hx-swap-oob="true">{coords_html}</div>'
    return HTMLResponse(response_html)


# ── Queue ────────────────────────────────--------------------------------

def _queue_response(eval_root: Path) -> HTMLResponse:
    queue = read_queue(eval_root)
    html = _queue_html(eval_root)
    headers = {}
    if not any_running(queue):
        headers["HX-Trigger"] = "eval-artifacts-updated"
    return HTMLResponse(html, headers=headers)


@app.get("/eval/queue/status")
def eval_queue_status(session):
    rd = _run(session)
    opt = _ckpt(session, rd)
    if rd is None or opt is None:
        return HTMLResponse('<div id="queue-panel"></div>')
    return _queue_response(opt.artifact_root)


@app.post("/eval/kill-job")
def eval_kill_job(session, job_id: str, pid: int):
    rd = _run(session)
    opt = _ckpt(session, rd)
    if rd is None or opt is None:
        return HTMLResponse("")

    eval_root = opt.artifact_root
    job_file = eval_root / "jobs" / f"{job_id}.json"

    kill_job(pid)

    if job_file.exists():
        try:
            data = json.loads(job_file.read_text())
            data["status"] = "cancelled"
            import time as _time
            data["finished_at"] = _time.strftime("%Y-%m-%d %H:%M:%S")
            job_file.write_text(json.dumps(data, indent=2, sort_keys=True))
        except Exception:
            pass

    return _queue_response(eval_root)


@app.post("/eval/retry-job")
def eval_retry_job(session, job_id: str):
    rd = _run(session)
    opt = _ckpt(session, rd)
    if rd is None or opt is None:
        return HTMLResponse("")
    eval_root = opt.artifact_root
    retry_job(eval_root, job_id)
    return _queue_response(eval_root)


@app.post("/eval/clear-queue-history")
def eval_clear_queue_history(session):
    rd = _run(session)
    opt = _ckpt(session, rd)
    if rd is None or opt is None:
        return HTMLResponse("")
    eval_root = opt.artifact_root
    clear_queue_history(eval_root)
    return _queue_response(eval_root)


@app.post("/eval/open-log")
def eval_open_log(log_path: str):
    """
    Open a WSL terminal tailing the job log file.
    Uses wt.exe (Windows Terminal) → wsl.exe → tail -f.
    NOTE: wt.exe uses ';' as its own pane/tab separator, so we must
    never include a bare semicolon in the arguments passed to it.
    """
    import subprocess as _sp

    p = Path(log_path)

    # Convert Windows path to WSL path if needed.
    # Server runs in WSL so paths are usually already Linux-style.
    log_str = str(p)
    if len(log_str) >= 2 and log_str[1] == ":":
        drive = log_str[0].lower()
        rest = log_str[2:].replace("\\", "/")
        wsl_path = f"/mnt/{drive}{rest}"
    else:
        wsl_path = log_str.replace("\\", "/")

    # Ensure the log file exists so tail -f doesn't immediately error
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.touch()

    job_name = p.stem
    title = f"Log: {job_name}"

    # IMPORTANT: do NOT put a ';' anywhere outside of a bash-quoted string —
    # wt.exe splits on ';' to create new tabs/panes, causing a second terminal.
    # We use a simple 'tail -n 100 -f path' with no semicolons at all.
    bash_cmd = f"tail -n 100 -f '{wsl_path}'"

    try:
        # Windows Terminal: open a new tab running WSL tail -f
        _sp.Popen(
            ["wt.exe", "--title", title, "wsl.exe", "--", "bash", "-c", bash_cmd],
            close_fds=True,
        )
    except FileNotFoundError:
        try:
            # Fallback: cmd.exe start → wsl tail -f
            _sp.Popen(
                ["cmd.exe", "/c", "start", title, "wsl.exe", "--", "bash", "-c", bash_cmd],
                close_fds=True,
                shell=False,
            )
        except Exception:
            pass

    # 204 No Content — HTMX does nothing (hx-swap="none")
    return Response("", status_code=204)


@app.get("/eval/artifacts/reload")
def eval_artifacts_reload(session):
    rd = _run(session)
    cfg = load_cfg(rd) if rd else {}
    ckpt_opt = _ckpt(session, rd)
    if rd is None or ckpt_opt is None:
        return HTMLResponse("")

    eval_root = ckpt_opt.artifact_root
    legacy = rd / "videos" / "eval"
    arts = eval_artifacts(eval_root, legacy_eval=legacy if legacy.exists() else None)
    vids = arts["videos"]
    founds = arts["found_heatmaps"]
    chains = arts["chain_heatmaps"]

    # Auto-load the newest video creation
    if vids:
        newest_vid = max(vids, key=lambda p: p.stat().st_mtime)
        vi = vids.index(newest_vid)
        session["eval_vi"] = vi
    else:
        vi = 0

    # Auto-load the newest heatmap creation
    kind = session.get("eval_hkind", "found")
    heat_paths = founds if kind == "found" else chains
    if heat_paths:
        newest_heat = max(heat_paths, key=lambda p: p.stat().st_mtime)
        hi = heat_paths.index(newest_heat)
        session["eval_hi"] = hi
    else:
        hi = 0

    ckpt_str = str(ckpt_opt.path)

    video_box = _video_box(
        vids, vi,
        f"/eval/video?idx={vi-1}", f"/eval/video?idx={vi+1}",
        "/eval/video", box_id="eval-video-box", ckpt_path=ckpt_str,
    )

    heatmap_section = _eval_heatmap_box(heat_paths, hi, kind, ckpt_str=ckpt_str)

    return HTMLResponse(
        f'<div class="eval-artifact-grid" id="eval-artifact-grid" '
        f'  hx-trigger="eval-artifacts-updated from:body" '
        f'  hx-get="/eval/artifacts/reload" '
        f'  hx-swap="outerHTML">'
        f'  <div><div class="panel-header">Eval Video</div>{video_box}</div>'
        f'  <div><div class="panel-header">Heatmap</div>{heatmap_section}</div>'
        f'</div>'
    )


# ── Job launchers ─────────────────────────────────────────────────────────

@app.post("/eval/queue-render")
def eval_queue_render(
    session,
    ckpt: str,
    x: Optional[str] = None,
    y: Optional[str] = None,
    coord_x: Optional[str] = None,
    coord_y: Optional[str] = None,
    overrides_input: str = "",
):
    rd = _run(session)
    opt = _ckpt(session, rd)
    if rd is None or opt is None:
        return HTMLResponse("")

    tx = float(session.get("target_x", 0.0))
    ty = float(session.get("target_y", 0.0))
    # Prefer freshly posted coords if present
    val_x = x or coord_x
    val_y = y or coord_y
    if val_x:
        try:
            tx = float(val_x)
        except ValueError:
            pass
    if val_y:
        try:
            ty = float(val_y)
        except ValueError:
            pass

    command = [
        "uv", "run", "python", "src/training/evaluate.py",
        f"checkpoint={ckpt}",
        f"target_pos={tx:.3f},{ty:.3f}",
    ]
    if overrides_input.strip():
        command.extend(overrides_input.strip().split())

    name = f"target_x{tx:.2f}_y{ty:.2f}"
    start_job(opt.artifact_root, name, command, _REPO)
    return _queue_response(opt.artifact_root)


@app.post("/eval/queue-job")
def eval_queue_job(
    session,
    job: str,
    ckpt: str,
    pipe_overrides: Optional[str] = None,
):
    rd = _run(session)
    opt = _ckpt(session, rd)
    if rd is None or opt is None:
        return HTMLResponse("")
    eval_root = opt.artifact_root

    if job == "default_video":
        cmd = ["uv", "run", "python", "src/training/evaluate.py", f"checkpoint={ckpt}"]
        start_job(eval_root, "default_video", cmd, _REPO)
    elif job == "default_heatmaps":
        cmd = ["uv", "run", "python", "src/training/evaluate_pipeline.py", f"checkpoint={ckpt}", "--heatmap"]
        start_job(eval_root, "default_heatmaps", cmd, _REPO)
    elif job == "pipeline":
        extra = (pipe_overrides or "--heatmap --cluster").strip().split()
        cmd = ["uv", "run", "python", "src/training/evaluate_pipeline.py", f"checkpoint={ckpt}"] + extra
        start_job(eval_root, "heatmap_pipeline", cmd, _REPO)

    return HTMLResponse(_queue_html(eval_root))


# ── Media server ──────────────────────────────────────────────────────────

@app.get("/media")
def serve_media(p: str):
    path = Path(urllib.parse.unquote(p))
    if not path.exists() or not path.is_file():
        return Response("Not found", status_code=404)
    suffix = path.suffix.lower()
    mime = {
        ".mp4": "video/mp4", ".gif": "image/gif",
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    }.get(suffix, "application/octet-stream")
    return Response(
        content=path.read_bytes(),
        media_type=mime,
        headers={"Cache-Control": "public, max-age=86400"},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SwarmEcho Artefacts")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--reload", action="store_true", default=False)
    args = parser.parse_args()

    host = args.host or _CFG_VALS.server.host
    port = args.port or _CFG_VALS.server.port

    print(f"\n{'═'*46}")
    print(f"  🔬 SwarmEcho Artefacts")
    print(f"  http://localhost:{port}")
    print(f"{'═'*46}\n")

    serve(host=host, port=port, reload=args.reload)
