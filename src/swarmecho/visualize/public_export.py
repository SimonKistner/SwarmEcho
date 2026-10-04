"""Export one completed artifact as a static, browser-only inspector page."""

from __future__ import annotations

import gzip
import base64
import json
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


CHUNK_BYTES = 8 * 1024 * 1024
PUBLIC_MANIFEST_FIELDS = {
    "format", "map_name", "frames", "agents", "dt", "world_size_m",
    "cell_size_m", "coverage_voxel_size_m", "coverage_shape",
    "visual_radius_m", "comm_radius_m", "comm_radius_base_m",
    "success_condition", "allow_redundancy_reward", "robustness_runs",
    "obstacle_layout_mode", "map_suite_map_id", "random_eval_map_id",
    "source_map_name", "eval_name", "training_steps", "training_update",
}


def default_public_root() -> Path:
    return Path(__file__).resolve().parents[3] / "public-inspector"


def validate_page_name(name: str) -> str:
    if not isinstance(name, str) or len(name) > 80 or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
        raise ValueError("Page name must contain lowercase letters, numbers and single hyphens (max 80 characters).")
    return name


def validate_gallery_metadata(section: str, tags: list[str]) -> tuple[str, list[str]]:
    if not isinstance(section, str) or len(section) > 80:
        raise ValueError("Section name must be at most 80 characters.")
    section = " ".join(section.split()) or "General"
    if not isinstance(tags, list) or len(tags) > 12:
        raise ValueError("Provide at most 12 tags.")
    cleaned = []
    for tag in tags:
        if not isinstance(tag, str) or len(tag) > 32:
            raise ValueError("Each tag must be at most 32 characters.")
        tag = " ".join(tag.split())
        if tag and tag.casefold() not in {value.casefold() for value in cleaned}:
            cleaned.append(tag)
    return section, cleaned


def thumbnail_bytes(thumbnail: str | None) -> bytes | None:
    if thumbnail is None:
        return None
    prefix = "data:image/png;base64,"
    if not isinstance(thumbnail, str) or not thumbnail.startswith(prefix) or len(thumbnail) > 1500000:
        raise ValueError("Preview must be a PNG image no larger than 1 MiB.")
    try:
        data = base64.b64decode(thumbnail[len(prefix):], validate=True)
    except ValueError as exc:
        raise ValueError("Invalid preview image encoding.") from exc
    if len(data) > 1024 * 1024 or not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Preview must be a PNG image no larger than 1 MiB.")
    return data


def _publication_date(value: object, fallback: float) -> str:
    try:
        date = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
    except ValueError:
        date = datetime.fromtimestamp(fallback, timezone.utc)
    return date.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def public_catalogue(root: str | Path | None = None) -> list[dict]:
    """Read export metadata only, including older pages without gallery fields."""
    root = Path(root) if root is not None else default_public_root()
    catalogue = []
    for category in ("replays", "heatmaps"):
        for marker in sorted((root / category).glob("*/export.json")):
            item = json.loads(marker.read_text(encoding="utf-8"))
            if item.get("format") != "swarmecho-public-page/v1":
                continue
            section, tags = validate_gallery_metadata(item.get("section", "General"), item.get("tags", []))
            relative = marker.parent.relative_to(root).as_posix() + "/"
            catalogue.append({"title": item["title"], "kind": item["kind"], "path": relative,
                              "section": section, "tags": tags,
                              "published_at": _publication_date(item.get("published_at"), marker.stat().st_mtime),
                              "updated_at": item.get("updated_at"), "result": item.get("result"),
                              "success_rate": item.get("success_rate"),
                              "thumbnail": relative + "thumbnail.png" if item.get("thumbnail") == "thumbnail.png" else None})
    return sorted(catalogue, key=lambda item: (item["published_at"], item["path"]), reverse=True)


def public_sections() -> list[str]:
    # Catalogue order puts the most recent publication first. Keep that order
    # for section choices, coalescing names differing only by capitalization.
    names = {}
    for item in public_catalogue():
        names.setdefault(item["section"].casefold(), item["section"])
    return list(names.values())


def public_html(title: str, reference_path: bool, path_message: str) -> str:
    """Use the local viewer's rendering code with static loading and reduced UI."""
    from swarmecho.visualize.inspector import HTML

    config = json.dumps({"title": title, "reference_path": reference_path,
                         "path_message": path_message}).replace("<", "\\u003c")
    css = """<style>
    .public-inspector #artifactPicker,.public-inspector .header-refresh,
    .public-inspector #categoryNavigation,.public-inspector #stepNavigation,
    .public-inspector #mapNavigation,.public-inspector #mapNavigationHelp,
    .public-inspector #heatmapCommand,.public-inspector #copyHeatmapCommand,
    .public-inspector #exportDialog{display:none!important}
    .public-inspector header{height:52px!important;grid-template-columns:max-content minmax(0,1fr);grid-template-rows:52px!important;column-gap:24px;padding:0 20px}
    .public-inspector header.has-frame-navigation{height:96px!important;grid-template-rows:52px 44px!important}
    .public-inspector #layout{height:calc(100vh - 52px)!important}
    .public-inspector header.has-frame-navigation+#layout{height:calc(100vh - 96px)!important}
    .public-inspector header h1{font-size:18px;padding:0;letter-spacing:0}
    .public-inspector .header-selection{grid-column:2;grid-row:1;gap:20px;padding:0}
    .public-inspector .gallery-link{color:var(--cyan);text-decoration:none;white-space:nowrap;font-size:14px}
    .public-inspector #replayName{display:block!important;font-size:15px}
    .public-inspector .header-help{display:none}
    .public-inspector .header-status{grid-column:2;grid-row:1;justify-self:end;padding:0;white-space:normal}
    .public-inspector #frameNavigation{grid-column:1 / -1;grid-row:2!important;width:100%;max-width:none;justify-self:stretch;padding:0 0 8px;gap:12px}
    .public-inspector #frameNavigation .navigation-dot{display:none}
    .public-inspector #frameNavigation .navigation-arrow{font-size:18px;min-width:44px;padding:4px 12px;margin:0}
    .public-inspector #frameNavigation #frameSelection{min-width:80px;font-size:14px}
    .public-inspector #mainView > #scene{position:relative}
    .public-inspector #mainView > #scene:after{content:'Drag to orbit · scroll to zoom';position:absolute;bottom:10px;left:14px;color:var(--muted);font-size:11px;pointer-events:none;background:#0c1422b3;border-radius:4px;padding:4px 7px}
    </style>"""
    return HTML.replace(
        '<script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>',
        '<script src="../../assets/plotly.min.js"></script>',
    ).replace("</head>", css + f"<script>window.SWARMECHO_PUBLIC={config};</script></head>").replace(
        "<body>", '<body class="public-inspector">',
    ).replace("Click to create replay command", "Click to inspect target").replace(
        "Click a point to create a replay command.", "Click a point to inspect its coordinates.",
    ).replace(
        '<div class="header-selection">',
        '<div class="header-selection"><a class="gallery-link" href="../../">← All results</a>',
    )


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False,
                               separators=(",", ":")) + "\n", encoding="utf-8")


def export_public_artifact(
    source: str | Path, *, name: str, title: str = "", reference_path: bool = True,
    overwrite: bool = False, output: str | Path | None = None,
    section: str | None = None, tags: list[str] | None = None, thumbnail: str | None = None,
) -> dict:
    """Prepare a page locally; publishing is a separate commit/push operation.

    Payloads are compressed and split so each file stays below hosting asset
    limits. No checkpoints, paths, discovery indices or raw run folders are copied.
    """
    from plotly.offline import get_plotlyjs
    from swarmecho.visualize.inspector import heatmap_payload, replay_payload
    from swarmecho.visualize.replay_path import cached_route, compute_route, source_stamp

    name = validate_page_name(name)
    source = Path(source).resolve()
    if not source.is_file():
        raise ValueError("Artifact file does not exist.")
    if source.suffix.lower() == ".csv":
        kind, payload = "heatmap", heatmap_payload(source)
    elif source.suffix.lower() == ".json":
        kind, payload = "replay", replay_payload(source)
    else:
        raise ValueError("Select a replay JSON manifest or an evaluation heatmap CSV.")
    title = title.strip() or name.replace("-", " ").title()
    if len(title) > 160:
        raise ValueError("Page title must be at most 160 characters.")
    root = Path(output).resolve() if output is not None else default_public_root().resolve()
    destination = root / ("replays" if kind == "replay" else "heatmaps") / name
    # Existing folders must be known exports before any files may be replaced.
    previous = {}
    if destination.exists():
        if not overwrite:
            raise ValueError("This page already exists. Choose another name or enable replacement.")
        marker = destination / "export.json"
        previous = json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else {}
        if previous.get("format") != "swarmecho-public-page/v1":
            raise ValueError("Refusing to replace a folder that is not a public inspector export.")
    if not destination.resolve().is_relative_to(root):
        raise ValueError("Export destination must be inside the public inspector folder.")
    section, tags = validate_gallery_metadata(
        section if section is not None else previous.get("section", "General"),
        tags if tags is not None else previous.get("tags", []),
    )
    for item in public_catalogue(root):
        if item["section"].casefold() == section.casefold():
            section = item["section"]
            break
    image = thumbnail_bytes(thumbnail)
    now = datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
    published_at = _publication_date(previous.get("published_at"),
                                     (destination / "export.json").stat().st_mtime) if previous else now
    if kind == "replay":
        result = "success" if any(payload.get("success", [])) else "fail"
        success_rate = None
    else:
        result = None
        rates = payload.get("stage_rates", {}).get("chain_success")
        success_rate = sum(rates) / len(rates) if rates else None

    route, warning = None, ""
    if kind == "replay" and reference_path:
        try:
            stamp = source_stamp(source)
            route = cached_route(source, stamp) or compute_route(source)
            if source_stamp(source) != stamp:
                raise ValueError("Replay changed during export. Export it again after recording finishes.")
        except (ValueError, OSError, KeyError) as exc:
            warning = f"Reference path unavailable: {exc}"
    manifest = payload["manifest"]
    payload["manifest"] = {key: value for key, value in manifest.items()
                           if key in PUBLIC_MANIFEST_FIELDS}
    # Only the viewer's data is published; numerical arrays and geometry stay intact.
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    compressed = gzip.compress(encoded, compresslevel=6, mtime=0)
    public_message = "" if route is not None else (
        "Reference path unavailable for this export." if warning else "Reference path was not included in this export."
    )
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".export-", dir=root) as temporary:
        stage = Path(temporary)
        chunks = []
        for number, offset in enumerate(range(0, len(compressed), CHUNK_BYTES)):
            chunk = f"data-{number:04d}.gz"
            (stage / chunk).write_bytes(compressed[offset:offset + CHUNK_BYTES])
            chunks.append(chunk)
        _write_json(stage / "data.json", {"format": "swarmecho-public-data/v1",
                                         "encoding": "gzip", "chunks": chunks})
        _write_json(stage / "reference-path.json", {"status": "ready", "route": route}
                    if route is not None else {"status": "error", "message": public_message})
        (stage / "index.html").write_text(public_html(title, route is not None, public_message), encoding="utf-8")
        url_path = destination.relative_to(root).as_posix() + "/"
        _write_json(stage / "export.json", {"format": "swarmecho-public-page/v1",
                                           "title": title, "kind": kind, "path": url_path,
                                           "section": section, "tags": tags,
                                           "published_at": published_at, "updated_at": now,
                                           "result": result, "success_rate": success_rate,
                                           "thumbnail": "thumbnail.png" if image is not None else None})
        if image is not None:
            (stage / "thumbnail.png").write_bytes(image)
        assets = root / "assets"
        assets.mkdir(exist_ok=True)
        plotly = assets / "plotly.min.js"
        plotly_temp = stage / "plotly.min.js"
        plotly_temp.write_text(get_plotlyjs(), encoding="utf-8")
        plotly_temp.replace(plotly)
        destination.mkdir(parents=True, exist_ok=True)
        # Publish the new data index and HTML last, after their dependencies exist.
        for filename in [*chunks, *(["thumbnail.png"] if image is not None else []),
                         "reference-path.json", "export.json", "data.json", "index.html"]:
            (stage / filename).replace(destination / filename)
        if image is None:
            (destination / "thumbnail.png").unlink(missing_ok=True)
        # Remove obsolete chunks from this named export when replacing it.
        for old_chunk in destination.glob("data-*.gz"):
            if re.fullmatch(r"data-\d{4,}\.gz", old_chunk.name) and old_chunk.name not in chunks:
                old_chunk.unlink()
    (root / ".nojekyll").touch()
    if not (root / "index.html").exists():
        template = default_public_root() / "index.html"
        if template.is_file():
            shutil.copyfile(template, root / "index.html")
    catalogue = public_catalogue(root)
    catalogue_temp = root / ".catalogue.json.tmp"
    _write_json(catalogue_temp, catalogue)
    catalogue_temp.replace(root / "catalogue.json")
    size = sum(path.stat().st_size for path in destination.iterdir() if path.is_file())
    return {"path": "public-inspector/" + url_path if output is None else str(destination),
            "url_path": url_path, "size_mib": size / (1024 * 1024), "warning": warning}

