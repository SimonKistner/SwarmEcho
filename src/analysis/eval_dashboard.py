"""Streamlit dashboard for SwarmEcho training/eval artifacts.

Run with:
    uv run streamlit run src/analysis/eval_dashboard.py
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from omegaconf import OmegaConf

# Allow running via `streamlit run src/analysis/eval_dashboard.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.artifacts import eval_checkpoint_artifact_root, parse_checkpoint_update, steps_for_update


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_ROOT = REPO_ROOT / "outputs"
VIDEO_EXTS = ("*.mp4", "*.gif")
IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg")


@dataclass(frozen=True)
class CheckpointOption:
    path: Path
    update: int
    steps: int
    artifact_root: Path

    @property
    def label(self) -> str:
        dot = "🔵" if self.artifact_root.exists() else "⚪"
        return f"{dot} {self.path.name} · update {self.update:,} · {self.steps:,} steps"


st.set_page_config(page_title="SwarmEcho Eval", page_icon="🎞️", layout="wide")
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.1rem; }
    div[data-testid="stTabs"] button { font-weight: 800; font-size: 1.05rem; }
    div[data-testid="stMetric"] { background: rgba(120,120,120,.08); padding: .35rem .5rem; border-radius: .4rem; }
    .artifact-name { color: #888; font-size: .78rem; text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=30)
def discover_runs(outputs_root: str) -> list[Path]:
    root = Path(outputs_root)
    if not root.exists():
        return []
    runs: list[Path] = []
    ignored = {"checkpoints", "videos", "artifacts", "wandb", "logs", "files", "to_delete", ".git"}
    for current, dirs, _files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in ignored]
        path = Path(current)
        if (path / "config.yaml").exists() or (path / "checkpoints").exists() or (path / "artifacts").exists() or (path / "videos").exists():
            runs.append(path)
    return sorted(set(runs), key=lambda p: p.stat().st_mtime, reverse=True)


@st.cache_data(ttl=30)
def load_cfg_dict(run_dir: str) -> dict:
    cfg_path = Path(run_dir) / "config.yaml"
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {"training": {"num_envs": 1024, "num_steps": 128}, "env": {}}


@st.cache_data(ttl=30)
def discover_files(paths: tuple[str, ...], patterns: tuple[str, ...]) -> list[str]:
    found: list[Path] = []
    for raw in paths:
        base = Path(raw)
        if not base.exists():
            continue
        for pattern in patterns:
            found.extend(base.glob(pattern))
    return [str(p) for p in sorted(set(found), key=lambda item: item.name)]


@st.cache_data(ttl=120, show_spinner=False)
def load_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


def files_in(paths: Iterable[Path], patterns: tuple[str, ...]) -> list[Path]:
    return [Path(p) for p in discover_files(tuple(str(path) for path in paths), patterns)]


def cfg_get(cfg: dict, section: str, key: str, default=None):
    value = cfg.get(section, {}) if isinstance(cfg, dict) else {}
    return value.get(key, default) if isinstance(value, dict) else default


def clean_name(path: Path) -> str:
    stem = path.stem
    stem = stem.replace("eval_", "").replace("failed_chain_", "chain ")
    stem = stem.replace("found_and_delivered_", "found+delivered ")
    stem = stem.replace("delivered_", "delivered ").replace("found_", "found ")
    return stem.replace("_", " ")


def set_index(key: str, idx: int, size: int) -> None:
    if size <= 0:
        st.session_state[key] = 0
    else:
        st.session_state[key] = idx % size


def current_path(paths: list[Path], key: str) -> Path | None:
    if not paths:
        return None
    st.session_state.setdefault(key, 0)
    st.session_state[key] = min(int(st.session_state[key]), len(paths) - 1)
    return paths[int(st.session_state[key])]


def nav_controls(paths: list[Path], key: str, label: str) -> Path | None:
    if not paths:
        st.info(f"No {label.lower()} discovered.")
        return None
    selected = current_path(paths, key)
    left, mid, right = st.columns([1, 5, 1])
    with left:
        if st.button("◀", key=f"{key}_prev", width="stretch"):
            set_index(key, int(st.session_state[key]) - 1, len(paths))
            st.rerun()
    with mid:
        st.markdown(f"<div class='artifact-name'>{clean_name(selected)}<br>{selected.name}</div>", unsafe_allow_html=True)
    with right:
        if st.button("▶", key=f"{key}_next", width="stretch"):
            set_index(key, int(st.session_state[key]) + 1, len(paths))
            st.rerun()
    return current_path(paths, key)


def thumbnail_strip(paths: list[Path], key: str, label: str) -> None:
    if not paths:
        return
    st.caption(label)
    cols = st.columns(min(6, max(1, len(paths))))
    for idx, path in enumerate(paths[:12]):
        selected = idx == int(st.session_state.get(key, 0))
        button_label = ("● " if selected else "○ ") + clean_name(path)[:22]
        with cols[idx % len(cols)]:
            if st.button(button_label, key=f"{key}_thumb_{idx}", width="stretch"):
                set_index(key, idx, len(paths))
                st.rerun()


def preload_media(paths: list[Path], limit: int = 50) -> None:
    for path in paths[:limit]:
        load_bytes(str(path))


def show_video(path: Path | None) -> None:
    if path is not None:
        st.video(load_bytes(str(path)))


def show_image(path: Path | None) -> None:
    if path is not None:
        st.image(load_bytes(str(path)), width="stretch")


def artifact_count(run_dir: Path, ckpt: CheckpointOption | None) -> int:
    paths = [run_dir / "artifacts", run_dir / "videos"]
    if ckpt is not None:
        paths.append(ckpt.artifact_root)
    return sum(1 for base in paths if base.exists() for p in base.rglob("*") if p.is_file())


def checkpoint_options(run_dir: Path, cfg: dict) -> list[CheckpointOption]:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return []
    # Tiny object with cfg.training.get(...) compatibility for shared helpers.
    cfg_obj = type("Cfg", (), {"training": cfg.get("training", {})})()
    options: list[CheckpointOption] = []
    for ckpt in sorted(ckpt_dir.glob("ckpt*")):
        if not ckpt.is_dir():
            continue
        update = parse_checkpoint_update(ckpt) or 0
        steps = steps_for_update(update, cfg_obj)
        artifact_root = eval_checkpoint_artifact_root(run_dir, ckpt, cfg_obj)
        options.append(CheckpointOption(ckpt, update, steps, artifact_root))
    return sorted(options, key=lambda item: item.update, reverse=True)


def map_name_from_cfg(cfg: dict) -> str | None:
    names = cfg_get(cfg, "env", "map_names", []) or []
    return str(names[0]) if names else None


def map_preview_path(cfg: dict) -> Path | None:
    map_name = map_name_from_cfg(cfg)
    if not map_name:
        return None
    generated = REPO_ROOT / "outputs" / "map_previews" / f"map_preview_{map_name}.png"
    if generated.exists():
        return generated
    legacy = REPO_ROOT / "src" / "curriculum_config" / "maps" / f"{map_name}_snapshot.png"
    return legacy if legacy.exists() else generated


def ensure_map_preview_job(eval_root: Path, cfg: dict) -> None:
    preview = map_preview_path(cfg)
    map_name = map_name_from_cfg(cfg)
    if not map_name or (preview is not None and preview.exists()):
        return
    jobs_dir = eval_root / "jobs"
    existing = list(jobs_dir.glob("*map_preview*.json")) if jobs_dir.exists() else []
    if existing:
        return
    start_job(
        eval_root,
        "map_preview",
        ["uv", "run", "python", "src/visualize/render_preview.py", map_name, "--mode", "image", "--format", "png", "--no-spawns"],
    )


def point_tables(eval_root: Path) -> list[Path]:
    data_dir = eval_root / "data"
    return sorted(data_dir.glob("*.points.csv"), key=lambda p: p.name) if data_dir.exists() else []


def write_job(job_dir: Path, payload: dict) -> Path:
    job_dir.mkdir(parents=True, exist_ok=True)
    job_path = job_dir / f"{payload['id']}.json"
    job_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return job_path


def start_job(eval_root: Path, name: str, command: list[str]) -> None:
    jobs_dir = eval_root / "jobs"
    logs_dir = jobs_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    job_id = f"{int(time.time())}_{name}"
    log_path = logs_dir / f"{job_id}.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
    write_job(
        jobs_dir,
        {
            "id": job_id,
            "name": name,
            "status": "running",
            "pid": proc.pid,
            "command": command,
            "log_path": str(log_path),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    st.toast(f"Started {name}", icon="⏳")


def pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def job_finished_badly(log_path: str | None) -> bool:
    if not log_path:
        return False
    path = Path(log_path)
    if not path.exists():
        return False
    tail = path.read_text(errors="ignore")[-4000:].lower()
    return "traceback" in tail or "error:" in tail or "failed" in tail


def queue_items(eval_root: Path) -> list[dict]:
    jobs_dir = eval_root / "jobs"
    if not jobs_dir.exists():
        return []
    items = []
    changed = False
    for job_file in sorted(jobs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        data = json.loads(job_file.read_text())
        if data.get("status") == "running" and not pid_running(int(data.get("pid", -1))):
            data["status"] = "crashed" if job_finished_badly(data.get("log_path")) else "finished"
            data["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            job_file.write_text(json.dumps(data, indent=2, sort_keys=True))
            changed = True
        items.append(data)
    if changed:
        st.cache_data.clear()
    return items


def render_queue(eval_root: Path) -> None:
    items = queue_items(eval_root)
    if not items:
        st.info("Queue is empty.")
        return
    status_icon = {"running": "⏳ running", "finished": "✅ finished", "crashed": "❌ crashed"}
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "status": status_icon.get(item.get("status"), item.get("status")),
                    "name": item.get("name"),
                }
                for item in items
            ]
        ),
        width="stretch",
        height=150,
        hide_index=True,
    )
    if any(item.get("status") == "running" for item in items):
        st.info("⏳ Job running. This dashboard auto-refreshes every 30 seconds while work is active.")
        st.markdown("<meta http-equiv='refresh' content='30'>", unsafe_allow_html=True)
    with st.expander("Queue details / logs", expanded=False):
        for item in items:
            st.code(" ".join(item.get("command", [])), language="bash")
            st.caption(item.get("log_path", ""))


def selected_plotly_point(event) -> tuple[float, float] | None:
    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, dict):
        selection = event.get("selection")
    points = getattr(selection, "points", None) if selection is not None else None
    if points is None and isinstance(selection, dict):
        points = selection.get("points")
    if not points:
        return None
    point = points[0]
    custom = point.get("customdata") if isinstance(point, dict) else getattr(point, "customdata", None)
    if custom is not None and len(custom) >= 2:
        return float(custom[0]), float(custom[1])
    if isinstance(point, dict) and "x" in point and "y" in point:
        return float(point["x"]), float(point["y"])
    return None


def render_target_picker(eval_root: Path, cfg: dict) -> None:
    preview = map_preview_path(cfg)
    if preview is None or not preview.exists():
        st.info("Map preview is being rendered in the queue. The target picker will appear as soon as it is ready.")
        return

    env_width = float(cfg_get(cfg, "env", "box_width", 0.0) or 0.0)
    env_height = float(cfg_get(cfg, "env", "box_height", 0.0) or 0.0)
    if env_width <= 0 or env_height <= 0:
        st.error("Cannot render interactive target picker because map dimensions are missing from the run config.")
        return

    st.session_state.setdefault("target_x", env_width / 2)
    st.session_state.setdefault("target_y", env_height / 2)
    mode = st.segmented_control("Selection mode", ["Free", "Found overlay", "Chain overlay"], default="Free")

    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source="data:image/png;base64," + base64.b64encode(load_bytes(str(preview))).decode("ascii"),
            xref="x", yref="y", x=0, y=env_height, sizex=env_width, sizey=env_height, sizing="stretch", layer="below",
        )
    )

    step = max(1.0, max(env_width, env_height) / 120.0)
    xs = []
    ys = []
    x = 0.0
    while x <= env_width:
        y = 0.0
        while y <= env_height:
            xs.append(x)
            ys.append(y)
            y += step
        x += step
    fig.add_trace(go.Scattergl(
        x=xs, y=ys, mode="markers", name="Free target grid",
        marker={"size": 8, "color": "rgba(0,0,0,0.01)"},
        customdata=list(zip(xs, ys)), hovertemplate="free target<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
    ))

    table_paths = point_tables(eval_root)
    wanted = []
    if mode == "Found overlay":
        wanted = [p for p in table_paths if "found" in p.name or "delivered" in p.name]
    elif mode == "Chain overlay":
        wanted = [p for p in table_paths if "chain" in p.name or "failed" in p.name]
    for points_path in wanted:
        df = pd.read_csv(points_path)
        if {"x", "y"}.issubset(df.columns):
            color = "#2563eb" if mode == "Found overlay" else "#f97316"
            fig.add_trace(go.Scattergl(
                x=df["x"], y=df["y"], mode="markers", name=points_path.stem,
                marker={"size": 8, "color": color, "opacity": 0.75},
                customdata=df[["x", "y"]].to_numpy(),
                hovertemplate=f"{points_path.stem}<br>x=%{{x:.2f}}<br>y=%{{y:.2f}}<extra></extra>",
            ))

    fig.add_trace(go.Scatter(
        x=[st.session_state.target_x], y=[st.session_state.target_y], mode="markers", name="Selected target",
        marker={"size": 16, "color": "#ef4444", "symbol": "x", "line": {"width": 2, "color": "white"}},
        hovertemplate="selected<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
    ))
    fig.update_xaxes(range=[0, env_width], visible=False, constrain="domain")
    fig.update_yaxes(range=[0, env_height], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(
        height=520, margin={"l": 0, "r": 0, "t": 0, "b": 0}, showlegend=False,
        dragmode="select", clickmode="event+select",
    )

    event = st.plotly_chart(fig, width="stretch", key="target_picker", on_select="rerun", selection_mode=("points", "box", "lasso"))
    selected = selected_plotly_point(event)
    if selected is not None:
        st.session_state.target_x, st.session_state.target_y = selected
        st.rerun()


with st.sidebar:
    st.markdown("### Matrix Selection")
    if st.button("Refresh", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    runs = discover_runs(str(OUTPUTS_ROOT))
    if not runs:
        st.warning(f"No runs found below {OUTPUTS_ROOT}.")
        st.stop()

    grouped_runs: dict[str, list[Path]] = {}
    for run in runs:
        group = run.parent.name if run.parent != OUTPUTS_ROOT else "Standalone"
        grouped_runs.setdefault(group, []).append(run)

    selected_run_label = None
    run_lookup: dict[str, Path] = {}
    for group, group_runs in grouped_runs.items():
        st.caption(group)
        labels = [run.name for run in group_runs]
        default = 0
        selected = st.radio(group, labels, index=default, label_visibility="collapsed", key=f"run_group_{group}")
        label = f"{group} | {selected}"
        run_lookup[label] = group_runs[labels.index(selected)]
        if selected_run_label is None:
            selected_run_label = label

    selected_run_label = st.selectbox("Selected run", list(run_lookup), label_visibility="collapsed")
    run_dir = run_lookup[selected_run_label]
    cfg = load_cfg_dict(str(run_dir))
    st.caption(str(run_dir))

ckpts = checkpoint_options(run_dir, cfg)
selected_ckpt = None
if ckpts:
    ckpt_label = st.selectbox("Checkpoint", [item.label for item in ckpts], label_visibility="visible")
    selected_ckpt = ckpts[[item.label for item in ckpts].index(ckpt_label)]
else:
    st.warning("No checkpoints discovered for this run.")

st.caption(f"Artifacts discovered: {artifact_count(run_dir, selected_ckpt)}")
training_tab, eval_tab = st.tabs(["TRAINING", "EVAL"])

with training_tab:
    train_new = run_dir / "artifacts" / "train"
    train_legacy = run_dir / "videos" / "train"
    video_paths = files_in([train_new / "vids", train_legacy], VIDEO_EXTS)
    found_paths = files_in([train_new / "found_heatmaps", train_legacy / "found_heatmaps"], IMAGE_EXTS)
    chain_paths = files_in([train_new / "chain_heatmaps", train_legacy / "chain_heatmaps"], IMAGE_EXTS)
    preload_media(video_paths + found_paths + chain_paths)

    video_col, heat_col, _spacer_col = st.columns([0.35, 0.14, 0.51])
    with video_col:
        selected_video = current_path(video_paths, "train_video_idx")
        show_video(selected_video)
        nav_controls(video_paths, "train_video_idx", "training video")
        thumbnail_strip(video_paths, "train_video_idx", "Videos")
    with heat_col:
        selected_found = current_path(found_paths, "train_found_idx")
        show_image(selected_found)
        nav_controls(found_paths, "train_found_idx", "found/delivered heatmap")
        thumbnail_strip(found_paths, "train_found_idx", "Found / delivered")
        selected_chain = current_path(chain_paths, "train_chain_idx")
        show_image(selected_chain)
        nav_controls(chain_paths, "train_chain_idx", "chain heatmap")
        thumbnail_strip(chain_paths, "train_chain_idx", "Chain")

with eval_tab:
    if selected_ckpt is None:
        st.stop()

    eval_root = selected_ckpt.artifact_root
    ensure_map_preview_job(eval_root, cfg)
    legacy_eval = run_dir / "videos" / "eval"
    video_paths = files_in([eval_root / "vids", legacy_eval], VIDEO_EXTS)
    found_paths = files_in([eval_root / "found_heatmaps"], IMAGE_EXTS)
    chain_paths = files_in([eval_root / "chain_heatmaps"], IMAGE_EXTS)
    cluster_paths = files_in([eval_root / "clusters"], IMAGE_EXTS + VIDEO_EXTS)
    preload_media(video_paths + found_paths + chain_paths + cluster_paths)

    video_col, heatmap_col, _spacer_col = st.columns([0.35, 0.14, 0.51])
    with video_col:
        selected_video = current_path(video_paths, "eval_video_idx")
        show_video(selected_video)
        nav_controls(video_paths, "eval_video_idx", "eval video")
        thumbnail_strip(video_paths, "eval_video_idx", "Videos")
        if not video_paths:
            cmd = ["uv", "run", "python", "src/training/evaluate.py", f"checkpoint={selected_ckpt.path}"]
            if st.button("Create default video", width="stretch"):
                start_job(eval_root, "default_video", cmd)
                st.rerun()

    with heatmap_col:
        heatmap_kind = st.segmented_control("Heatmap", ["Found / delivered", "Chain"], default="Found / delivered")
        heatmap_paths = found_paths if heatmap_kind == "Found / delivered" else chain_paths
        state_key = "eval_found_idx" if heatmap_kind == "Found / delivered" else "eval_chain_idx"
        selected_heatmap = current_path(heatmap_paths, state_key)
        show_image(selected_heatmap)
        nav_controls(heatmap_paths, state_key, "eval heatmap")
        thumbnail_strip(heatmap_paths, state_key, "Heatmaps")
        if not heatmap_paths:
            cmd = ["uv", "run", "python", "src/training/evaluate_pipeline.py", f"checkpoint={selected_ckpt.path}", "--heatmap"]
            if st.button("Create default heatmaps", width="stretch"):
                start_job(eval_root, "default_heatmaps", cmd)
                st.rerun()

    st.markdown("---")
    q_col, target_col = st.columns([0.24, 0.76])
    with q_col:
        st.markdown("#### Queue")
        render_queue(eval_root)
        if st.button("Refresh queue", width="stretch"):
            st.rerun()

    with target_col:
        st.markdown("#### Target selection")
        env_width = float(cfg_get(cfg, "env", "box_width", 0.0) or 0.0)
        env_height = float(cfg_get(cfg, "env", "box_height", 0.0) or 0.0)
        st.session_state.setdefault("target_x", env_width / 2 if env_width else 0.0)
        st.session_state.setdefault("target_y", env_height / 2 if env_height else 0.0)

        render_target_picker(eval_root, cfg)

        coord_col_1, coord_col_2 = st.columns(2)
        with coord_col_1:
            tx = st.number_input("x", value=float(st.session_state.target_x), key="target_x_input", label_visibility="visible")
        with coord_col_2:
            ty = st.number_input("y", value=float(st.session_state.target_y), key="target_y_input", label_visibility="visible")
        st.session_state.target_x = tx
        st.session_state.target_y = ty

        overrides = st.text_input("Overrides", value="", placeholder="env.max_steps=5000 --renderer=slow")
        inside = (env_width <= 0 or 0 <= tx <= env_width) and (env_height <= 0 or 0 <= ty <= env_height)
        if not inside:
            st.error("Target is outside the configured map bounds.")
        command = [
            "uv", "run", "python", "src/training/evaluate.py",
            f"checkpoint={selected_ckpt.path}",
            f"target_pos={tx},{ty}",
            "--renderer=fast",
        ] + overrides.split()
        with st.expander("Command preview"):
            st.code(" ".join(command), language="bash")
        if st.button("Start targeted render", type="primary", disabled=not inside, width="stretch"):
            start_job(eval_root, f"target_x{tx:.2f}_y{ty:.2f}", command)
            st.rerun()

    if cluster_paths:
        with st.expander("Cluster artifacts", expanded=False):
            for path in cluster_paths:
                st.write(path.name)

    with st.expander("Advanced / legacy eval tools"):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            max_eps = st.number_input("Max compute episodes", min_value=1, value=100, step=10)
            successes = st.number_input("Success videos", min_value=0, value=5, step=1)
            failures = st.number_input("Failure videos", min_value=0, value=5, step=1)
            overrides_sel = st.text_input("Selective overrides", key="selective_overrides")
            selective_cmd = [
                "uv", "run", "python", "src/training/evaluate.py", f"checkpoint={selected_ckpt.path}",
                "selective=true", f"eval_max_compute_episodes={int(max_eps)}",
                f"render_successes={int(successes)}", f"render_failures={int(failures)}",
            ] + overrides_sel.split()
            if st.button("Queue selective render", width="stretch"):
                start_job(eval_root, "selective_render", selective_cmd)
                st.rerun()
        with col_b:
            pipeline_overrides = st.text_input("Pipeline overrides", value="--heatmap --cluster")
            pipeline_cmd = ["uv", "run", "python", "src/training/evaluate_pipeline.py", f"checkpoint={selected_ckpt.path}"] + pipeline_overrides.split()
            if st.button("Queue heatmap pipeline", width="stretch"):
                start_job(eval_root, "heatmap_pipeline", pipeline_cmd)
                st.rerun()
        with col_c:
            corner_overrides = st.text_input("Corner overrides", value="selective=true render_success_closest_to_corners=true")
            corner_cmd = ["uv", "run", "python", "src/training/evaluate.py", f"checkpoint={selected_ckpt.path}"] + corner_overrides.split()
            if st.button("Queue corner render", width="stretch"):
                start_job(eval_root, "corner_render", corner_cmd)
                st.rerun()
