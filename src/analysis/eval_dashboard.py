"""Streamlit dashboard for SwarmEcho training/eval artifacts.

Run with:
    uv run streamlit run src/analysis/eval_dashboard.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
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


def map_preview_path(cfg: dict) -> Path | None:
    names = cfg_get(cfg, "env", "map_names", []) or []
    if not names:
        return None
    preview = REPO_ROOT / "src" / "curriculum_config" / "maps" / f"{names[0]}_snapshot.png"
    return preview if preview.exists() else None


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


def queue_items(eval_root: Path) -> list[dict]:
    jobs_dir = eval_root / "jobs"
    if not jobs_dir.exists():
        return []
    items = []
    changed = False
    for job_file in sorted(jobs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        data = json.loads(job_file.read_text())
        if data.get("status") == "running" and not pid_running(int(data.get("pid", -1))):
            data["status"] = "finished"
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
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "status": item.get("status"),
                    "name": item.get("name"),
                    "pid": item.get("pid"),
                    "started": item.get("started_at"),
                    "finished": item.get("finished_at", ""),
                    "log": item.get("log_path"),
                }
                for item in items
            ]
        ),
        width="stretch",
        height=150,
        hide_index=True,
    )
    if any(item.get("status") == "running" for item in items):
        st.info("⏳ Render/eval job running… use Refresh queue to update status.")


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

    selected_video = current_path(video_paths, "train_video_idx")
    show_video(selected_video)
    nav_controls(video_paths, "train_video_idx", "training video")
    thumbnail_strip(video_paths, "train_video_idx", "Videos")

    found_col, chain_col = st.columns(2)
    with found_col:
        selected_found = current_path(found_paths, "train_found_idx")
        show_image(selected_found)
        nav_controls(found_paths, "train_found_idx", "found/delivered heatmap")
        thumbnail_strip(found_paths, "train_found_idx", "Found / delivered")
    with chain_col:
        selected_chain = current_path(chain_paths, "train_chain_idx")
        show_image(selected_chain)
        nav_controls(chain_paths, "train_chain_idx", "chain heatmap")
        thumbnail_strip(chain_paths, "train_chain_idx", "Chain")

with eval_tab:
    if selected_ckpt is None:
        st.stop()

    eval_root = selected_ckpt.artifact_root
    legacy_eval = run_dir / "videos" / "eval"
    video_paths = files_in([eval_root / "vids", legacy_eval], VIDEO_EXTS)
    found_paths = files_in([eval_root / "found_heatmaps"], IMAGE_EXTS)
    chain_paths = files_in([eval_root / "chain_heatmaps"], IMAGE_EXTS)
    cluster_paths = files_in([eval_root / "clusters"], IMAGE_EXTS + VIDEO_EXTS)
    preload_media(video_paths + found_paths + chain_paths + cluster_paths)

    video_col, heatmap_col = st.columns(2)
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
    q_col, target_col = st.columns([1, 1.25])
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

        preview = map_preview_path(cfg)
        if preview:
            st.image(load_bytes(str(preview)), caption="Map preview", width="stretch")

        tables = point_tables(eval_root)
        if tables:
            csv_path = st.selectbox("Overlay points", tables, format_func=lambda p: p.name)
            points_df = pd.read_csv(csv_path)
            st.dataframe(points_df.head(250), width="stretch", height=180, hide_index=True)
            row_idx = st.number_input("Use point row", min_value=0, max_value=max(0, len(points_df) - 1), value=0, step=1)
            if st.button("Set target from selected point", width="stretch"):
                st.session_state.target_x = float(points_df.iloc[int(row_idx)]["x"])
                st.session_state.target_y = float(points_df.iloc[int(row_idx)]["y"])
                st.rerun()
        else:
            st.caption("No point CSVs available yet. Create heatmaps to enable overlay-guided target selection.")

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
