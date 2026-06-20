"""Streamlit dashboard for SwarmEcho training/eval artifacts.

Run with:
    uv run streamlit run src/analysis/eval_dashboard.py
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st
from omegaconf import OmegaConf

# Allow running via `streamlit run src/analysis/eval_dashboard.py`.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.artifacts import checkpoint_artifact_suffix, eval_checkpoint_artifact_root, parse_checkpoint_update, steps_for_update


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUTS = REPO_ROOT / "outputs"
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


st.set_page_config(page_title="SwarmEcho Eval Dashboard", page_icon="🎞️", layout="wide")
st.title("SwarmEcho Eval Dashboard")
st.caption("Inspect training artifacts, browse checkpoint evals, and queue targeted evaluation commands.")


@st.cache_data(ttl=30)
def discover_runs(outputs_root: str) -> list[Path]:
    root = Path(outputs_root)
    if not root.exists():
        return []
    runs: list[Path] = []
    ignored = {"checkpoints", "videos", "artifacts", "wandb", "logs", "files", "to_delete", ".git"}
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in ignored]
        path = Path(current)
        if (path / "config.yaml").exists() or (path / "checkpoints").exists() or (path / "artifacts").exists() or (path / "videos").exists():
            runs.append(path)
    return sorted(set(runs), key=lambda p: p.stat().st_mtime, reverse=True)


def load_cfg(run_dir: Path):
    cfg_path = run_dir / "config.yaml"
    if cfg_path.exists():
        return OmegaConf.load(cfg_path)
    return OmegaConf.create({"training": {"num_envs": 1024, "num_steps": 128}, "env": {}})


def files_in(paths: Iterable[Path], patterns: tuple[str, ...]) -> list[Path]:
    found: list[Path] = []
    for base in paths:
        if not base.exists():
            continue
        for pattern in patterns:
            found.extend(base.glob(pattern))
    return sorted(set(found), key=lambda p: p.name)


def choose_file(label: str, paths: list[Path], key: str) -> Path | None:
    if not paths:
        st.info(f"No {label.lower()} discovered.")
        return None
    names = [p.name for p in paths]
    idx_key = f"{key}_idx"
    st.session_state.setdefault(idx_key, 0)
    st.session_state[idx_key] = min(st.session_state[idx_key], len(paths) - 1)
    c_prev, c_sel, c_next = st.columns([1, 6, 1])
    with c_prev:
        if st.button("◀", key=f"{key}_prev", use_container_width=True):
            st.session_state[idx_key] = (st.session_state[idx_key] - 1) % len(paths)
    with c_sel:
        selected_name = st.selectbox(label, names, index=st.session_state[idx_key], key=key)
        st.session_state[idx_key] = names.index(selected_name)
    with c_next:
        if st.button("▶", key=f"{key}_next", use_container_width=True):
            st.session_state[idx_key] = (st.session_state[idx_key] + 1) % len(paths)
    selected = paths[st.session_state[idx_key]]
    st.caption(str(selected))
    return selected


def render_video_viewer(paths: list[Path], key: str) -> None:
    selected = choose_file("Video", paths, key)
    if selected:
        st.video(str(selected))


def render_image_viewer(label: str, paths: list[Path], key: str) -> Path | None:
    selected = choose_file(label, paths, key)
    if selected:
        st.image(str(selected), use_container_width=True)
    return selected


def artifact_count(run_dir: Path, ckpt: CheckpointOption | None) -> int:
    paths = [run_dir / "artifacts", run_dir / "videos"]
    if ckpt is not None:
        paths.append(ckpt.artifact_root)
    count = 0
    for base in paths:
        if base.exists():
            count += sum(1 for p in base.rglob("*") if p.is_file())
    return count


def checkpoint_options(run_dir: Path, cfg) -> list[CheckpointOption]:
    ckpt_dir = run_dir / "checkpoints"
    options: list[CheckpointOption] = []
    if not ckpt_dir.exists():
        return options
    for ckpt in sorted(ckpt_dir.glob("ckpt*")):
        if not ckpt.is_dir():
            continue
        update = parse_checkpoint_update(ckpt) or 0
        steps = steps_for_update(update, cfg)
        artifact_root = eval_checkpoint_artifact_root(run_dir, ckpt, cfg)
        options.append(CheckpointOption(ckpt, update, steps, artifact_root))
    return sorted(options, key=lambda item: item.update, reverse=True)


def map_preview_path(cfg) -> Path | None:
    names = cfg.get("env", {}).get("map_names", []) if cfg is not None else []
    if not names:
        return None
    preview = REPO_ROOT / "src" / "curriculum_config" / "maps" / f"{names[0]}_snapshot.png"
    return preview if preview.exists() else None


with st.sidebar:
    outputs_root = Path(st.text_input("Outputs root", value=str(DEFAULT_OUTPUTS)))
    if st.button("Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    runs = discover_runs(str(outputs_root))
    if not runs:
        st.warning("No runs found.")
        st.stop()
    run_labels = [str(p.relative_to(outputs_root)) if p.is_relative_to(outputs_root) else str(p) for p in runs]
    run_label = st.selectbox("Run", run_labels)
    run_dir = runs[run_labels.index(run_label)]
    cfg = load_cfg(run_dir)

    ckpts = checkpoint_options(run_dir, cfg)
    selected_ckpt = None
    if ckpts:
        ckpt_label = st.selectbox("Checkpoint", [c.label for c in ckpts])
        selected_ckpt = ckpts[[c.label for c in ckpts].index(ckpt_label)]
    else:
        st.info("No checkpoints discovered for this run.")

    st.metric("Discovered artifacts", artifact_count(run_dir, selected_ckpt))
    st.caption(str(run_dir))

training_tab, eval_tab = st.tabs(["Training", "Eval"])

with training_tab:
    st.header("Training artifacts")
    train_new = run_dir / "artifacts" / "train"
    train_legacy = run_dir / "videos" / "train"

    video_paths = files_in([train_new / "vids", train_legacy], VIDEO_EXTS)
    found_paths = files_in([train_new / "found_heatmaps", train_legacy / "found_heatmaps"], IMAGE_EXTS)
    chain_paths = files_in([train_new / "chain_heatmaps", train_legacy / "chain_heatmaps"], IMAGE_EXTS)

    left, mid, right = st.columns(3)
    with left:
        st.subheader("Video progression")
        render_video_viewer(video_paths, "train_video")
    with mid:
        st.subheader("Found / delivered heatmaps")
        render_image_viewer("Found / delivered heatmap", found_paths, "train_found")
    with right:
        st.subheader("Chain heatmaps")
        render_image_viewer("Chain heatmap", chain_paths, "train_chain")

with eval_tab:
    st.header("Checkpoint eval artifacts")
    if selected_ckpt is None:
        st.info("Select a checkpoint to inspect eval artifacts.")
        st.stop()

    eval_root = selected_ckpt.artifact_root
    legacy_eval = run_dir / "videos" / "eval"
    video_paths = files_in([eval_root / "vids", legacy_eval], VIDEO_EXTS)
    found_paths = files_in([eval_root / "found_heatmaps"], IMAGE_EXTS)
    chain_paths = files_in([eval_root / "chain_heatmaps"], IMAGE_EXTS)
    cluster_paths = files_in([eval_root / "clusters"], IMAGE_EXTS + VIDEO_EXTS)

    v_col, h_col = st.columns(2)
    with v_col:
        st.subheader("Selected video")
        render_video_viewer(video_paths, "eval_video")
    with h_col:
        heatmap_kind = st.radio("Heatmap type", ["Found / delivered", "Chain"], horizontal=True)
        if heatmap_kind == "Found / delivered":
            selected_heatmap = render_image_viewer("Eval heatmap", found_paths, "eval_found")
        else:
            selected_heatmap = render_image_viewer("Eval heatmap", chain_paths, "eval_chain")
        if selected_heatmap:
            stem = selected_heatmap.stem
            csv_candidates = list((eval_root / "data").glob(f"{stem}.points.csv")) + list((eval_root / "data").glob("*.points.csv"))
            if csv_candidates:
                st.caption("Heatmap point data")
                st.dataframe(pd.read_csv(csv_candidates[0]), use_container_width=True, height=220)

    if cluster_paths:
        with st.expander("Cluster artifacts", expanded=False):
            for path in cluster_paths:
                st.write(path.name)

    st.divider()
    st.subheader("Targeted render queue")
    preview = map_preview_path(cfg)
    if preview:
        st.image(str(preview), caption="Run map preview", use_container_width=True)
    env_cfg = cfg.get("env", {})
    width = float(env_cfg.get("box_width", 0.0) or 0.0)
    height = float(env_cfg.get("box_height", 0.0) or 0.0)
    tx = st.number_input("Target x", value=width / 2 if width else 0.0)
    ty = st.number_input("Target y", value=height / 2 if height else 0.0)
    renderer = st.selectbox("Renderer", ["fast", "slow"])
    overrides = st.text_input("Optional overrides", placeholder="env.max_steps=5000 visualize.render_conn_matrix=true")
    inside = (width <= 0 or 0 <= tx <= width) and (height <= 0 or 0 <= ty <= height)
    if not inside:
        st.error("Target is outside the configured map bounds.")
    command = [
        "uv", "run", "python", "src/training/evaluate.py",
        f"checkpoint={selected_ckpt.path}",
        f"target_pos={tx},{ty}",
        f"--renderer={renderer}",
    ] + overrides.split()
    st.code(" ".join(command), language="bash")
    if st.button("Queue targeted render", disabled=not inside):
        jobs_dir = eval_root / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        log_path = jobs_dir / f"target_x{tx:.2f}_y{ty:.2f}.log"
        with log_path.open("w") as log:
            subprocess.Popen(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
        st.success(f"Started background render. Log: {log_path}")

    with st.expander("Advanced / legacy eval tools"):
        st.markdown("Use these commands when you need the older bulk workflows:")
        st.code(f"uv run python src/training/evaluate.py checkpoint={selected_ckpt.path} selective=true eval_max_compute_episodes=100 render_successes=5 render_failures=5")
        st.code(f"uv run python src/training/evaluate_pipeline.py checkpoint={selected_ckpt.path} --heatmap --cluster")
        st.code(f"uv run python src/training/evaluate.py checkpoint={selected_ckpt.path} selective=true render_success_closest_to_corners=true")
