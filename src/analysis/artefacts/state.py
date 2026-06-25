"""
state.py — Pure data-access helpers for the SwarmEcho Artefacts dashboard.

No web framework dependency. All functions return plain Python types so they
can be unit-tested independently of the HTTP layer.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Allow import when run from the project root via `uv run`.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from omegaconf import OmegaConf

from training.artifacts import (
    eval_checkpoint_artifact_root,
    parse_checkpoint_update,
    steps_for_update,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS_ROOT = REPO_ROOT / "outputs"

VIDEO_EXTS = ("*.mp4", "*.gif")
IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg")

_IGNORED_DIRS = {"checkpoints", "videos", "artifacts", "wandb", "logs", "files", "to_delete", ".git"}


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

def discover_runs(outputs_root: Path | None = None) -> list[Path]:
    """Return all run directories, newest-first."""
    root = outputs_root or OUTPUTS_ROOT
    if not root.exists():
        return []
    runs: set[Path] = set()
    for current, dirs, _ in os.walk(root):
        dirs[:] = [d for d in dirs if d not in _IGNORED_DIRS]
        path = Path(current)
        if (
            (path / "config.yaml").exists()
            or (path / "checkpoints").exists()
            or (path / "artifacts").exists()
            or (path / "videos").exists()
        ):
            runs.add(path)
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def group_runs(runs: list[Path]) -> dict[str, list[Path]]:
    """Group run paths by their parent directory name."""
    groups: dict[str, list[Path]] = {}
    for run in runs:
        group = run.parent.name if run.parent != OUTPUTS_ROOT else "Standalone"
        groups.setdefault(group, []).append(run)
    return groups


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_cfg(run_dir: Path) -> dict:
    cfg_path = run_dir / "config.yaml"
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)  # type: ignore[return-value]
    return {"training": {"num_envs": 1024, "num_steps": 128}, "env": {}}


def cfg_get(cfg: dict, section: str, key: str, default: Any = None) -> Any:
    value = cfg.get(section, {}) if isinstance(cfg, dict) else {}
    return value.get(key, default) if isinstance(value, dict) else default


# ---------------------------------------------------------------------------
# Checkpoint enumeration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CheckpointOption:
    path: Path
    update: int
    steps: int
    artifact_root: Path

    @property
    def label(self) -> str:
        dot = "●" if self.artifact_root.exists() else "○"
        steps_str = _fmt_steps(self.steps)
        return f"{dot} {self.path.name}  ·  update {self.update:,}  ·  {steps_str} steps"

    @property
    def short_label(self) -> str:
        return f"update {self.update:,} · {_fmt_steps(self.steps)} steps"

    @property
    def has_artifacts(self) -> bool:
        return self.artifact_root.exists()


def _fmt_steps(steps: int) -> str:
    m = steps // 1_000_000
    k = (steps % 1_000_000) // 1_000
    return f"{m}M" if m else f"{k}k" if k else str(steps)


def checkpoint_options(run_dir: Path, cfg: dict) -> list[CheckpointOption]:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return []
    cfg_obj = type("Cfg", (), {"training": cfg.get("training", {})})()
    options: list[CheckpointOption] = []
    for ckpt in sorted(ckpt_dir.glob("ckpt*")):
        if not ckpt.is_dir():
            continue
        update = parse_checkpoint_update(ckpt) or 0
        steps = steps_for_update(update, cfg_obj)
        artifact_root = eval_checkpoint_artifact_root(run_dir, ckpt, cfg_obj)
        options.append(CheckpointOption(ckpt, update, steps, artifact_root))
    return sorted(options, key=lambda o: o.update, reverse=True)


# ---------------------------------------------------------------------------
# Artifact enumeration
# ---------------------------------------------------------------------------

def _glob_paths(directories: list[Path], patterns: tuple[str, ...]) -> list[Path]:
    found: list[Path] = []
    for base in directories:
        if not base.exists():
            continue
        for pattern in patterns:
            found.extend(base.glob(pattern))
    return sorted(set(found), key=lambda p: p.name)


def training_artifacts(run_dir: Path) -> dict[str, list[Path]]:
    train_new = run_dir / "artifacts" / "train"
    train_legacy = run_dir / "videos" / "train"
    return {
        "videos": _glob_paths(
            [train_new / "vids", train_legacy], VIDEO_EXTS
        ),
        "found_heatmaps": _glob_paths(
            [train_new / "found_heatmaps", train_legacy / "found_heatmaps"], IMAGE_EXTS
        ),
        "chain_heatmaps": _glob_paths(
            [train_new / "chain_heatmaps", train_legacy / "chain_heatmaps"], IMAGE_EXTS
        ),
    }


def eval_artifacts(eval_root: Path, legacy_eval: Path | None = None) -> dict[str, list[Path]]:
    dirs_vid = [eval_root / "vids"]
    if legacy_eval is not None:
        dirs_vid.append(legacy_eval)
    return {
        "videos": _glob_paths(dirs_vid, VIDEO_EXTS),
        "found_heatmaps": _glob_paths([eval_root / "found_heatmaps"], IMAGE_EXTS),
        "chain_heatmaps": _glob_paths([eval_root / "chain_heatmaps"], IMAGE_EXTS),
    }


# ---------------------------------------------------------------------------
# Map preview
# ---------------------------------------------------------------------------

def map_name_from_cfg(cfg: dict) -> str | None:
    names = cfg_get(cfg, "env", "map_names", []) or []
    return str(names[0]) if names else None


def map_preview_path(run_dir: Path, cfg: dict) -> Path:
    """
    Return the path where the clean map preview PNG for this run should live.
    Stored at <run_dir>/artifacts/eval/map_preview_<map_name>.png so it is
    shared across all checkpoints in the same run.
    The file may not exist yet — callers should check .exists().
    """
    map_name = map_name_from_cfg(cfg)
    if not map_name:
        return run_dir / "artifacts" / "eval" / "map_preview_unknown.png"
    return run_dir / "artifacts" / "eval" / f"map_preview_{map_name}.png"


# ---------------------------------------------------------------------------
# Artifact filename parsing
# ---------------------------------------------------------------------------

_UPDATE_RE = re.compile(r"u(\d{6})")
_STEPS_RE = re.compile(r"s(\d+)(M|k)?")


def parse_artifact_label(path: Path) -> str:
    """Extract a human-readable progression label from an artifact filename."""
    name = path.stem
    m_u = _UPDATE_RE.search(name)
    m_s = _STEPS_RE.search(name)
    if m_u and m_s:
        update = int(m_u.group(1))
        raw = m_s.group(1)
        suffix = m_s.group(2) or ""
        return f"u{update:,} · {raw}{suffix}"
    stem = (
        name.replace("eval_", "")
        .replace("failed_chain_", "chain ")
        .replace("found_and_delivered_", "found ")
        .replace("delivered_", "delivered ")
        .replace("found_", "found ")
        .replace("_heatmap", "")
        .replace("_", " ")
    )
    return stem.strip()


def video_status(path: Path) -> str | None:
    """Return 'SUCCESS', 'FAIL', or None based on filename prefix."""
    name = path.name.upper()
    if name.startswith("SUCCESS"):
        return "SUCCESS"
    if name.startswith("FAIL"):
        return "FAIL"
    return None
