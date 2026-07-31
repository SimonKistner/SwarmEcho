"""
config.py — Load and validate artefacts/config.yaml.

All sizes are in pixels unless otherwise noted.
Missing keys fall back to the defaults defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


# ---------------------------------------------------------------------------
# Dataclasses — one per section in config.yaml
# ---------------------------------------------------------------------------

@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class LayoutConfig:
    sidebar_width: int = 256


@dataclass
class TrainingConfig:
    video_max_height: int = 650
    video_col_width: int | None = None
    heatmap_max_height: int = 325
    heatmap_col_width: int | None = None


@dataclass
class EvalConfig:
    video_max_height: int = 700
    video_col_width: int | None = None
    heatmap_max_height: int = 450
    heatmap_col_width: int | None = None
    section2_width: int | None = None
    queue_width: int = 400
    map_height: int = 340
    map_width: int | None = None
    target_group_gap: int = 24


@dataclass
class ArtifactsListConfig:
    max_height: int = 120


@dataclass
class QueueConfig:
    poll_interval_seconds: int = 4


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    artifacts_list: ArtifactsListConfig = field(default_factory=ArtifactsListConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path: Path | None = None) -> Config:
    """
    Return the Config instance directly.
    Users can edit the defaults in this file to configure the dashboard.
    """
    return Config()


def css_vars(cfg: Config) -> str:
    """
    Return a <style> block that sets all layout CSS custom properties
    from the loaded config. Injected into every page's <head>.
    """
    t = cfg.training
    e = cfg.eval

    def val_to_style(val: int | None) -> str:
        return f"{val}px" if val is not None else "auto"

    return (
        "<style>:root{"
        f"--sidebar-w:{cfg.layout.sidebar_width}px;"
        f"--train-video-h:{val_to_style(t.video_max_height)};"
        f"--train-video-w:{val_to_style(t.video_col_width)};"
        f"--train-heat-h:{val_to_style(t.heatmap_max_height)};"
        f"--train-heat-col:{val_to_style(t.heatmap_col_width)};"
        f"--eval-video-h:{val_to_style(e.video_max_height)};"
        f"--eval-video-w:{val_to_style(e.video_col_width)};"
        f"--eval-heat-h:{val_to_style(e.heatmap_max_height)};"
        f"--eval-heat-w:{val_to_style(e.heatmap_col_width)};"
        f"--eval-sec2-w:{val_to_style(e.section2_width)};"
        f"--eval-queue-w:{val_to_style(e.queue_width)};"
        f"--eval-map-w:{val_to_style(e.map_width)};"
        f"--eval-map-h:{e.map_height}px;"
        f"--eval-tg-gap:{e.target_group_gap}px;"
        f"--artifact-list-h:{cfg.artifacts_list.max_height}px;"
        "}}</style>"
    )
