"""Command-line launchers for framework-owned analysis applications."""

from __future__ import annotations

from pathlib import Path
import sys


def _run_streamlit(filename: str) -> None:
    from streamlit.web import cli as streamlit_cli

    dashboard = Path(__file__).with_name(filename)
    sys.argv = ["streamlit", "run", str(dashboard), *sys.argv[1:]]
    streamlit_cli.main()


def dashboard_main() -> None:
    """Launch the cross-run configuration and video dashboard."""
    _run_streamlit("dashboard.py")


