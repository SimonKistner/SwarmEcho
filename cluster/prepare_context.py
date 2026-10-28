"""Build a small, explicit Determined model context from this repository.

Run with a system Python: python cluster/prepare_context.py
The resulting directory is passed as the model_def argument to det experiment create.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "swarmecho"
ROOT_FILES = ("pyproject.toml", "uv.lock", "README.md", "cluster/run_driver.py")
SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".html", ".txt"}
MAX_BYTES = 90 * 1024 * 1024


def source_files() -> list[Path]:
    files = [ROOT / name for name in ROOT_FILES]
    for parent, dirs, names in os.walk(PACKAGE):
        dirs[:] = sorted(name for name in dirs if name not in {"pools", "__pycache__"})
        if any((Path(parent) / name).is_symlink() for name in dirs):
            raise ValueError("Refusing a source directory symlink: {}".format(parent))
        for name in sorted(names):
            path = Path(parent) / name
            if path.is_symlink():
                raise ValueError(f"Refusing symlink in model context: {path}")
            if path.suffix not in SOURCE_SUFFIXES:
                raise ValueError(f"Review unexpected source file before upload: {path}")
            files.append(path)
    if any(path.is_symlink() for path in files):
        raise ValueError("Refusing a model context symlink.")
    if not all(path.is_file() for path in files):
        raise FileNotFoundError("A required model context file is missing.")
    return files


def prepare_context(target=None):
    """Copy an allowlisted snapshot, without importing any training modules."""
    files = source_files()
    total_bytes = sum(path.stat().st_size for path in files)
    if total_bytes > MAX_BYTES:
        raise ValueError(f"Model context is too large: {total_bytes} bytes")

    if target is None:
        target = Path(tempfile.mkdtemp(prefix=".context-", dir=Path(__file__).parent))
    else:
        target = Path(target)
        target.mkdir(parents=True, exist_ok=False)
    for path in files:
        destination = target / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    return target, len(files), total_bytes


def main() -> None:
    target, count, total_bytes = prepare_context()
    print(f"Prepared {count} files ({total_bytes:,} bytes) in {target}")
    print("No .env, outputs, W&B runs, existing map pools, or project virtual environment copied.")


if __name__ == "__main__":
    main()
