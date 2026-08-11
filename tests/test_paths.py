from pathlib import Path

from swarmecho.core.paths import normalize_wsl_path


def test_normalize_wsl_path_converts_windows_drive_paths():
    assert normalize_wsl_path(
        r"Q:\_0_Projects\000_SwarmEcho\SwarmEcho\outputs\run\checkpoints\ckpt_001201"
    ) == Path(
        "/mnt/q/_0_Projects/000_SwarmEcho/SwarmEcho/outputs/run/checkpoints/ckpt_001201"
    )


def test_normalize_wsl_path_preserves_linux_paths():
    path = "/mnt/q/_0_Projects/000_SwarmEcho/SwarmEcho/outputs/run"
    assert normalize_wsl_path(path) == Path(path)


def test_normalize_wsl_path_accepts_forward_slash_windows_paths():
    assert normalize_wsl_path(r"Q:/outputs/run/checkpoints/ckpt_001201") == Path(
        "/mnt/q/outputs/run/checkpoints/ckpt_001201"
    )
