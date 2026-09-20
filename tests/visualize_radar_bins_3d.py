"""Render the actual 3D radar-bin partition for 8, 16, and 32 bins.

Run from the repository root, for example:

    python tests/visualize_radar_bins_3d.py --output radar_bins_3d.png

The bin centres and assignment rule are imported from ``environment`` so this
visualization stays aligned with the environment implementation:

    bin_index = argmax(unit_relative_direction @ directions.T)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb


_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from swarmecho.env.environment import spherical_directions


def _sphere_grid(
    *, longitude_samples: int = 361, latitude_samples: int = 181
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a unit-sphere grid and its flattened unit direction vectors."""
    longitude = np.linspace(-np.pi, np.pi, longitude_samples)
    latitude = np.linspace(-np.pi / 2, np.pi / 2, latitude_samples)
    longitude_grid, latitude_grid = np.meshgrid(longitude, latitude)
    x = np.cos(latitude_grid) * np.cos(longitude_grid)
    y = np.cos(latitude_grid) * np.sin(longitude_grid)
    z = np.sin(latitude_grid)
    points = np.stack([x, y, z], axis=-1)
    return x, y, z, points.reshape(-1, 3)


def _bin_colours(bin_count: int) -> np.ndarray:
    """Return visually distinct, repeatable colours for the bin labels."""
    hues = np.arange(bin_count, dtype=np.float32) / bin_count
    hsv = np.stack(
        [hues, np.full(bin_count, 0.78), np.full(bin_count, 0.92)], axis=-1
    )
    return hsv_to_rgb(hsv)


def render(output: Path, bin_counts: tuple[int, ...] = (8, 16, 32)) -> None:
    x, y, z, points = _sphere_grid()
    figure = plt.figure(figsize=(16, 5.5), constrained_layout=True)

    for subplot_index, bin_count in enumerate(bin_counts, start=1):
        # These are the exact directions used by make_env_fns().
        directions = np.asarray(spherical_directions(bin_count), dtype=np.float32)

        # This is the exact bin assignment used by the 3D radar observation:
        # rel_norm @ directions.T followed by argmax over the bin dimension.
        assigned_bins = np.argmax(points @ directions.T, axis=-1)
        colours = _bin_colours(bin_count)[assigned_bins].reshape(*x.shape, 3)

        axis = figure.add_subplot(1, len(bin_counts), subplot_index, projection="3d")
        axis.plot_surface(
            x,
            y,
            z,
            facecolors=colours,
            rstride=2,
            cstride=2,
            linewidth=0,
            antialiased=False,
            shade=False,
        )
        axis.quiver(
            np.zeros(bin_count),
            np.zeros(bin_count),
            np.zeros(bin_count),
            directions[:, 0],
            directions[:, 1],
            directions[:, 2],
            color="black",
            length=1.12,
            arrow_length_ratio=0.06,
            linewidth=0.45 if bin_count >= 32 else 0.7,
        )
        axis.scatter(
            directions[:, 0],
            directions[:, 1],
            directions[:, 2],
            color="black",
            s=7 if bin_count >= 32 else 14,
            depthshade=False,
        )
        axis.set_title(f"{bin_count} bins")
        axis.set_box_aspect((1, 1, 1))
        axis.set_xlim(-1.2, 1.2)
        axis.set_ylim(-1.2, 1.2)
        axis.set_zlim(-1.2, 1.2)
        axis.set_xlabel("x", labelpad=-8)
        axis.set_ylabel("y", labelpad=-8)
        axis.set_zlabel("z", labelpad=-8)
        axis.view_init(elev=22, azim=-55)

    figure.suptitle(
        "3D radar bins: coloured Voronoi regions from argmax(direction · bin centre)\n"
        "black arrows/dots = the exact spherical bin centres",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("radar_bins_3d.png"),
        help="PNG output path (default: radar_bins_3d.png)",
    )
    args = parser.parse_args()
    render(args.output)


if __name__ == "__main__":
    main()
