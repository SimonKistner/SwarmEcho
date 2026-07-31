"""Evaluation CSV and heatmap artifact generation.

This module consumes completed evaluation results.  It deliberately owns no
model construction, checkpoint loading, or simulation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from swarmecho.env.maps import MapDefinition
from swarmecho.training.artifacts import (
    artifact_suffix,
    load_eval_info_csv,
    save_eval_info_csv,
    write_manifest,
)
from swarmecho.training.evaluation import ParallelEvaluationResult
from swarmecho.visualize.render_preview import _resolve_map_path, render_png


SCALE = 8.0
SHOW_SPAWN_ZONES = False
FAILED_CHAIN_HEATMAP_ALPHA = 0.9
FAILED_CHAIN_HEATMAP_DOT_RADIUS = 2
HEATMAP_ALPHA = 1.0
HEATMAP_DOT_RADIUS = 2


def load_map_data(cfg):
    """Loads map definition and blueprint configs."""
    map_names = cfg.env.get("map_names", [])
    if not map_names:
        print("ERROR: No map specified in configuration.")
        sys.exit(1)
    map_name = map_names[0]

    map_path = _resolve_map_path(map_name)
    if not map_path.exists():
        print(f"ERROR: Map definition not found at {map_path}")
        sys.exit(1)

    with open(map_path, "r") as f:
        map_data = yaml.safe_load(f)

    try:
        map_def = MapDefinition.load(map_path, cell_size=1.0)
    except Exception as e:
        print(f"Warning: Could not load MapDefinition: {e}")
        map_def = None

    return map_name, map_data, map_def


def render_and_save_failed_chain_heatmap(
    failed_positions,
    map_data,
    map_def,
    success_rate,
    num_fail,
    run_dir,
    video_dir,
    run_timestamp,
    save_png=True,
    total_episodes=4096,
    manifest_dir=None,
    artifact_stem=None,
    source_csv=None,
):
    """
    Render the failed-chain heatmap from in-memory evaluation results.

    The heatmap is generated with a 60px white border at the top displaying the 
    sliding window size/total episodes, failed chain counts, success rate, and legend.
    """
    artifact_stem = artifact_stem or f"{run_timestamp}_failed_chain"
    heatmap_path = Path(video_dir) / f"{artifact_stem}.png"

    # Generate Heatmap image
    if save_png:
        background_img = render_png(
            data=map_data,
            map_def=map_def,
            state=None,
            show_zones=SHOW_SPAWN_ZONES,
            show_spawns=False,
            scale=SCALE,
        )

        overlay = background_img.copy()
        height = float(map_data["height"])

        for pos in failed_positions:
            px = int(pos[0] * SCALE)
            py = int((height - pos[1]) * SCALE)
            cv2.circle(overlay, (px, py), FAILED_CHAIN_HEATMAP_DOT_RADIUS, (68, 68, 239), -1, cv2.LINE_AA)

        heatmap_img = cv2.addWeighted(overlay, FAILED_CHAIN_HEATMAP_ALPHA, background_img, 1.0 - FAILED_CHAIN_HEATMAP_ALPHA, 0)
        
        # Add 60px top padding for title and legend
        padded_img = cv2.copyMakeBorder(heatmap_img, 60, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        
        rate_str = f"{success_rate:.1f}%" if success_rate is not None else "?"
        info_str = f"Run: {run_dir.name} | Failed Chain: {num_fail}/{total_episodes} | Success Rate: {rate_str}"
        cv2.putText(padded_img, info_str, (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.42, (55, 41, 31), 1, cv2.LINE_AA)
        
        # Draw Legend
        cv2.circle(padded_img, (15, 46), 4, (68, 68, 239), -1, cv2.LINE_AA)
        cv2.putText(padded_img, "Failed Chain Target", (25, 50), cv2.FONT_HERSHEY_DUPLEX, 0.38, (55, 41, 31), 1, cv2.LINE_AA)

        Path(video_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(heatmap_path), padded_img)
        print(f"Saved failed chain targets heatmap to: {heatmap_path.name}")
    if manifest_dir is not None:
        write_manifest(Path(manifest_dir) / f"{artifact_stem}.heatmap.json", {
            "type": "heatmap",
            "kind": "failed_chain",
            "image_path": str(heatmap_path),
            "total_episodes": int(total_episodes),
            "num_points": int(num_fail),
            "success_rate": None if success_rate is None else float(success_rate),
            "data_path": None if source_csv is None else str(source_csv),
        })

    return heatmap_path

def render_and_save_not_found_heatmap(not_found_positions, map_data, map_def, found_rate, num_not_found, run_dir, video_dir, run_timestamp, filename_prefix, label, total_episodes=4096, manifest_dir=None, artifact_stem=None, source_csv=None):
    """
    Generates a secondary heatmap plotting target coordinates that were not found/delivered.
    Uses blue dots on the map, with a 60px top padding containing title stats and a color legend.
    """
    artifact_stem = artifact_stem or f"{run_timestamp}_{filename_prefix}"
    heatmap_path = Path(video_dir) / f"{artifact_stem}.png"

    # Generate heatmap background
    background_img = render_png(
        data=map_data,
        map_def=map_def,
        state=None,
        show_zones=SHOW_SPAWN_ZONES,
        show_spawns=False,
        scale=SCALE,
    )

    overlay = background_img.copy()
    height = float(map_data["height"])

    for pos in not_found_positions:
        px = int(pos[0] * SCALE)
        py = int((height - pos[1]) * SCALE)
        cv2.circle(
            overlay,
            (px, py),
            HEATMAP_DOT_RADIUS,
            (235, 99, 37),
            -1,
            cv2.LINE_AA,
        )

    heatmap_img = cv2.addWeighted(
        overlay,
        HEATMAP_ALPHA,
        background_img,
        1.0 - HEATMAP_ALPHA,
        0,
    )
    padded_img = cv2.copyMakeBorder(
        heatmap_img,
        60,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=[255, 255, 255],
    )
    info_str = (
        f"Run: {run_dir.name} | {label}: "
        f"{num_not_found}/{total_episodes} | Rate: {found_rate:.1f}%"
    )
    cv2.putText(
        padded_img,
        info_str,
        (10, 25),
        cv2.FONT_HERSHEY_DUPLEX,
        0.42,
        (55, 41, 31),
        1,
        cv2.LINE_AA,
    )
    cv2.circle(
        padded_img,
        (15, 46),
        4,
        (235, 99, 37),
        -1,
        cv2.LINE_AA,
    )
    cv2.putText(
        padded_img,
        label,
        (25, 50),
        cv2.FONT_HERSHEY_DUPLEX,
        0.38,
        (55, 41, 31),
        1,
        cv2.LINE_AA,
    )

    Path(video_dir).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(heatmap_path), padded_img)
    print(f"Saved {label.lower()} heatmap to: {heatmap_path.name}")
    if manifest_dir is not None:
        write_manifest(
            Path(manifest_dir) / f"{artifact_stem}.heatmap.json",
            {
                "type": "heatmap",
                "kind": filename_prefix,
                "label": label,
                "image_path": str(heatmap_path),
                "data_path": None if source_csv is None else str(source_csv),
                "total_episodes": int(total_episodes),
                "num_points": int(num_not_found),
                "rate": float(found_rate),
            },
        )


def render_and_save_found_and_delivered_heatmap(
    visually_found_not_delivered_positions,
    not_visually_found_positions,
    map_data,
    map_def,
    delivered_rate,
    visually_found_rate,
    num_not_delivered,
    num_not_visually_found,
    run_dir,
    video_dir,
    run_timestamp,
    total_episodes=4096,
    manifest_dir=None,
    artifact_stem=None,
    source_csv=None,
):
    """Render combined not-found and not-delivered target positions."""
    artifact_stem = artifact_stem or f"{run_timestamp}_found_and_delivered"
    heatmap_path = Path(video_dir) / f"{artifact_stem}.png"

    if num_not_visually_found > num_not_delivered:
        print(
            "WARNING: num_not_visually_found "
            f"({num_not_visually_found}) is greater than "
            f"num_not_delivered ({num_not_delivered})."
        )

    background_img = render_png(
        data=map_data,
        map_def=map_def,
        state=None,
        show_zones=SHOW_SPAWN_ZONES,
        show_spawns=False,
        scale=SCALE,
    )
    overlay = background_img.copy()
    height = float(map_data["height"])
    for pos in visually_found_not_delivered_positions:
        px = int(pos[0] * SCALE)
        py = int((height - pos[1]) * SCALE)
        cv2.circle(
            overlay, (px, py), HEATMAP_DOT_RADIUS, (6, 119, 217), -1,
            cv2.LINE_AA,
        )

    for pos in not_visually_found_positions:
        px = int(pos[0] * SCALE)
        py = int((height - pos[1]) * SCALE)
        cv2.circle(
            overlay, (px, py), HEATMAP_DOT_RADIUS, (235, 99, 37), -1,
            cv2.LINE_AA,
        )

    heatmap_img = cv2.addWeighted(overlay, HEATMAP_ALPHA, background_img, 1.0 - HEATMAP_ALPHA, 0)
    
    # Add 60px top padding for title and legend
    padded_img = cv2.copyMakeBorder(heatmap_img, 60, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    # Title stats
    info_str = f"Run: {run_dir.name} | Not Delivered: {num_not_delivered}/{total_episodes} | Not Visually Found: {num_not_visually_found}/{total_episodes}"
    cv2.putText(padded_img, info_str, (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.40, (55, 41, 31), 1, cv2.LINE_AA)
    
    # Dual color legend items
    # 1. Visually Found, Not Delivered (Orange)
    cv2.circle(padded_img, (15, 46), 4, (6, 119, 217), -1, cv2.LINE_AA)
    cv2.putText(padded_img, "Visually Found, Not Delivered", (25, 50), cv2.FONT_HERSHEY_DUPLEX, 0.38, (55, 41, 31), 1, cv2.LINE_AA)
    
    # 2. Not Visually Found (Blue)
    cv2.circle(padded_img, (260, 46), 4, (235, 99, 37), -1, cv2.LINE_AA)
    cv2.putText(padded_img, "Not Visually Found", (270, 50), cv2.FONT_HERSHEY_DUPLEX, 0.38, (55, 41, 31), 1, cv2.LINE_AA)

    Path(video_dir).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(heatmap_path), padded_img)
    print(f"Saved found-and-delivered heatmap to: {heatmap_path.name}")
    if manifest_dir is not None:
        write_manifest(Path(manifest_dir) / f"{artifact_stem}.heatmap.json", {
            "type": "heatmap",
            "kind": "found_and_delivered",
            "image_path": str(heatmap_path),
            "data_path": None if source_csv is None else str(source_csv),
            "total_episodes": int(total_episodes),
            "num_not_delivered": int(num_not_delivered),
            "num_not_visually_found": int(num_not_visually_found),
            "delivered_rate": float(delivered_rate),
            "visually_found_rate": float(visually_found_rate),
        })


def write_training_evaluation_artifacts(
    result: ParallelEvaluationResult,
    cfg,
    *,
    run_dir: Path,
    data_dir: Path,
    manifest_dir: Path,
    artifact_root: Path,
    update: int,
    steps_done: int,
) -> None:
    """Write canonical evaluation data and any configured heatmaps."""
    generate_any_heatmap = bool(
        cfg.evaluation.get("eval_failed_chain_heatmap", False)
        or cfg.evaluation.get(
            "eval_not_delivered_or_visually_found_heatmap", False
        )
    )

    target_positions = np.asarray(result.final_state.physics.target_pos)
    base_positions = np.asarray(result.final_state.physics.base_pos)
    successes = np.asarray(result.final_successes, dtype=bool)
    delivered = np.asarray(result.final_delivered, dtype=bool)
    visually_found = np.asarray(result.final_visually_found, dtype=bool)
    total_episodes = len(successes)
    suffix = artifact_suffix(update, steps_done)

    try:
        info_path = save_eval_info_csv(
            data_dir / f"eval_info_{suffix}.csv",
            target_positions=target_positions,
            base_positions=base_positions,
            successes=successes,
            delivered=delivered,
            visually_found=visually_found,
        )
        print(
            f"  [eval-csv] Saved {total_episodes} episodes to: "
            f"{info_path.name}"
        )
    except Exception as exc:
        print(
            "  [eval-csv-error] Failed to save evaluation information: "
            f"{exc}"
        )
        return

    if not generate_any_heatmap:
        return

    try:
        records = load_eval_info_csv(info_path)
        target_positions = records["positions"]
        stages = records["stages"]
        successes = stages == "chain_success"
        delivered = np.isin(
            stages,
            ("found_and_delivered", "chain_success"),
        )
        visually_found = stages != "not_found"
        total_episodes = len(stages)
        _, map_data, map_def = load_map_data(cfg)
        chain_dir = artifact_root / "chain_heatmaps"
        found_dir = artifact_root / "found_heatmaps"

        if cfg.evaluation.get("eval_failed_chain_heatmap", False):
            failed_positions = target_positions[~successes]
            success_rate = float(np.mean(successes) * 100.0)
            render_and_save_failed_chain_heatmap(
                failed_positions=failed_positions,
                map_data=map_data,
                map_def=map_def,
                success_rate=success_rate,
                num_fail=len(failed_positions),
                run_dir=run_dir,
                video_dir=chain_dir,
                run_timestamp=suffix,
                save_png=True,
                total_episodes=total_episodes,
                manifest_dir=manifest_dir,
                artifact_stem=f"failed_chain_{suffix}",
                source_csv=info_path,
            )

        if not cfg.evaluation.get(
            "eval_not_delivered_or_visually_found_heatmap", False
        ):
            return

        split_in_two = bool(
            cfg.evaluation.get(
                "eval_not_deliv_not_visual_splitt_in_two", False
            )
        )
        not_delivered = target_positions[~delivered]
        delivered_rate = float(np.mean(delivered) * 100.0)
        not_visually_found = target_positions[~visually_found]
        visually_found_rate = float(np.mean(visually_found) * 100.0)

        if not split_in_two:
            visually_found_not_delivered = target_positions[
                (~delivered) & visually_found
            ]
            render_and_save_found_and_delivered_heatmap(
                visually_found_not_delivered_positions=(
                    visually_found_not_delivered
                ),
                not_visually_found_positions=not_visually_found,
                map_data=map_data,
                map_def=map_def,
                delivered_rate=delivered_rate,
                visually_found_rate=visually_found_rate,
                num_not_delivered=len(not_delivered),
                num_not_visually_found=len(not_visually_found),
                run_dir=run_dir,
                video_dir=found_dir,
                run_timestamp=suffix,
                total_episodes=total_episodes,
                manifest_dir=manifest_dir,
                artifact_stem=f"found_and_delivered_{suffix}",
                source_csv=info_path,
            )
            return

        render_and_save_not_found_heatmap(
            not_found_positions=not_delivered,
            map_data=map_data,
            map_def=map_def,
            found_rate=delivered_rate,
            num_not_found=len(not_delivered),
            run_dir=run_dir,
            video_dir=found_dir,
            run_timestamp=suffix,
            filename_prefix="delivered",
            label="Not Delivered",
            total_episodes=total_episodes,
            manifest_dir=manifest_dir,
            artifact_stem=f"delivered_{suffix}",
            source_csv=info_path,
        )
        render_and_save_not_found_heatmap(
            not_found_positions=not_visually_found,
            map_data=map_data,
            map_def=map_def,
            found_rate=visually_found_rate,
            num_not_found=len(not_visually_found),
            run_dir=run_dir,
            video_dir=found_dir,
            run_timestamp=suffix,
            filename_prefix="found",
            label="Not Visually Found",
            total_episodes=total_episodes,
            manifest_dir=manifest_dir,
            artifact_stem=f"found_{suffix}",
            source_csv=info_path,
        )
    except Exception as exc:
        print(
            "  [heatmap-error] Failed to render evaluation heatmaps: "
            f"{exc}"
        )
