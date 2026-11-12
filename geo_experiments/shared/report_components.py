"""Shared report palette, legends and error histograms."""
from __future__ import annotations
import html
import numpy as np

# One palette throughout tables, learning curves, comparisons and matrices.
BACKGROUND, PANEL, TEXT, MUTED, GRID = "#0b1220", "#111c2e", "#e4edf7", "#9aacc3", "#2b3b53"


def _legend(items):
    return '<div class="legend">' + "".join(
        f'<span><i style="background:{color}"></i>{html.escape(label)}</span>' for label, color in items
    ) + "</div>"


def _histogram(errors, bins, color, unit, measure="count", count_limit=1):
    if errors is None:
        return '<span class="muted">No evaluation samples</span>'
    if not len(errors):
        return '<span class="perfect">All predictions correct</span>'
    width, left, top, bottom = 430, 48, 12, 92
    limit = 100 if measure == "share" else count_limit
    step = (width - left - 10) / len(bins)
    bars = []
    for i, value in enumerate(bins):
        count = int((errors == value).sum())
        percentage = 100 * count / len(errors)
        height = (bottom - top) * (percentage if measure == "share" else count) / limit
        x = left + i * step
        label = "Wrong map" if value == -1 else str(int(value)) if np.isfinite(value) else "No route"
        bars.append(f'<g><title>{label}{"" if value == -1 else " " + unit + " off"}: {count:,} wrong predictions '
                    f'({percentage:.1f}% of errors, pooled across seeds)</title>'
                    f'<rect x="{x + step * .15:.2f}" y="{bottom - height:.2f}" '
                    f'width="{step * .7:.2f}" height="{height:.2f}" rx="2" fill="{color}"/>'
                    f'<text x="{x + step / 2:.2f}" y="106" text-anchor="middle">{label}</text></g>')
    lines = ""
    for fraction in (0, .5, 1):
        y = bottom - fraction * (bottom - top)
        tick = f"{fraction * limit:.0f}%" if measure == "share" else f"{fraction * limit:,.0f}"
        lines += (f'<line x1="{left}" x2="{width - 10}" y1="{y}" y2="{y}" stroke="{GRID}"/>'
                  f'<text x="{left - 5}" y="{y + 3}" text-anchor="end">{tick}</text>')
    axis_label = "% of incorrect predictions" if measure == "share" else "Incorrect prediction count"
    return (f'<svg class="histogram" viewBox="0 0 {width} 127" role="img" '
            f'aria-label="Distribution of {unit} off among incorrect predictions">'
            f'<g fill="{MUTED}" font-family="system-ui,sans-serif" font-size="10">'
            + lines + "".join(bars) + f'<text x="{(left + width) / 2}" y="122" text-anchor="middle">'
            f'{unit} off · {axis_label}</text></g></svg>')


