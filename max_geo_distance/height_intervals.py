"""Analytic safe-height intervals for a connector at one fixed XY position."""

import numpy as np
from placement_checks import SAFETY_MARGIN_M, range_limit


SEGMENT_EPSILON = 1e-5
HEIGHT_MARGIN = SAFETY_MARGIN_M


def union_intervals(intervals):
    result = []
    for low, high in sorted(intervals):
        if low > high:
            continue
        if result and low <= result[-1][1]:
            result[-1] = (result[-1][0], max(high, result[-1][1]))
        else:
            result.append((float(low), float(high)))
    return result


def remove_intervals(allowed, forbidden):
    blocked = union_intervals((low - HEIGHT_MARGIN, high + HEIGHT_MARGIN)
                              for low, high in forbidden)
    for block_low, block_high in blocked:
        remaining = []
        for low, high in allowed:
            if block_high < low or block_low > high:
                remaining.append((low, high))
                continue
            if low < block_low:
                remaining.append((low, float(np.nextafter(block_low, -np.inf))))
            if high > block_high:
                remaining.append((float(np.nextafter(block_high, np.inf)), high))
        allowed = remaining
        if not allowed:
            break
    return allowed


def visibility_forbidden_heights(observer, xy, solid_lo, solid_hi):
    """Return heights causing a segment/box hit for this fixed horizontal ray."""
    if not len(solid_lo):
        return []
    direction = xy - observer[:2]
    parallel = np.abs(direction) <= SEGMENT_EPSILON
    safe = np.where(parallel, 1.0, direction)
    first = (solid_lo[:, :2] - observer[:2]) / safe
    last = (solid_hi[:, :2] - observer[:2]) / safe
    near = np.max(np.where(parallel, -np.inf, np.minimum(first, last)), axis=1)
    far = np.min(np.where(parallel, np.inf, np.maximum(first, last)), axis=1)
    outside = np.any(parallel & ((observer[:2] < solid_lo[:, :2]) |
                                 (observer[:2] > solid_hi[:, :2])), axis=1)
    enter = np.maximum(near, SEGMENT_EPSILON)
    leave = np.minimum(far, 1 - SEGMENT_EPSILON)
    active = (leave >= enter) & ~outside
    if not np.any(active):
        return []
    enter, leave = enter[active], leave[active]
    lower_delta = solid_lo[active, 2] - observer[2]
    upper_delta = solid_hi[active, 2] - observer[2]
    below = observer[2] + np.minimum(lower_delta / enter, lower_delta / leave)
    above = observer[2] + np.maximum(upper_delta / enter, upper_delta / leave)
    return list(zip(below, above))


def allowed_connector_heights(xy, window, prepared, left_is_base, right_is_target):
    a, b, c, d = window
    if np.any(xy < prepared.lower[:2]) or np.any(xy > prepared.upper[:2]):
        return [], "intersection outside drone bounds"
    low, high = float(prepared.lower[2]), float(prepared.upper[2])
    cfg = prepared.cfg
    endpoints = ((a, cfg.comm_radius_base if left_is_base else cfg.comm_radius, "left"),
                 (d, cfg.visual_radius if right_is_target else cfg.comm_radius, "right"))
    for observer, radius, side in endpoints:
        radius = range_limit(radius, construction=True)
        horizontal_squared = float(np.sum((xy - observer[:2]) ** 2))
        if horizontal_squared > radius * radius:
            return [], f"{side} horizontal distance exceeds range"
        extent = np.sqrt(max(0.0, radius * radius - horizontal_squared))
        low = max(low, float(observer[2] - extent + 1e-8))
        high = min(high, float(observer[2] + extent - 1e-8))
    if low > high:
        return [], "endpoint ranges have no common height"
    allowed = [(low, high)]

    inside_xy = np.all((xy >= prepared.solid_lo[:, :2]) &
                       (xy <= prepared.solid_hi[:, :2]), axis=1)
    allowed = remove_intervals(allowed, zip(prepared.solid_lo[inside_xy, 2],
                                           prepared.solid_hi[inside_xy, 2]))
    if not allowed:
        return [], "no height with point clearance"
    for observer, _, side in endpoints:
        forbidden = visibility_forbidden_heights(observer, xy,
                                                 prepared.physical_solid_lo - SAFETY_MARGIN_M,
                                                 prepared.physical_solid_hi + SAFETY_MARGIN_M)
        allowed = remove_intervals(allowed, forbidden)
        if not allowed:
            return [], f"no height with {side} visibility"

    # A replacement must remain reachable from at least one removed position.
    reachable_b = remove_intervals(allowed, visibility_forbidden_heights(
        b, xy, prepared.solid_lo, prepared.solid_hi))
    reachable_c = remove_intervals(allowed, visibility_forbidden_heights(
        c, xy, prepared.solid_lo, prepared.solid_hi))
    allowed = union_intervals(reachable_b + reachable_c)
    if not allowed:
        return [], "no height with local planning-clear connection"
    return allowed, "height intervals available"


def nearest_height(preferred, intervals):
    """Project once onto the interval set; equal-distance ties use the lower z."""
    return min((min(max(preferred, low), high) for low, high in intervals),
               key=lambda height: (abs(height - preferred), height))
