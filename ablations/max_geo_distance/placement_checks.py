"""Host float32 placement checks, following the runtime's range/slab arithmetic."""

import numpy as np


SAFETY_MARGIN_M = 1e-3
PROTOCOL = "legal-drone-start / float32-links / finder-attachment-v2"


def range_limit(radius, construction=False):
    margin = min(SAFETY_MARGIN_M, 0.01 * radius)
    return max(0.0, radius - (2 if construction else 1) * margin)


def rounded(point):
    return np.asarray(point, dtype=np.float32).astype(np.float64)


def visible32(start, ends, solid_lo, solid_hi, margin=SAFETY_MARGIN_M):
    """Mirror obstacles._segments_blocked_block with float32 operands.

    Inflating physical boxes slightly rejects numerically fragile grazing rays.
    This host check does not replace the independent actual-GPU hold validator.
    """
    start, ends = np.asarray(start, np.float32), np.asarray(ends, np.float32)
    lo = np.asarray(solid_lo, np.float32) - np.float32(margin)
    hi = np.asarray(solid_hi, np.float32) + np.float32(margin)
    direction = ends[:, None, :] - start
    epsilon = np.float32(1e-5)
    parallel = np.abs(direction) <= epsilon
    safe = np.where(parallel, np.float32(1), direction)
    first, last = (lo - start) / safe, (hi - start) / safe
    near = np.max(np.where(parallel, np.float32(-np.inf), np.minimum(first, last)), axis=-1)
    far = np.min(np.where(parallel, np.float32(np.inf), np.maximum(first, last)), axis=-1)
    outside = np.any(parallel & ((start < lo) | (start > hi)), axis=-1)
    blocked = np.any((far >= np.maximum(near, epsilon)) &
                     (near <= np.float32(1) - epsilon) & ~outside, axis=-1)
    return ~blocked


def point_valid(point, prepared, planning=True):
    point = np.asarray(point, np.float32)
    if not np.isfinite(point).all():
        return False
    if np.any(point < np.asarray(prepared.lower, np.float32)) or np.any(point > np.asarray(prepared.upper, np.float32)):
        return False
    buffer = np.float32(prepared.cfg.drone_radius + SAFETY_MARGIN_M)
    lo = np.asarray(prepared.physical_solid_lo, np.float32) - buffer
    hi = np.asarray(prepared.physical_solid_hi, np.float32) + buffer
    if np.any(np.all((point >= lo) & (point <= hi), axis=1)):
        return False
    if planning and np.any(np.all((point >= prepared.solid_lo) & (point <= prepared.solid_hi), axis=1)):
        return False
    return True


def link_reason(start, end, radius, prepared):
    delta = np.asarray(end, np.float32) - np.asarray(start, np.float32)
    length = np.sqrt(np.sum(delta * delta, dtype=np.float32))
    if not np.isfinite(length) or length > np.float32(range_limit(radius)):
        return "float32 link exceeds safe range"
    if not visible32(start, np.asarray(end)[None], prepared.physical_solid_lo,
                     prepared.physical_solid_hi)[0]:
        return "float32 link obstructed or grazing"
    return "accepted"


def chain_reason(chain, target, prepared):
    if chain is None or not len(chain):
        return "empty chain"
    previous = prepared.base
    for index, point in enumerate(chain):
        if not point_valid(point, prepared):
            return f"drone {index + 1}: invalid rounded placement"
        radius = prepared.cfg.comm_radius_base if index == 0 else prepared.cfg.comm_radius
        reason = link_reason(previous, point, radius, prepared)
        if reason != "accepted":
            return f"incoming link {index + 1}: {reason}"
        previous = point
    reason = link_reason(previous, target, prepared.cfg.visual_radius, prepared)
    return "accepted" if reason == "accepted" else f"finder tip: {reason}"


def connector_reason(point, window, prepared, left_is_base, right_is_target):
    if not point_valid(point, prepared):
        return "invalid rounded connector placement"
    left = prepared.cfg.comm_radius_base if left_is_base else prepared.cfg.comm_radius
    right = prepared.cfg.visual_radius if right_is_target else prepared.cfg.comm_radius
    for start, end, radius, side in ((window[0], point, left, "left"),
                                     (point, window[3], right, "right")):
        reason = link_reason(start, end, radius, prepared)
        if reason != "accepted":
            return f"{side}: {reason}"
    if not np.any(visible32(point, window[1:3], prepared.solid_lo, prepared.solid_hi, margin=0)):
        return "no rounded planning-clear connection"
    return "accepted"
