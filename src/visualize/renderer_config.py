"""
swarmecho/visualize/renderer_config.py
======================================
Centralized visual and layout configuration for both the Fast OpenCV (CV2)
and Premium Matplotlib (MPL) renderers.

All dimensions are in logical pixels/points (rendered at target DPI)
and are completely independent of the physical world bounds (meters).
"""

class RendererConfig:
    # --- Target DPI & Strides ---
    RENDER_DPI: int = 150
    FRAME_STRIDE: int = 1

    # --- Designated Map Bounding Box (Logical Pixels) ---
    # The physical map (regardless of width/height in meters) is fitted inside
    # this bounding box while strictly preserving its aspect ratio.
    MAP_DISPLAY_WIDTH: int = 800
    MAP_DISPLAY_HEIGHT: int = 800

    # --- Margins & Spacings (Logical Pixels) ---
    MARGIN_LEFT: int = 80
    MARGIN_RIGHT: int = 240   # Wide margin to prevent legend clipping
    MARGIN_TOP: int = 80
    MARGIN_BOTTOM: int = 60   # Basic bottom padding (extra space is added for rewards)
    
    # Gap sizes between visual components
    GAP_TITLE_MAP: int = 30
    GAP_MAP_REWARDS: int = 65
    GAP_BETWEEN_REWARDS: int = 50

    # --- Plot Heights for Reinforcement Learning Rewards ---
    TEAM_REWARD_HEIGHT: int = 120
    INDIV_REWARD_HEIGHT: int = 120

    # --- Static Visual Entity/Marker Sizes ---
    # (Independent of the map dimensions in meters)
    BASE_MARKER_SIZE: int = 10        # CV2 half-width (px), MPL scatter size `s` is proportional
    TARGET_MARKER_SIZE: int = 14       # CV2 star size, MPL marker size
    DRONE_MARKER_SIZE: int = 6         # CV2 circle radius (px), MPL scatter `s` is proportional
    COLLISION_GLOW_RADIUS: int = 18    # Glow size around colliding drones
    
    # MPL specific scatter point sizes (s = Area in points^2)
    MPL_BASE_S: float = 220.0
    MPL_TARGET_S: float = 240.0
    MPL_DRONE_S: float = 70.0
    MPL_GLOW_S: float = 500.0

    # --- Static Font Scales / Sizes ---
    # For CV2: OpenCV FONT_SCALE multiplier
    CV2_FONT_SCALE_TITLE: float = 0.55
    CV2_FONT_SCALE_LEGEND: float = 0.45
    CV2_FONT_SCALE_LABELS: float = 0.40
    CV2_FONT_SCALE_AXES: float = 0.35

    # For Matplotlib: Pt sizes
    MPL_FONT_SIZE_TITLE: float = 11.5
    MPL_FONT_SIZE_LEGEND_TITLE: float = 9.5
    MPL_FONT_SIZE_LEGEND: float = 7.5
    MPL_FONT_SIZE_LABELS: float = 7.5
    MPL_FONT_SIZE_AXES: float = 6.5
