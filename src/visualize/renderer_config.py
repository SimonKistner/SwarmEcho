"""
swarmecho/visualize/renderer_config.py
======================================
Centralized visual and layout configuration for the OpenCV renderer.

All dimensions are in logical pixels (rendered at target DPI)
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
    BASE_MARKER_SIZE: int = 10
    TARGET_MARKER_SIZE: int = 14
    DRONE_MARKER_SIZE: int = 6
    COLLISION_GLOW_RADIUS: int = 18    # Glow size around colliding drones

    # --- Static Font Scales / Sizes ---
    CV2_FONT_SCALE_TITLE: float = 0.55
    CV2_FONT_SCALE_LEGEND: float = 0.45
    CV2_FONT_SCALE_LABELS: float = 0.40
    CV2_FONT_SCALE_AXES: float = 0.35
