import sys
from pathlib import Path

# Add src directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw
import os

from core.config import MAP_DIR
from env.maps import MapDefinition

st.set_page_config(page_title="SwarmEcho Architectural Builder", layout="wide")

st.title("🏛️ SwarmEcho Architectural Builder")
st.markdown("Design blueprints using Rooms and Hallways.")

# ---------------------------------------------------------------------------
# Sidebar: Project Management & Snap
# ---------------------------------------------------------------------------
map_dir = MAP_DIR
map_dir.mkdir(exist_ok=True)
existing_maps = [f.stem for f in map_dir.glob("*.yaml")]

with st.sidebar:
    st.header("Project")
    action = st.radio("Action", ["Edit Existing", "Create New"])
    
    if action == "Edit Existing":
        selected_map_name = st.selectbox("Select Map", existing_maps)
        if selected_map_name:
            map_path = map_dir / f"{selected_map_name}.yaml"
            m = MapDefinition.load(map_path)
        else:
            st.stop()
    else:
        new_name = st.text_input("New Map Name", "office_delta")
        m = MapDefinition(
            name=new_name,
            width=500.0,
            height=100.0,
            base_spawn_zone=[5.0, 45.0, 15.0, 55.0],
            target_spawn_zone=[480.0, 45.0, 495.0, 55.0],
            drone_spawn_zone=[0.0, 0.0, 50.0, 100.0],
            rooms=[], hallways=[], walls=[]
        )

    st.divider()
    st.header("World Settings")
    m.width = st.number_input("Width (m)", value=float(m.width), step=10.0, format="%.1f")
    m.height = st.number_input("Height (m)", value=float(m.height), step=10.0, format="%.1f")
    
    st.divider()
    snap = st.checkbox("Snap to 1m Grid", value=True)

# ---------------------------------------------------------------------------
# Spawn Zones
# ---------------------------------------------------------------------------
st.header(f"Editing: {m.name}")
with st.expander("📍 Spawn Zones", expanded=False):
    col1, col2, col3 = st.columns(3)
    def zone_sliders(label, current_zone, max_x, max_y):
        x1, y1, x2, y2 = current_zone
        xr = st.slider(f"{label} X", 0.0, max_x, (float(min(x1,x2)),float(max(x1,x2))), step=1.0)
        yr = st.slider(f"{label} Y", 0.0, max_y, (float(min(y1,y2)),float(max(y1,y2))), step=1.0)
        return [xr[0], yr[0], xr[1], yr[1]]
    with col1: m.base_spawn_zone = zone_sliders("Base", m.base_spawn_zone, m.width, m.height)
    with col2: m.target_spawn_zone = zone_sliders("Target", m.target_spawn_zone, m.width, m.height)
    with col3: m.drone_spawn_zone = zone_sliders("Drone", m.drone_spawn_zone, m.width, m.height)

# ---------------------------------------------------------------------------
# Architectural Sections
# ---------------------------------------------------------------------------
st.divider()
tab1, tab2, tab3 = st.tabs(["🏛️ Rooms", "🛣️ Hallways", "✏️ Custom Walls"])

with tab1:
    st.subheader("Manage Rooms")
    room_df = pd.DataFrame(m.rooms) if m.rooms else pd.DataFrame(columns=["x", "y", "w", "h", "door_side"])
    edited_rooms = st.data_editor(room_df, num_rows="dynamic", width='stretch',
                                  column_config={"door_side": st.column_config.SelectboxColumn("Door", options=["N", "S", "E", "W", "none"])})
    m.rooms = edited_rooms.to_dict('records')

with tab2:
    st.subheader("Manage Hallways")
    hall_df = pd.DataFrame(m.hallways) if m.hallways else pd.DataFrame(columns=["x1", "y1", "x2", "y2", "width"])
    edited_halls = st.data_editor(hall_df, num_rows="dynamic", width='stretch')
    m.hallways = edited_halls.to_dict('records')

with tab3:
    st.subheader("Direct Wall Segments")
    wall_df = pd.DataFrame(m.walls, columns=["x1", "y1", "x2", "y2"]) if m.walls else pd.DataFrame(columns=["x1", "y1", "x2", "y2"])
    edited_walls = st.data_editor(wall_df, num_rows="dynamic", width='stretch')
    m.walls = edited_walls.values.tolist()

if snap:
    # Round all primitive values to 1.0 (handling None from new data_editor rows)
    for r in m.rooms: r.update({k: float(round(v)) if v is not None else 0.0 for k,v in r.items() if k != 'door_side'})
    for h in m.hallways: h.update({k: float(round(v)) if v is not None else 0.0 for k,v in h.items()})

# ---------------------------------------------------------------------------
# Render Preview & Save
# ---------------------------------------------------------------------------
st.divider()
st.subheader("🖼️ Blueprint Preview")
m.rasterize(resolution=1.0)
display_grid = np.flipud(m.occupancy_grid.T)
img = Image.fromarray((1.0 - display_grid).astype(np.uint8) * 255).convert("RGB")

def draw_zones(base_img, m_def):
    overlay = Image.new("RGBA", base_img.size, (0,0,0,0))
    d = ImageDraw.Draw(overlay)
    sx, sy = base_img.width / m_def.width, base_img.height / m_def.height
    def to_px(zone): return [zone[0]*sx, (m_def.height-zone[3])*sy, zone[2]*sx, (m_def.height-zone[1])*sy]
    d.rectangle(to_px(m_def.drone_spawn_zone), fill=(100,100,100,60), outline=(50,50,50,150))
    d.rectangle(to_px(m_def.base_spawn_zone), fill=(0,0,255,80), outline=(0,0,150,150))
    d.rectangle(to_px(m_def.target_spawn_zone), fill=(255,0,0,80), outline=(150,0,0,150))
    return Image.alpha_composite(base_img.convert("RGBA"), overlay).convert("RGB")

st.image(draw_zones(img, m), width='stretch')

if st.button("💾 Save Architectural Blueprint", type="primary"):
    yaml_data = {
        "name": m.name, "width": float(m.width), "height": float(m.height),
        "spawn_zones": {"base": m.base_spawn_zone, "target": m.target_spawn_zone, "drone": m.drone_spawn_zone},
        "rooms": m.rooms, "hallways": m.hallways, "walls": m.walls
    }
    with open(map_dir / f"{m.name}.yaml", "w") as f: yaml.dump(yaml_data, f)
    st.success(f"Saved {m.name}!")


