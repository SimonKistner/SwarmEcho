import streamlit as st
import os
import re
from pathlib import Path
from omegaconf import OmegaConf
import pandas as pd
import numpy as np
from datetime import datetime

# Set page config for professional matrix layout
st.set_page_config(
    page_title="SwarmEcho Intelligence Operations",
    page_icon="O",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Screenshot-Ready Matrix tables
st.markdown("""
<style>
    /* Typography and Layout */
    .header-style { font-size: 26px; font-weight: 600; color: #FFFFFF; margin-bottom: 2px; }
    .subheader-style { font-size: 13px; font-weight: 500; color: #00ADB5; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 20px;}
    .section-title { font-size: 18px; font-weight: 600; color: #EEEEEE; margin-top: 30px; margin-bottom: 10px; border-bottom: 1px solid #333; padding-bottom: 5px; }
    
    /* Table Styling for Screenshot Readiness */
    .se-table-container { width: 100%; overflow-x: auto; margin-bottom: 25px; border-radius: 6px; border: 1px solid #333; background-color: #121418; }
    .se-table { width: 100%; border-collapse: collapse; font-family: 'Inter', 'Segoe UI', monospace; font-size: 13.5px; }
    .se-table th { background-color: #1A1D24; color: #00ADB5; text-align: left; padding: 12px 15px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 2px solid #222831; }
    .se-table td { padding: 10px 15px; border-bottom: 1px solid #222831; color: #D3D6DA; }
    .se-table tr:hover { background-color: #1A1D24; }
    
    /* Deviation Highlighting */
    .row-diff td { background-color: rgba(255, 60, 60, 0.15) !important; color: #FF8A8A !important; font-weight: 600; }
    
    /* Checkbox Group Styling in Sidebar */
    .sidebar-cgroup { font-size: 14px; font-weight: 600; color: #EEEEEE; margin-top: 20px; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 1px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='header-style'>SwarmEcho Operations Matrix</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader-style'>Cross-Curriculum Telemetry Analysis</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Logic - Discovery Engine  & Naming Cleaners
# ---------------------------------------------------------------------------

def extract_timestamp(name: str):
    clean = name.replace("run", "").strip("-_")
    ts_str = clean[:15]
    try:
        return datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
    except ValueError:
        return None

def clean_semantic_name(raw_name: str) -> str:
    """Strips timestamps and fixes capitalization for clean UI display.
       E.g. curriculum_base_v1_20260418_063439 -> Curriculum Base V1
    """
    # Remove trailing timestamp _YYYYMMDD_HHMMSS
    clean = re.sub(r'_\d{8}_\d{6}$', '', raw_name)
    # Replace underscores and capitalize
    clean = clean.replace("_", " ").title()
    if clean.lower() == "Standalone": return "Direct Operations"
    return clean

def flatten_wandb_config(cfg_dict):
    flat = {}
    for key, val in cfg_dict.items():
        if isinstance(val, dict) and 'value' in val:
            flat[key] = val['value']
        else:
            flat[key] = val
    return flat

def find_all_runs(outputs_root: Path, wandb_root: Path):
    """Timestamp-linked discovery of valid operation footprints."""
    local_runs = []
    
    if outputs_root.exists():
        for p in outputs_root.rglob("run_*"):
            if not p.is_dir(): continue
            v_dir = p / "videos"
            if not v_dir.exists() or not any(v_dir.glob("*.mp4")): continue
            ts = extract_timestamp(p.name)
            if not ts: continue
            
            c_raw = "Standalone"
            if "curriculum" in p.parts:
                try: c_raw = p.parts[p.parts.index("curriculum") + 1]
                except IndexError: pass
            
            local_runs.append({
                "ts": ts,
                "outputs_path": p,
                "curriculum_raw": c_raw,
                "curriculum_clean": clean_semantic_name(c_raw),
                "video_path": v_dir,
                "folder_name": p.name
            })
            
    run_registry = {}
    
    if wandb_root.exists():
        for w_path in wandb_root.glob("run-*"):
            if not w_path.is_dir() or w_path.name == "to_delete": continue
            cfg_path = w_path / "files" / "config.yaml"
            if not cfg_path.exists(): continue
            w_ts = extract_timestamp(w_path.name)
            if not w_ts: continue
            
            match = None
            for local in local_runs:
                if abs((w_ts - local["ts"]).total_seconds()) <= 60:
                    match = local
                    break
                    
            if match:
                try:
                    raw_cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
                    if "_wandb" in raw_cfg: raw_cfg = flatten_wandb_config(raw_cfg)
                except Exception: continue
                
                # Calculate True Timesteps Trained from Checkpoints
                actual_steps = "Unknown"
                ckpt_dir = match["outputs_path"] / "checkpoints"
                if ckpt_dir.exists():
                    ckpts = sorted(list(ckpt_dir.glob("ckpt_*")), key=lambda x: x.name)
                    if ckpts:
                        latest_ckpt = ckpts[-1].name
                        try:
                            # Extract integer from ckpt_000381
                            updates = int(latest_ckpt.split("_")[1])
                            t_cfg = raw_cfg.get("training", {})
                            envs = int(t_cfg.get("num_envs", 1024))
                            rollout_steps = int(t_cfg.get("num_steps", 128))
                            
                            total_steps = updates * envs * rollout_steps
                            
                            if total_steps >= 1_000_000:
                                actual_steps = f"{total_steps / 1_000_000:.1f}M"
                            else:
                                actual_steps = f"{total_steps:,}"
                        except Exception: pass
                
                # Inject as a pseudo-parameter so it bubbles up in the UI matrix
                if "telemetry" not in raw_cfg: raw_cfg["telemetry"] = {}
                raw_cfg["telemetry"]["actual_steps_trained"] = actual_steps
                
                reg_key = f"{match['curriculum_clean']}::{match['folder_name']}"
                run_registry[reg_key] = {
                    "id": reg_key,
                    "config": raw_cfg,
                    "outputs_path": match["outputs_path"],
                    "wandb_path": w_path,
                    "curriculum_clean": match["curriculum_clean"],
                    "video_path": match["video_path"],
                    "created": match["ts"].timestamp(),
                    "folder_name": match["folder_name"]
                }
                
    # Sequence mapping (Level_X)
    sorted_reg = dict(sorted(run_registry.items(), key=lambda x: x[1]["created"]))
    c_counts = {}
    final_registry = {}
    for r_key, r_data in sorted_reg.items():
        c_name = r_data["curriculum_clean"]
        lvl = c_counts.get(c_name, 0)
        display_name = f"{c_name} | L{lvl}"
        r_data["display_name"] = display_name
        r_data["level_label"] = f"L{lvl}"
        final_registry[display_name] = r_data
        c_counts[c_name] = lvl + 1
                    
    return final_registry

# ---------------------------------------------------------------------------
# State & Data Loading
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUTS_ROOT = BASE_DIR / "outputs" / "curriculum"
WANDB_ROOT = BASE_DIR / "wandb"

registry = find_all_runs(OUTPUTS_ROOT, WANDB_ROOT)

if not registry:
    st.warning("Intelligence Matrix Offline.")
    st.info("No verified operations located. Require WandB parameter logs and local mp4 telemetry footprint.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar - Checkbox Matrix
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("<div class='header-style' style='font-size: 20px;'>Matrix Selection</div>", unsafe_allow_html=True)
    st.caption("Select stages across domains for variance testing.")
    
    grouped_runs = {}
    for r_disp, r_data in registry.items():
        c_name = r_data["curriculum_clean"]
        if c_name not in grouped_runs: grouped_runs[c_name] = []
        grouped_runs[c_name].append((r_disp, r_data["level_label"]))
        
    selected_operations = []
    
    for c_name, runs in grouped_runs.items():
        st.markdown(f"<div class='sidebar-cgroup'>{c_name}</div>", unsafe_allow_html=True)
        # Display checkboxes smoothly
        for r_disp, lvl_label in runs:
            # Default to selecting the first L0 just to show something on load
            is_def = (selected_operations == []) and (lvl_label == "L0")
            if st.checkbox(lvl_label, value=is_def, key=r_disp):
                selected_operations.append(r_disp)

# ---------------------------------------------------------------------------
# Parameter Matrix Rendering
# ---------------------------------------------------------------------------

if not selected_operations:
    st.info("Engage at least one operation stage from the sidebar matrix.")
    st.stop()

# ---------------------------------------------------------------------------
# Parameter Matrix Rendering
# ---------------------------------------------------------------------------

if not selected_operations:
    st.info("Engage at least one operation stage from the sidebar matrix.")
    st.stop()

# Build DataFrame
comp_data = {}
for r_disp in selected_operations:
    cfg = registry[r_disp]["config"]
    flat_cfg = {}
    for section, params in cfg.items():
        # Exclude massive internal WandB system blobs
        if str(section).lower() in ["_wandb", "wandb"]: continue
        
        if isinstance(params, dict):
            for k, v in params.items():
                flat_cfg[f"{section.upper()} | {k}"] = str(v)
        else:
            flat_cfg[f"GLOBAL | {section}"] = str(params)
    comp_data[r_disp] = flat_cfg

diff_df = pd.DataFrame(comp_data)
# Sort index so Domain groups stick together
diff_df.sort_index(inplace=True)

def render_html_table(df, title, show_domain=True):
    if df.empty: return
    
    html = f"<div class='section-title'>{title}</div>"
    html += "<div class='se-table-container'><table class='se-table'>"
    
    # Headers
    html += "<thead><tr>"
    if show_domain: html += "<th>Domain</th>"
    html += "<th>Parameter</th>"
    for col in df.columns: html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>"
    
    for row_idx, row in df.iterrows():
        domain, key = row_idx.split(" | ", 1)
        html += "<tr>"
        if show_domain: html += f"<td style='color:#00ADB5;'><i>{domain}</i></td>"
        html += f"<td><b>{key}</b></td>"
        for val in row:
            html += f"<td>{val}</td>"
        html += "</tr>"
        
    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)


if len(selected_operations) == 1:
    # ── Single View (Operation Report Card) ────────────────
    st.markdown("---")
    st.markdown("<div class='header-style'>Core Intelligence Profile</div>", unsafe_allow_html=True)
    st.caption("Key operational parameters for the selected stage.")
    
    # Extract specific core keys
    r_disp = selected_operations[0]
    cfg = registry[r_disp]["config"]
    env_cfg = cfg.get("env", {})
    rew_cfg = cfg.get("reward", {})
    tel_cfg = cfg.get("telemetry", {})
    
    def render_vertical_card(title, items):
        html = f"<div style='font-size: 13px; font-weight: 600; color: #00ADB5; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; margin-top: 20px;'>{title}</div>"
        html += "<div style='border: 1px solid #333; border-radius: 6px; background-color: #121418; width: 380px;'>"
        for i, (k, v) in enumerate(items):
            bb = "" if i == len(items) - 1 else "border-bottom: 1px solid #222831;"
            html += f"<div style='display: flex; padding: 10px 15px; {bb}'>"
            html += f"<div style='color: #EEEEEE; font-weight: 600; font-size: 13.5px; flex: 1;'>{k}</div>"
            html += f"<div style='color: #D3D6DA; font-size: 13.5px; text-align: left; width: 120px;'>{v}</div>"
            html += "</div>"
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)
        
    env_items = [
        ("map_names", str(env_cfg.get("map_names", "N/A"))),
        ("max_steps", str(env_cfg.get("max_steps", "N/A"))),
        ("num_agents", str(env_cfg.get("num_agents", "N/A"))),
        ("num_bases", str(env_cfg.get("num_bases", "0"))),
        ("num_targets", str(env_cfg.get("num_targets", "0"))),
        ("actual_steps_trained", str(tel_cfg.get("actual_steps_trained", "Unknown")))
    ]
    
    rew_items = [
        ("collision_penalty", str(rew_cfg.get("collision_penalty", "N/A"))),
        ("exploration_bonus", str(rew_cfg.get("exploration_bonus", "N/A"))),
        ("max_gap_penalty", str(rew_cfg.get("max_gap_penalty", "N/A"))),
        ("success_bonus", str(rew_cfg.get("success_bonus", "0")))
    ]
    
    render_vertical_card("Env", env_items)
    render_vertical_card("Reward", rew_items)

        
    st.markdown("<br><div class='header-style'>Complete Operation Profile</div>", unsafe_allow_html=True)
    # Group beautifully by domain with strict ordering
    domain_order = ["ENV", "REWARD", "TELEMETRY", "TRAINING", "LOGGING"]
    domains = sorted(list(set([idx.split(" | ")[0] for idx in diff_df.index])), key=lambda x: domain_order.index(x) if x in domain_order else 999)
    for dom in domains:
        sub_df = diff_df[diff_df.index.str.startswith(dom + " | ")]
        render_html_table(sub_df, dom, show_domain=False)

else:
    # ── Multi View (Cross-Check Matrix) ────────────────────
    is_varying = diff_df.nunique(axis=1) > 1
    varying_df = diff_df[is_varying]
    static_df = diff_df[~is_varying]

    st.markdown("---")
    st.markdown("<div class='header-style'>Configuration Deviations</div>", unsafe_allow_html=True)
    
    domain_order = ["ENV", "REWARD", "TELEMETRY", "TRAINING", "LOGGING"]

    if not varying_df.empty:
        st.caption("Parameters exhibiting variability across the selected operational domains.")
        v_domains = sorted(list(set([idx.split(" | ")[0] for idx in varying_df.index])), key=lambda x: domain_order.index(x) if x in domain_order else 999)
        for dom in v_domains:
            sub_df = varying_df[varying_df.index.str.startswith(dom + " | ")]
            render_html_table(sub_df, dom, show_domain=False)
    else:
        st.success("Selected operations maintain identical hyperparameter configurations.")

# ---------------------------------------------------------------------------
# Visual Telemetry Grid (Moved Above Static Baseline)
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("<div class='header-style'>Visual Telemetry</div>", unsafe_allow_html=True)

cols = st.columns(len(selected_operations))
for idx, r_disp in enumerate(selected_operations):
    with cols[idx]:
        st.markdown(f"**{r_disp}**")
        v_dir = registry[r_disp]["video_path"]
        v_files = sorted(list(v_dir.glob("*.mp4")), key=os.path.getmtime, reverse=True)
        if v_files:
            st.video(str(v_files[0]))
            int_meta = registry[r_disp]["folder_name"]
            st.markdown(f"<span style='color:#666; font-size:10px;'>{int_meta}</span>", unsafe_allow_html=True)
        else:
            st.warning("Video Link Broken")

# ---------------------------------------------------------------------------
# Static Baseline Grouped (Moved to Bottom)
# ---------------------------------------------------------------------------
if len(selected_operations) > 1:
    st.markdown("<br><div class='header-style'>Static Configuration Baseline</div>", unsafe_allow_html=True)
    st.caption("Parameters maintaining constant values across all viewed operations.")
    with st.expander("Explore Static Baseline Configurations", expanded=False):
        domains = sorted(list(set([idx.split(" | ")[0] for idx in static_df.index])), key=lambda x: domain_order.index(x) if x in domain_order else 999)
        for dom in domains:
            sub_df = static_df[static_df.index.str.startswith(dom + " | ")]
            render_html_table(sub_df, dom, show_domain=False)
