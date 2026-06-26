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

def extract_timestamp_robust(name: str):
    """Finds YYYYMMDD_HHMMSS or YYYY_MM_DD_HH_MM anywhere in the string."""
    match = re.search(r'(\d{8})_(\d{6})', name)
    if match:
        ts_str = f"{match.group(1)}_{match.group(2)}"
        try:
            return datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
        except ValueError:
            pass
            
    match2 = re.search(r'(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})', name)
    if match2:
        try:
            return datetime(int(match2.group(1)), int(match2.group(2)), int(match2.group(3)), int(match2.group(4)), int(match2.group(5)))
        except ValueError:
            pass
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

def extract_timestamp_from_video(v_dir: Path):
    for v_file in v_dir.rglob("*.mp4"):
        ts = extract_timestamp_robust(v_file.name)
        if ts:
            return ts
    return None

@st.cache_data(ttl=60)
def find_all_runs(outputs_root: Path, wandb_root: Path):
    """Heuristic discovery of operation footprints, prioritizing local folders."""
    local_candidates = {} # path -> metadata
    
    if outputs_root.exists():
        exclude_dirs = {"videos", "artifacts", "checkpoints", "to_delete", "wandb", "files", "logs", "bin", ".git"}
        for root, dirs, files in os.walk(str(outputs_root)):
            # Prune directory search in-place so we don't descend into excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            p = Path(root)
            cfg_file = p / "config.yaml"
            v_dir_old = p / "videos"
            v_dir_new = p / "artifacts"
            
            has_mp4 = False
            active_v_dir = None
            
            if v_dir_new.exists():
                for _ in v_dir_new.rglob("*.mp4"):
                    has_mp4 = True
                    active_v_dir = v_dir_new
                    break
            
            if not has_mp4 and v_dir_old.exists():
                for _ in v_dir_old.rglob("*.mp4"):
                    has_mp4 = True
                    active_v_dir = v_dir_old
                    break
            
            if cfg_file.exists() or has_mp4:
                # We found a potential run directory
                ts = extract_timestamp_robust(p.name)
                if not ts and active_v_dir:
                    ts = extract_timestamp_from_video(active_v_dir)
                if not ts:
                    # Windows creation time is usually reliable for linking
                    ts = datetime.fromtimestamp(p.stat().st_ctime)
                
                group_name = p.parent.name if p.parent != outputs_root else "Standalone"
                if group_name == "curriculum": group_name = "Direct Ops" # Flatten one level if needed
                
                local_candidates[str(p)] = {
                    "ts": ts,
                    "path": p,
                    "group": group_name,
                    "name": p.name,
                    "v_dir": active_v_dir or v_dir_new,
                    "cfg_file": cfg_file if cfg_file.exists() else None
                }

    # Match with WandB runs
    wandb_runs = []
    if wandb_root.exists():
        for w_path in wandb_root.glob("run-*"):
            if not w_path.is_dir() or w_path.name == "to_delete": continue
            w_ts = extract_timestamp_robust(w_path.name)
            if w_ts:
                wandb_runs.append({"ts": w_ts, "path": w_path})

    run_registry = {}
    
    for _, local in local_candidates.items():
        # Source of Truth: The local config.yaml snapshot created at run-start
        raw_cfg = None
        if local["cfg_file"]:
            try:
                raw_cfg = OmegaConf.to_container(OmegaConf.load(local["cfg_file"]), resolve=True)
            except: pass
            
        if raw_cfg is None: continue # Skip folders that aren't valid runs
        
        # Best-Effort WandB Linking (for URL/Metrics only)
        match_w_path = None
        internal_wandb = local["path"] / "wandb"
        if internal_wandb.exists():
            runs = list(internal_wandb.glob("run-*"))
            if runs:
                match_w_path = sorted(runs, key=os.path.getmtime)[-1]
        
        if not match_w_path:
            for w in wandb_runs:
                if abs((w["ts"] - local["ts"]).total_seconds()) <= 300:
                    match_w_path = w["path"]
                    break
        
        verified = (match_w_path is not None)
        if "_wandb" in raw_cfg: raw_cfg = flatten_wandb_config(raw_cfg)
        
        # Calculate Steps Trained (Priority: WandB Summary > Checkpoints)
        actual_steps = "Unknown"
        
        # 1. Try WandB Summary
        if match_w_path:
            summary_path = match_w_path / "files" / "wandb-summary.json"
            if summary_path.exists():
                try:
                    with open(summary_path, 'r') as f:
                        import json
                        summary = json.load(f)
                        # Look for common step keys
                        step_val = summary.get("global_step") or summary.get("total_steps") or summary.get("_step")
                        if step_val is not None:
                            total_steps = int(step_val)
                            if total_steps >= 1_000_000:
                                actual_steps = f"{total_steps / 1_000_000:.1f}M"
                            else:
                                actual_steps = f"{total_steps:,}"
                except: pass

        # 2. Fallback to Checkpoints
        if actual_steps == "Unknown":
            ckpt_dir = local["path"] / "checkpoints"
            if ckpt_dir.exists():
                ckpts = sorted(list(ckpt_dir.glob("ckpt_*")), key=lambda x: x.name)
                if ckpts:
                    latest_ckpt = ckpts[-1].name
                    try:
                        updates = int(latest_ckpt.split("_")[1])
                        t_cfg = raw_cfg.get("training", {})
                        envs = int(t_cfg.get("num_envs", 1024))
                        rollout_steps = int(t_cfg.get("num_steps", 128))
                        total_steps = updates * envs * rollout_steps
                        if total_steps >= 1_000_000:
                            actual_steps = f"{total_steps / 1_000_000:.1f}M"
                        else:
                            actual_steps = f"{total_steps:,}"
                    except: pass
        
        if "telemetry" not in raw_cfg: raw_cfg["telemetry"] = {}
        raw_cfg["telemetry"]["actual_steps_trained"] = actual_steps
        
        display_name = f"{local['group']} | {local['name']}"
        run_registry[display_name] = {
            "id": display_name,
            "config": raw_cfg,
            "outputs_path": local["path"],
            "wandb_path": match_w_path,
            "group_name": local["group"],
            "run_name": local["name"],
            "display_name": display_name,
            "video_path": local["v_dir"],
            "created": local["ts"].timestamp(),
            "folder_name": local["path"].name,
            "verified": verified
        }
                
    # Sort by creation time (newest first)
    sorted_reg = dict(sorted(run_registry.items(), key=lambda x: x[1]["created"], reverse=True))
    return sorted_reg

# ---------------------------------------------------------------------------
# Resumption History Parser
# ---------------------------------------------------------------------------

def get_run_history_chain(run_data) -> str:
    # 1. Try to read step_history.json from the latest checkpoint
    outputs_path = Path(run_data["outputs_path"])
    ckpt_dir = outputs_path / "checkpoints"
    if ckpt_dir.exists():
        ckpts = sorted(list(ckpt_dir.glob("ckpt_*")), key=lambda x: x.name)
        if ckpts:
            latest_ckpt = ckpts[-1]
            history_file = latest_ckpt / "step_history.json"
            if history_file.exists():
                try:
                    import json
                    with open(history_file, "r") as f:
                        data = json.load(f)
                    history_list = data.get("history", [])
                    if history_list:
                        chain = []
                        for entry in history_list:
                            name = entry.get("run_name", "unknown")
                            steps = entry.get("steps", 0)
                            steps_str = f"{steps / 1_000_000:.1f}M" if steps >= 1_000_000 else f"{steps:,}"
                            chain.append(f"<b>{name}</b> ({steps_str})")
                        return " ➔ ".join(chain)
                except:
                    pass
                
    # 2. Fallback: Check config.yaml for checkpoint_path
    cfg = run_data["config"]
    t_cfg = cfg.get("training", {}) if isinstance(cfg, dict) else {}
    checkpoint_path = t_cfg.get("checkpoint_path", None)
    if checkpoint_path:
        try:
            parent_ckpt_name = Path(checkpoint_path).name
            parent_run_name = Path(checkpoint_path).parents[1].name
            # Also get steps from the checkpoint name
            try:
                updates = int(parent_ckpt_name.split("_")[1]) if "_" in parent_ckpt_name else 0
                envs = int(t_cfg.get("num_envs", 1024))
                rollout_steps = int(t_cfg.get("num_steps", 128))
                parent_steps = updates * envs * rollout_steps
                parent_steps_str = f"{parent_steps / 1_000_000:.1f}M" if parent_steps >= 1_000_000 else f"{parent_steps:,}"
            except:
                parent_steps_str = "unknown steps"
            
            current_steps_str = run_data["config"].get("telemetry", {}).get("actual_steps_trained", "Unknown")
            return f"<b>{parent_run_name}</b> ({parent_steps_str}) [Resumed {parent_ckpt_name}] ➔ <b>{run_data['run_name']}</b> ({current_steps_str})"
        except:
            pass
        
    # 3. Scratch run (no history)
    current_steps_str = run_data["config"].get("telemetry", {}).get("actual_steps_trained", "Unknown")
    return f"<b>{run_data['run_name']}</b> ({current_steps_str}) [Started from Scratch]"


# ---------------------------------------------------------------------------
# State & Data Loading
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUTS_ROOT = BASE_DIR / "outputs"
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
        g_name = r_data["group_name"]
        if g_name not in grouped_runs: grouped_runs[g_name] = []
        grouped_runs[g_name].append((r_disp, r_data["run_name"]))
        
    selected_operations = []
    
    for g_name, runs in grouped_runs.items():
        st.markdown(f"<div class='sidebar-cgroup'>{g_name}</div>", unsafe_allow_html=True)
        for r_disp, r_name in runs:
            r_data = registry[r_disp]
            verified_icon = "🟢" if r_data.get("verified") else "⚪"
            label = f"{verified_icon} {r_name}"
            
            # Default to selecting the most recent standalone just to show something on load
            is_def = (selected_operations == []) and (g_name == "Standalone")
            if st.checkbox(label, value=is_def, key=r_disp):
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
all_keys = set()
for r_disp in selected_operations:
    cfg = registry[r_disp]["config"]
    flat_cfg = {}
    for section, params in cfg.items():
        if str(section).lower() in ["_wandb", "wandb"]: continue
        if isinstance(params, dict):
            for k, v in params.items():
                key = f"{section.upper()} | {k}"
                flat_cfg[key] = str(v)
                all_keys.add(key)
        else:
            key = f"GLOBAL | {section}"
            flat_cfg[key] = str(params)
            all_keys.add(key)
    comp_data[r_disp] = flat_cfg

# Ensure all keys exist in all runs (use "NaN" for missing)
for r_disp in selected_operations:
    for key in all_keys:
        if key not in comp_data[r_disp]:
            comp_data[r_disp][key] = "NaN"

diff_df = pd.DataFrame(comp_data)

# Sort: Differences FIRST, then alphabetically
is_different = diff_df.apply(lambda x: x.nunique() > 1, axis=1)
diff_df["_sort_diff"] = is_different
# Sort by difference status first, then by the parameter name (index)
diff_df.sort_values(by=["_sort_diff"], ascending=[False], inplace=True)
diff_df.drop(columns=["_sort_diff"], inplace=True)

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

    st.markdown("<div class='section-title'>Resumption History</div>", unsafe_allow_html=True)
    history_chain = get_run_history_chain(registry[r_disp])
    info_html = "<div class='se-table-container'><table class='se-table'>"
    info_html += "<thead><tr><th>Run Name</th><th>Heritage / Resumption History</th></tr></thead><tbody>"
    info_html += f"<tr><td><b>{r_disp}</b></td><td>{history_chain}</td></tr>"
    info_html += "</tbody></table></div>"
    st.markdown(info_html, unsafe_allow_html=True)
    
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
    st.caption("Parameters exhibiting variability across the selected operational domains.")

    # Resumption History (Unconditional Info)
    st.markdown("<div class='section-title'>Resumption History</div>", unsafe_allow_html=True)
    info_html = "<div class='se-table-container'><table class='se-table'>"
    info_html += "<thead><tr><th>Run Name</th><th>Heritage / Resumption History</th></tr></thead><tbody>"
    for r_disp in selected_operations:
        run_data = registry[r_disp]
        history_chain = get_run_history_chain(run_data)
        info_html += f"<tr><td><b>{r_disp}</b></td><td>{history_chain}</td></tr>"
    info_html += "</tbody></table></div>"
    st.markdown(info_html, unsafe_allow_html=True)
    
    domain_order = ["ENV", "REWARD", "TELEMETRY", "TRAINING", "LOGGING"]

    if not varying_df.empty:
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
        v_files = sorted(list(v_dir.rglob("*.mp4")), key=os.path.getmtime, reverse=True)
        if v_files:
            options = []
            option_map = {}
            for vf in v_files:
                stem = vf.stem
                cat_pref = ""
                # Categorize based on path structure or filename
                if "eval" in vf.parts or vf.parent.name == "eval" or stem.startswith("eval_") or "_eval_" in stem:
                    cat_pref = " [Eval]"
                elif "train" in vf.parts or vf.parent.name == "train" or stem.startswith("train_") or "_train_" in stem:
                    cat_pref = " [Train]"
                
                status_icon = "⚪"
                if stem.startswith("SUCCESS_"):
                    status_icon = "🟢"
                    stem = stem[len("SUCCESS_"):]
                elif stem.startswith("FAIL_"):
                    status_icon = "🔴"
                    stem = stem[len("FAIL_"):]
                
                # Check for update / ckpt numbers and episode numbers, plus new format features
                up_match = re.search(r'(?:update|ckpt)_(\d+)|_u(\d+)', stem)
                ep_match = re.search(r'ep(\d+)', stem)
                step_match = re.search(r'_s([\d\.]+[A-Za-z]?)', stem)
                
                parts = []
                if up_match:
                    up_val = int(up_match.group(1) or up_match.group(2))
                    parts.append(f"Update {up_val}")
                if step_match:
                    s_val = step_match.group(1)
                    s_val = re.sub(r'^0+', '', s_val)
                    if not s_val or not s_val[0].isdigit():
                        s_val = "0" + s_val
                    parts.append(f"Steps {s_val}")
                if ep_match:
                    parts.append(f"Ep {int(ep_match.group(1))}")
                
                if not parts:
                    # Tidy up raw name, dropping the timestamp prefix if present
                    clean_stem = re.sub(r'^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_', '', stem)
                    label = f"{status_icon}{cat_pref} {clean_stem.replace('_', ' ').title()}"
                else:
                    label = f"{status_icon}{cat_pref} " + " | ".join(parts)
                
                # Unique label check
                dup_idx = 1
                orig_label = label
                while label in option_map:
                    label = f"{orig_label} ({dup_idx})"
                    dup_idx += 1
                
                options.append(label)
                option_map[label] = vf
            
            selected_label = st.selectbox(
                "Select Rollout Video",
                options,
                index=0,
                key=f"vid_select_{r_disp}"
            )
            selected_vf = option_map[selected_label]
            st.video(str(selected_vf))
            int_meta = registry[r_disp]["folder_name"]
            st.markdown(f"<span style='color:#666; font-size:10px;'>Run: {int_meta}</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='color:#666; font-size:10px;'>Path: {selected_vf.relative_to(BASE_DIR)}</span>", unsafe_allow_html=True)
        else:
            st.warning("Video Link Broken / No Videos Found")

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
