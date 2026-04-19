"""
scripts/curriculum.py
=====================
Orchestrates a sequence of training runs (L0 -> L1 -> L2) where each 
stage inherits weights from the previous level's final checkpoint.

Usage
-----
    uv run python scripts/curriculum.py \
        training.run_name=stage1_full \
        logging.wandb_mode=online
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
from omegaconf import OmegaConf

from core.config import load_config, validate_config
from training.runner import train

def run_curriculum():
    # 1. Load the base config + global interactive overrides (e.g. wandb mode)
    global_cfg = load_config(cli_overrides=True)
    
    # Levels to train through
    levels = ["00", "01", "02"]
    
    # Shared identifier for this specific curriculum sequence
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_run_name = global_cfg.logging.get("run_name", "curriculum")
    curriculum_id = f"{base_run_name}_{timestamp}"
    
    # Create the parent directory for the entire curriculum
    curriculum_root = Path("outputs") / "curriculum" / curriculum_id
    curriculum_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "═"*60)
    print(f"  STARTING CURRICULUM: {base_run_name}")
    print(f"  Root Dir: {curriculum_root}")
    print(f"  Levels:   {', '.join(levels)}")
    print("═"*60)

    last_checkpoint = global_cfg.training.get("checkpoint_path", None)
    
    for idx, level_id in enumerate(levels):
        level_name = f"L{level_id}"
        print(f"\n🚀 [LEVEL {idx+1}/{len(levels)}] Starting {level_name}...")
        
        # 1. Load baseline + level-specific config
        cfg = load_config(cli_overrides=False)
        OmegaConf.set_readonly(cfg, False)
        
        # Search for a file starting with the level ID (e.g., 00)
        level_dir = Path("src/curriculum_config/levels")
        matches = list(level_dir.glob(f"{level_id}*.yaml")) + list(level_dir.glob(f"{level_id}*.yml"))
        
        if matches:
            level_cfg_path = matches[0]
            level_overrides = OmegaConf.load(level_cfg_path)
            cfg = OmegaConf.merge(cfg, level_overrides)
        
        # 2. Inherit/Override parameters
        # Apply the global CLI overrides (except level-specific things)
        cfg.logging = global_cfg.logging
        cfg.training.seed = global_cfg.training.seed
        cfg.training.num_envs = global_cfg.training.num_envs
        cfg.training.checkpoint_path = last_checkpoint
        
        # ── THE MAGIC: Nested Directory ──
        # By setting log_dir to our curriculum_root, train() will create STAGE_TS subfolders inside it.
        cfg.logging.log_dir = str(curriculum_root)
        cfg.logging.run_name = f"L{level_id}_{level_name}"
        
        OmegaConf.set_readonly(cfg, True)
        validate_config(cfg)
        
        # 3. RUN TRAINING
        final_ckpt_path = train(cfg)
        
        # 4. CAPTURE CHECKPOINT FOR NEXT STAGE
        last_checkpoint = final_ckpt_path
        print(f"✅ Stage {level_name} complete.")
        
        # Small cooldown for GPU/W&B sync
        time.sleep(2)

    print("\n" + "═"*60)
    print(f"  CURRICULUM FINISHED SUCCESSFULLY!")
    print(f"  Results: {curriculum_root}")
    print("═"*60)

if __name__ == "__main__":
    run_curriculum()


