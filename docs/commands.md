# SwarmEcho Command Reference

This document serves as a cheat sheet for all common execution commands in the SwarmEcho project, organized by the logical workflow of designing, testing, training, and analyzing.

Unless noted otherwise, run all commands from the project root directory.

---

## Environment Setup & Smoke Tests

### 1. Activate Virtual Environment
If you prefer to work inside the virtual environment rather than prefixing commands with `uv run`:
```bash
source .venv/bin/activate
```

---

## Map Design & Analysis Dashboard

### Architectural Map Builder
Launch the interactive Streamlit editor to design new maps and obstacle layouts:
```bash
uv run streamlit run src/curriculum_config/maps/scripts/map_builder.py
```

### Discovery & Analysis Dashboard
Launch the Streamlit visualization tool to check run stats, compare parameters, and view evaluation/rendering videos:
```bash
uv run streamlit run src/analysis/dashboard.py
```

### Consolidated Map Preview & Renderer
Render static images or short video/GIF rollouts (always simulated) of any map blueprint. All outputs are automatically saved to `outputs/map_previews/`:
```bash
# Render default static PNG blueprint image (showing spawn zones and real spawn positions)
uv run python src/visualize/render_preview.py M01_grid_maze

# Render a simulated video rollout (MP4, 10 frames, fast renderer, 3 drones, 1 target, 1 base)
uv run python src/visualize/render_preview.py M01_grid_maze --mode video

# Render a clean architectural SVG blueprint without spawns or zones
uv run python src/visualize/render_preview.py M01_grid_maze --format svg --no-spawns --no-zones

# Render a snappy simulated GIF preview using a specific level config
uv run python src/visualize/render_preview.py memory_t_maze_8 --mode gif --level MEM_T8_memory
```

---

## Drone Swarm Training

### Single Run Training
Train a drone swarm on a specific level:
```bash
uv run python src/training/train.py level=01 logging.run_name=maze01_v1 

logging.wandb_mode=disable
```

### Multi-Seed Sequential Training
Run multiple training iterations with the same configuration, utilizing different random seeds. The run name will automatically receive a `_seed_N` suffix:
```bash
uv run python src/training/multi_train.py level=M01 logging.run_name=maze01_v2 seeds=5 base_seed=99
```

### Curriculum Training
Train through sequential levels (inheriting checkpoint weights from the previous level) either with default stages or custom levels:
```bash
# Run default curriculum (B05a, B05b)
uv run python src/training/curriculum.py

# Run a custom sequence of levels
uv run python src/training/curriculum.py levels=B02,B03,B04
```

---

## Spatial Failure Analysis Pipeline

### Run Evaluation Sweep & Generate Heatmaps
Simulate parallel environments (4096 by default) to sweep for failure coordinates and generate heatmaps (failed chain targets, and merged or individual target-not-found heatmaps):
```bash
uv run python src/training/evaluate_pipeline.py checkpoint=outputs/my_run/checkpoints/ckpt_001000
```

### Re-simulate & Render Failed Targets from CSV
Extract the failed target positions from a generated CSV file and render individual simulation rollout videos for debugging (limiting to the first 5 in this example):
```bash
uv run python src/training/evaluate_pipeline.py checkpoint=outputs/my_run/checkpoints/ckpt_001000 --render-failed-csv=5
```
