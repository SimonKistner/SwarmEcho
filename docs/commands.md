# SwarmEcho Command Reference

This document serves as a cheat sheet for all common execution commands in the SwarmEcho project, organized by the logical workflow of designing, testing, training, and analyzing.

Unless noted otherwise, run all commands from the project root directory.

---

## Phase 1: Environment Setup & Smoke Tests

### 1. Activate Virtual Environment
If you prefer to work inside the virtual environment rather than prefixing commands with `uv run`:
```bash
source .venv/bin/activate
```

### 2. Run the Environment Smoke Test
Verify that the JAX physics engine, observations, and rewards are stepping correctly before starting long training sessions:
```bash
uv run python src/tests/smoke_test_env.py
```

---

## Phase 2: Map Design & Validation

### 1. Architectural Map Builder
Launch the interactive Streamlit editor to design new maps and obstacle layouts:
```bash
uv run streamlit run src/curriculum_config/maps/scripts/map_builder.py
```

### 2. Validate Map Geometry
Verify JAX compatibility for all map blueprints and render collision-checking videos:
```bash
uv run python src/tests/validate_maps.py
```

---

## Phase 3: Drone Swarm Training

### 1. Single Run Training
Train a drone swarm on a specific level:
```bash
uv run python src/training/train.py level=01 logging.run_name=maze01_v1 

logging.wandb_mode=disable
```

### 2. Multi-Seed Sequential Training
Run multiple training iterations with the same configuration, utilizing different random seeds. The run name will automatically receive a `_seed_N` suffix:
```bash
uv run python src/training/multi_train.py level=M01 logging.run_name=maze01_v2 seeds=5 base_seed=99
```

### 3. Curriculum Training
Train through sequential levels (inheriting checkpoint weights from the previous level) either with default stages or custom levels:
```bash
# Run default curriculum (B05a, B05b)
uv run python src/training/curriculum.py

# Run a custom sequence of levels
uv run python src/training/curriculum.py levels=B02,B03,B04
```

---

## Phase 4: Run Analysis & Diagnostics

### 1. Discovery & Analysis Dashboard
Launch the Streamlit visualization tool to check run stats, compare parameters, and view evaluation/rendering videos:
```bash
uv run streamlit run src/analysis/dashboard.py
```

### 2. Component Standalone Diagnostics
Run isolated logic tests for individual subsystems to debug specific components:
```bash
# Physics Engine Standalone
uv run python src/env/physics.py

# Observation Space Standalone
uv run python src/env/observations.py

# Neural Network Architecture
uv run python src/models/mappo.py
```
