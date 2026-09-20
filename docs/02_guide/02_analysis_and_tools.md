# Analysis and tools

## General run-analysis dashboard

```bash
uv run swarmecho-dashboard
```

The Streamlit dashboard in `analysis/dashboard.py` remains the general
cross-run configuration and telemetry browser, including runs. Its code
and primary launch behavior are preserved. Historical media browsing remains
available where a run contains media.

## inspector

```bash
uv run swarmecho-inspect root=outputs
uv run swarmecho-inspect root=outputs/RUN port=8765 open=false
```

The inspector discovers replay manifests and evaluation heatmaps. It provides
building visibility controls, replay inspection, target selection, and a
command for generating a replay at the selected XYZ position. Existing 3D
artifacts and compatibility fallbacks remain supported.

`swarmecho-eval-dashboard` and `swarmecho-artefacts` now launch the same
inspector. Use its `key=value` options, rather than old Streamlit or artefacts
server options. The old flat-map target picker, MP4 renderer, and evaluation
job queue are removed.

Evaluation CSVs contain XYZ coordinates with either categorical stages or
cumulative robust outcome rates. Both formats remain supported. Replay JSON
manifests and their numerical data are under the run's artifact hierarchy;
checkpoint evaluation groups artifacts beneath its checkpoint directory.

## building editor

```bash
uv run swarmecho-maze-builder --host 127.0.0.1 --port 8766
```

The historical command name is retained. The server serves the existing 3D
editor and `/api/buildings` routes for listing, loading, authoring, validation,
and saving. Map-name validation retains the same filename rules.

Create and edit storeys, walls, tiles, exclusions, roof geometry, and spawn
settings, then validate the saved YAML:

```bash
uv run swarmecho-validate-building src/swarmecho/curriculum_config/maps/custom_3d_building.yaml
```

The optional `--runtime-smoke` flag also constructs and resets the environment.
It requires the project's Python/JAX environment.

## Diagnostics

`swarmecho-validate` runs the small rollout/PPO validator.
`swarmecho-benchmark` benchmarks the environment.
The roadmap and radar diagnostic scripts remain under `tests/` and `tools/`.
