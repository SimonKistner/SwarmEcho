# 2D removal audit

Updated 2026-09-20.

The standalone and mixed-file cleanup is implemented. The general run-analysis
dashboard remains available for 3D runs and its implementation is unchanged.

## Removed

- 27 standalone 2D environment, training, evaluation, rendering, and validation
  files, plus their ten CLI registrations in the first pass.
- The 2D configuration schemas, loader, validation, and observation/action sizing.
- The old graph-masked privileged critic classes and model self-test.
- Legacy maze export/normalize/create routes and maze conversion core.
- The old evaluation dashboard and artefacts server, map renderer, queue, and
  supporting files. Their command names now launch the existing 3D inspector.
- XY-only evaluation CSV handling; both categorical-stage and robust-rate XYZ
  formats remain supported.
- Direct OpenCV, ImageIO/FFmpeg, Pillow, and FastHTML dependencies. Unreachable
  packages were pruned from the lockfile without upgrading retained packages.
  Pillow remains transitively required by the dashboard/plotting dependencies.

## Preserved and repaired

- All maps, levels, general run-analysis dashboard, inspector, replay serializer,
  actor, recurrent cells, production PPO, evaluation, checkpoint handling,
  orchestration, and building-editor UI/core remain unchanged.
- All building API routes remain; the name validator was extracted unchanged.
- 3D reward defaults now live directly in Baseline3DRewardConfig with identical
  values. Config declarations and loader behavior are retained.
- The generic MAPPO trainer remains because the 3D validator exercises it.
- The validator and benchmark were moved into the package so their console
  entry points and test imports resolve. The benchmark map path was repaired.
- Existing tests were repaired for renamed office levels and the inspector's
  current DOM/menu/camera behavior; production inspector code was not changed.
- Current guides and README describe the 3D runtime. Historical proposals are
  explicitly labeled; the old editor plan was moved into the archive.

## Verification

- All seven existing Node builder/inspector UI tests passed.
- All statically named internal package imports resolve.
- All 12 console entry points resolve to existing functions.
- Manifest and lockfile direct dependencies agree; locked references resolve.
- SHA-256 comparisons confirmed 30 protected files unchanged, including the
  general dashboard, inspector, every map and level, and core 3D subsystems.
- Source comparisons confirmed unchanged 3D config logic, environment calculations
  after substituting identical reward literals, retained critic implementations,
  validator logic, building routes, and map-name validation. The training change
  only removes the obsolete critic-construction switch.

No ML tests, training, dependency installation, virtual-environment commands, or
Python compilation checks were run. GPU numerical equivalence still needs
verification in the user's existing WSL environment:

```bash
uv run swarmecho-validate-3d
uv run pytest tests
```

Ordinary training/resume and evaluation against existing 3D checkpoints should
also be exercised there. Passing static/UI checks is not a claim that the full
GPU test suite has passed.
