# SwarmEcho Cleanup Plan

| Section | Job count |
|---|---:|
| Confirmed-to-do queue | 2 |
| To-investigate | 0 |
| Open proposals | 0 |
| Backlog | 1 |
| Denied proposals | 3 |
| **Total** | **6** |

## Confirmed-to-do queue

### Q06 — Separate task-specific 2D state from general environment state

Use this as the first structural step of the 3D transition/optimization phase.

Separate at least these concepts:

- Physical/kinematic world state.
- Communication and target-knowledge state.
- Relay-task and discrete finder-path state.
- Optional renderer/diagnostic state.

The 3D physics path should not inherit finder-path arrays and 2D maze-cell
bookkeeping merely because they currently live in `EnvState` and
`physics.py`. Preserve the maintained discrete-finder-path behavior while
moving its ownership to the relay task layer.

### Q07 — Reconcile documentation after code cleanup

At the end of cleanup:

- Update README structure, commands, supported levels, curriculum defaults,
  observation dimensions, output paths, and package descriptions.
- Update environment, training, analysis, and architecture documents against
  raw code.
- Mark historical experimental specifications as archival or remove them from
  the current documentation path.
- Update `assumptions.md`, `idea_archive.md`, and `future_features.md` when
  final implementation choices differ from their current statements.
- Remove branch-specific documentation from the active documentation set.
- Correct code comments and docstrings, including pre-squash action-buffer
  semantics and preview defaults.
- Document the final supported map/level sequence.

Known discrepancies to resolve include:

- README references to nonexistent `base_params.yaml`, IPPO, and numeric levels.
- 57-dimensional/coverage-probe documentation versus the current default
  37-dimensional observation.
- 1,024-environment documentation versus the 4,000 default.
- Old curriculum and command examples.
- Architecture text describing memory as future work although recurrent
  memory and TarMAC are current defaults.

## Backlog

### O09 — Split the large FastHTML artifact server

If the FastHTML interface remains, split its HTML construction, routes,
media service, session state, and queue control rather than retaining one large
server module. This is not denied, but it is not relevant to the 3D transition
or performance-preparation work now.

## Denied proposals

### D01 — Build a heavy maintained invariant suite before 3D

Denied for now. The project may change substantially during the 3D transition,
so the investment is not justified at this stage. A normal maintained training
run remains the integration test. This does not deny adding targeted tests
later when the post-transition architecture stabilizes.

### D03 — Remove unapproved runtime features

Denied as a blanket cleanup action. In particular, leave these untouched until
separately approved:

- Feed-forward versus recurrent actor/critic paths.
- TarMAC enablement and communication cadence.
- `terminate_on_target_found`.
- `target_found_requires_delivery`.
- Base-vector, target-vector, and coverage-probe observation switches.
- Evaluation heatmap options.
- Renderer debug overlays.
- Reward-system alternatives that are still used by retained levels.

The only feature removals currently approved are those explicitly named in the
confirmed queue.

### D04 — Consolidate or remove analysis applications

Denied. Keep the existing analysis interfaces and do not merge or remove them
as an architectural cleanup. Their separate applications remain.
