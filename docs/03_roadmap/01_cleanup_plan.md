# SwarmEcho Cleanup Plan

> Historical design/proposal document. References to the former runtime describe
> the migration context, not supported commands. Use the current guides and
> [removal audit](05_2d_removal_audit.md) for the implemented volumetric-only state.

| Section | Job count |
|---|---:|
| Confirmed-to-do queue | 0 |
| To-investigate | 0 |
| Open proposals | 0 |
| Backlog | 1 |
| Denied proposals | 3 |
| **Total** | **4** |

## Backlog

### O09 — Split the large FastHTML artifact server

If the FastHTML interface remains, split its HTML construction, routes,
media service, session state, and queue control rather than retaining one large
server module. This is not denied, but it is not relevant to the volumetric transition
or performance-preparation work now.

## Denied proposals

### D01 — Build a heavy maintained invariant suite before volumetric

Denied for now. The project may change substantially during the volumetric transition,
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
