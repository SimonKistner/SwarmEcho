# Maintained assumptions

- The maintained environment is three-dimensional. Actions and physical
  coordinates have XYZ components.
- Production training uses recurrent actor and critic memory.
- Config files are strict: unknown sections or fields are rejected.
- Each level selects one building map through `env.map_names`.
- All bundled map and level files remain available.
- The actor uses local observations and configured communication; privileged
  critic features affect training only.
- Reward defaults and every bundled level's values are preserved through the
  removal of the old environment.
- Checkpoint architecture, observation, and value-normalization contracts remain
  enforced. Existing compatibility paths are retained.
- Both categorical-stage and robust-rate evaluation CSVs remain readable.
  Evaluation coordinates require XYZ.
- The general run-analysis dashboard remains supported for runs. It may
  display archived media; displaying media does not provide a simulation runtime.
- The building editor uses a canvas for overlays; canvas API dimensionality does
  not define the environment geometry.
- Training, ML tests, and dependency installation run in the user's WSL
  environment. Static checks cannot establish GPU numerical equivalence.

## Current level families

The M-series includes the open cuboid, tall cuboid, and randomized-obstacle
levels. The B-series includes `B00_test`, `B01a_office_find_only`, and
`B01b_office`. Level YAML files are authoritative for settings and map choice.

Historical design proposals are explicitly labeled in the archive and roadmap.
