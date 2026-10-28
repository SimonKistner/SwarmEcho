# G03 frozen validation-map coordinate probe

Prepared 2026-10-26 for `G03_random_4k_v1_agents_4`; no policy training or target
generation changes. The latest completed checkpoint at inspection was
`ckpt_001250`. New runs resolve `latest` once and freeze it in the suite plan.

From the repository root in the existing WSL environment:

```bash
uv run python geo_experiments/suite.py plan=true
uv run python geo_experiments/suite.py
```

The checked-in `geo_experiments/probes.yaml` schedules one shared-coordinate
experiment: three input sources (finder GRU, finder message, first base message)
and three seeds, for nine independent MLP fits and one adaptive results page.
The target storey head, map-specific experts, map+cell classifier and own-cell
experiments are not scheduled.

## Selection and collection

`policy.map_source=validation` is an explicit opt-in for modern target probes.
Use all 400 recorded validation buildings, not the three inspection maps or the
3,600 policy-training maps. Verify the frozen validation list, stage and map file
hashes. Fail if a map is missing or changed; never generate a replacement.
Load the saved policy configuration, including five available agent slots and
the four-agent geometric budget. Use the production validation bank environment
and its target-feasibility rules, without inspection population/filter overrides,
adaptive training weights or the old fixed validation episode cases.

Collect fresh deterministic-policy episodes. Retain only episodes that delivered
the target to the base and contain all three feature snapshots; chain-held
success is an analysis attribute, not collection eligibility. The target
exclusions from the frozen geometry are unchanged. Reviewing easy-target
exclusion in general random-map generation remains separate work.

Budgets remain 10,000 training, 1,000 decoder-validation and 5,000 held-out
evaluation samples. Exact per-map quotas give 25 / 2–3 / 12–13 samples per map.
The 400 maps share one bank-backed compiled collector, and partial execution
lanes are padded without adding padding to sample counts or saved attempts.
Per-map episode keys and IDs make diagnostic replay reproducible within the
reported floating-point limitations.

Split each map by cell with exact quotas, but require observed coordinate-class
coverage only in the pooled training set. Requiring all 27 cells within each
25-sample training subset would defeat the shared-coordinate contract. No map ID
is supplied as an input. The usual conditional, source-specific ambiguity audit
remains enabled; packet collisions use saved snapshots and only conflicting
finder GRUs request history replay, using the same bank environment.

## Interpretation and compatibility

The 27 output classes are shared absolute XYZ cells. The same buildings occur in
all three probe splits: they were held out from policy training, not from MLP
training. This measures coordinate decodability on policy-held-out buildings,
not transfer to buildings excluded from decoder fitting. Random validation banks
with 20 or more maps omit individual-map breakdowns from held-out details; pooled
and target-required-agent/storey groups remain. Smaller authored-map reports keep
their map details. Three seeds share samples and are not three independent datasets.

Large suites store per-map distance blocks for true-map scoring. Existing
single-map and five-map geometry archives, default cell-coverage rules, rollout
selection and reporting remain supported. Historical G01 entry points retain
their original heads and eligibility. G02 saved suite configs/resume plans retain
their authored-map path; they do not opt into validation maps.

Configuration/map/split inspection can run on the host without JAX. GPU rollout,
bank preparation and MLP fitting must be run in the existing WSL environment.
