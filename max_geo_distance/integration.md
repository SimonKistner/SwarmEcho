# Dynamic spawn budget and agent-count curriculum

## Evidence and rollout boundary

The accepted reference run is
[20261014_230730_049601Z_11_maps_geometry.txt](results/20261014_230730_049601Z_11_maps_geometry.txt)
and [its real GPU hold check](results/20261014_230730_049601Z_12_maps_runtime_chain_hold.txt).
All 11,995 constructed assignments passed. Five of 12,000 targets remain
unresolved in cell (0,0,2) of 02344; its graph is connected but has no suitable
endpoint/finder witness for them. They are not known to be physically impossible.
The filter rejects unresolved pairs instead of guessing a small agent count.

Those results validate the reference calculator's placements. The integrated
JAX implementation is new and must be checked separately. No production query,
training run, compilation check or GPU experiment has been executed by the
coding agent. Query throughput and first compilation cost are not established.

## Parameters

```yaml
training:
  dynamic_max_geo_distance:
    enabled: false
  agent_count_curriculum:
    enabled: false
    start_agents: 2
    additional_agents: 0
  oom_recovery:
    enabled: false

evaluation:
  dynamic_max_geo_distance: null
  eval_parallel_envs: null # Optional: follow effective training.num_envs.
```

Both features default off. Evaluation null inherits the training filter;
false explicitly requests unrestricted evaluation. The curriculum requires
filtered evaluation, evaluation.num_agents=null, saved checkpoints, and
evaluation-based early exit. Existing early_exit_success_rate,
early_exit_hold_evals and optional length-reduction settings determine mastery.
At curriculum entry, env.num_agents is both the maximum geometric stage budget
and the hard deployable-population cap.
additional_agents defaults to zero and adds deployable slots without admitting
harder targets: stage k uses min(k + additional_agents, env.num_agents) slots and
accepts constructed counts only through k. The final stage has no spare capacity,
even when additional_agents is very large. G01 explicitly sets it to one.

When disabled, no relay preparation/query runs and the existing proposal keys,
sampling branches and adaptive controller identity remain unchanged. Global
defaults remain disabled. G01_static explicitly enables the filter, curriculum,
adaptive target spawning and adaptive size, with one initial agent and stage
geometric budgets two through seven and deployable caps 3,4,5,6,7,7;
G02_static copies those settings for five static maps; other levels inherit the
disabled feature defaults and zero additional agents.

## Runtime integration

Production logic lives in `src/swarmecho/env/relay.py`; feature documentation,
reference experiments and verification commands remain in max_geo_distance.
Map preparation adds raw-distance predecessor paths, existing graph-edge lists
and authored opening bounds to the normal immutable map bank. Geometry-derived
records and prepared banks are reused across population stages in the process.
There is no continuous endpoint atlas, resolution knob, configured memory cap
or CPU callback during reset. Cache lifetime is the training process; this
implementation does not write a new persistent atlas to the on-disk map bank.

The JAX query attaches the actual base's legal drone start and proposed target
to that graph, reconstructs one shortest path, repairs segment ranges, and
applies the horizontal, doorway, lifted, bypass and generalized connector
families in bounded arrays with the existing stage order. Height feasibility
is computed by an interval endpoint sweep rather than Python interval lists.
This is an analytic construction, not a simulation or alternate-path search.
Float32 path ties, interval endpoint treatment and conservative rollback can
produce different counts from the CPU reference; verification reports them.

A complete witness must pass rounded placement and physical link checks with
the numerical margin. If final simplification fails, the validated baseline
is retained. Missing endpoints can attach through visible finder positions on
existing reachable nodes/edges. A necessary Euclidean bound rejects obviously
over-budget proposals before the full query. The actual admission rule is
valid witness and constructed count <= the geometric stage budget. Counts remain upper
bounds, not guaranteed global minima.

The filter supports one static map, static suites, generated persistent maps,
cached random-map records and randomized bases. Cached random records obtain
opening metadata from their authored map YAMLs during enabled preparation.
Randomized bases use their actual proposed coordinates; no table of continuous
base/target pairs is required. Generated extra obstacles and coverage tasks
are rejected because the prepared target-map witness contract does not apply.

Existing noSpawn and wall-buffer rules remain in the proposal sampler. Minimum
geodesic separation, when enabled, is an additional required condition. With
dynamic filtering and randomized bases, ordinary proposals redraw the joint
base/target pair; externally supplied targets remain fixed. Rejection is
bounded by training.spawn_pair_max_attempts. Exhaustion sets the existing
zero-attempt sentinel, which startup, rollouts and evaluation check before
using the run's data. There is no unfiltered fallback or unbounded skip loop.

The calculator runs during reset, not every physics step. Batched resets use
small groups and sparse reset handling. This is intended to control working
memory, but the new GPU query/compile cost must be measured by the command below
before launching a large run, particularly a 4,000-map bank.

## Adaptive target spawning and adaptive size

Adaptive spawning supports static authored suites with a fixed base per map.
Persistent random-map banks are also supported when adaptive map sampling is
enabled. That toggle supplies a fixed per-map discovery budget, independent of
lane count (800 targets by default), retaining cells at the same 1% threshold.
Without the new toggle, the original startup-lane discovery behavior is retained:

1. Draw uniform legal-volume target proposals with the normal feasibility
   rejection. No adaptive weights are applied to this discovery batch.
2. Count the accepted target positions by cell. Keep cells with at least
   `ceil(0.01 * map_lane_count)` accepted targets: 40 of 4000 for one map,
   or eight of 800 for each G02 map with 4000 lanes total.
   Counts use actual lanes, excluding padding in grouped reset batches.
3. Set other cells' baseline and proposal probabilities to exactly zero.
   Normalize retained cells' original full legal spawn volumes to 100%; observed
   counts are used only for the mask, not as probability weights. No reachable
   fraction, atlas, additional feasibility sweep or configuration knob is added.
4. Apply the existing adaptive success weighting, probability floors and
   smoothing on the retained support only. Excluded cells do not affect worst
   failure, normalization or plateau warnings. Even when every retained cell
   succeeds, excluded cells' unmeasured zero rates cannot cause zero-division.

With adaptive map sampling enabled, extra probes use the same bounded uniform
filtered reset in groups of 32. A map with exhausted attempts fails explicitly.
The actor map selection is separate from its geometric agent budget; changing
maps never admits a target outside the current stage's certificate. See
[adaptive map sampling](../docs/01_reference/10_adaptive_map_sampling.md).

In the original discovery path, targets are reused as the initial live environments to avoid
another reset/search pass. Thus a cell excluded by the mask may still occur in
that one initial batch; its completed outcomes are ignored by the adaptive
controller. All later target proposals use the masked weights. A discovery
with no cells reaching the threshold fails explicitly rather than falling back
to unrestricted adaptive proposals.

Rejection remains unchanged: every retry may select another retained cell.
Partly reachable cells retain their full proposal weight, while only valid
positions survive. Accepted cell frequencies therefore depend on the reachable
space within each cell. Rejections are sampler work, not policy failures.
The mask is an empirical stage selection, not proof that excluded cells are
physically unreachable. No exploration allocation is reserved for excluded
cells within the stage.

The dashboard layout and plots are unchanged. All original cells remain visible;
excluded cells show zero probability and zero baseline, and retained weights sum
to 100%. A fresh run publishes metadata/history only after initialization, so
it does not show provisional probabilities across all cells while reset is
compiling. The existing plots still describe proposal probabilities, without an
additional accepted-frequency plot. Discovery counts, batch size, agent budget
and threshold are recorded in `adaptive_spawn/eligibility.json` and metadata for
inspection; no extra dashboard controls are added.

Controller checkpoint identity includes the budget, geometric settings and the
startup-mask policy. Checkpoints preserve the discovery counts, mask, adaptive
probabilities and success history. Same-stage resume restores them before its
new reset; it does not rediscover support using adaptively weighted targets.
The separate eligibility file also preserves the mask across an OOM retry that
has no completed checkpoint yet. A smaller retry batch only changes discovery's
threshold if discovery had not completed before the OOM. New agent stages
discover a fresh mask from all legal cells at their new budget and current
environment count, retaining model weights rather than the previous mask or
success windows.

Existing maskless relay/adaptive controller checkpoints have an older identity
and cannot be resumed under these new semantics. Their model weights may still
initialize a branch in a new run directory. Adaptive spawning without the
dynamic filter retains its previous sampler identity, checkpoint format and
full-volume distribution; no discovery mask is used.

Evaluation stays independent: it does not use this training mask or adaptive
weights. It samples uniformly from legal space conditioned only on its configured
feasibility filter and therefore can include cells omitted by startup discovery.

The filter uses the geometric budget, not the initial or current population.
Adaptive size uses the larger deployable cap, including possible reinforcements.
With one additional agent, budget two starts with one agent and permits calls up
to three, but every target still has a certified construction with at most two. Voting,
decommissioning and reference rewards continue unchanged. Initial agents are
clamped to the current stage cap. The geometric witness does not certify the
policy's reinforcement decisions, navigation or decommission avoidance.
Switching adaptive size on changes the existing action/observation contract;
checkpoint compatibility remains enforced. Within a curriculum, that mode is
constant across stages and shared weights can transfer between agent counts.

## Curriculum lifecycle

Ordinary training automatically dispatches the enabled agent curriculum.
For maximum seven, stages have geometric budgets 2,3,4,5,6,7 and cumulative target
sets 1..2, 1..3, and so on. With additional_agents=1, their deployable caps are
3,4,5,6,7,7, respecting the hard maximum of seven. Each has a separate
`<prefix>_agents_<k>` run name, where k remains the geometric budget, with
fresh environments/recurrent state/optimizer/controller and fresh evaluation
hold windows. Model checkpoint weights and cumulative branch step history
transfer to the next stage. No curriculum advance occurs on timestep-budget
exhaustion without evaluation mastery, including when nested in the existing
multi-level curriculum command.

Stage replays sample the current eligible target set; the diagnostic
replay_targets_from_storey override is cleared so a high-floor replay cannot
force a target outside an early-stage bucket. Custom checkpoint roots receive
separate stage subdirectories. A small atomic `agent_curriculum.json` inside
the first stage's run folder records the maximum geometric budget, additional
agents, per-stage deployable populations, mastery and checkpoint paths. It stays
in that first folder when resuming later stages.
Existing `<prefix>_agent_curriculum.json` files at the output root are
migrated into that folder when next opened, if the new journal does not exist.
Saved stage configs contain the actual population in env.num_agents and the
target budget in env.relay_agent_budget (null retains legacy population-based
filtering). The latter is generated by the curriculum and is preserved in replay
metadata. It also sets adaptive-spawn discovery and checkpoint sampler identity;
the spare never enlarges the discovery target set. Both training and evaluation
use this same target budget. additional_agents is ignored when curriculum
dispatch is disabled; explicit saved relay_agent_budget still applies.

Resume infers the interrupted budget/name from its saved stage config and checks
the maximum and additional_agents against the journal. Changing spare population
requires a branch with a new run name. To resume at the same stage, provide its
checkpoint with ckpt_loading_mode=resume and the original level maximum.

G01 and G02 use evaluation-based mastery at their configured 98% success threshold.
For static suites this is the equally weighted mean across maps, not a separate
98% requirement for every map. Per-map and worst-map evaluation success are
logged to expose imbalance without changing that inherited gate.
Evaluation inherits the current stage's feasibility filter and agent cap. It
samples uniformly over the legal spawn distribution conditioned on passing that
filter, without adaptive training proposal weights or action noise. Normal
noSpawn, base-cell and wall-buffer rules still apply. Stage two therefore tests
constructed counts 1..2; later stages expand the evaluated population.

G01 also enables [automatic OOM recovery](oom_recovery.md), starts with 4000
environments, and follows the effective training count in evaluation. A failed
worker is replaced with 500 fewer environments, resuming this stage's latest
complete checkpoint or retrying its original incoming handoff. Reduced counts
persist into later stages and are recorded in the curriculum journal. Enabled
recovery uses fresh processes, so the in-memory geometry cache described above
does not survive between those stages; persistent map-bank caches still apply.

## Commands and pending verification

Start with a small actual-production verification on the troublesome map:

```bash
uv run python max_geo_distance/validate_integration.py --maps B02c_static_map_02344 --targets 32 --reset-smoke
```

This prints CPU/JAX coverage and count differences, query/first-compilation and
warm-batch times, independently injects the production JAX witness positions
for the normal real-runtime hold check, and exercises actual filtered reset
with all four adaptive-spawn/adaptive-size combinations at cap two. It runs
no training episodes or policy. All output, including partial errors, is saved
as `results/<UTC timestamp>_13_production_queries.txt`. GPU is required.

Then use the full three-map comparison:

```bash
uv run python max_geo_distance/validate_integration.py --reset-smoke
```

The production gate fails if it loses a CPU-valid assignment or any produced
witness fails the real hold/placement checks. Shared unresolved pairs are still
reported in the overall denominator and rejected; they do not fail the port
gate solely for remaining unresolved. Count differences, including budget-by-
budget eligibility, remain visible and need assessment before a large run.

After production verification, enable both features with the normal command:

```bash
uv run swarmecho-train level=G01_static logging.run_name=G01_agent_curriculum training.dynamic_max_geo_distance.enabled=true training.agent_count_curriculum.enabled=true
```

G01 now enables all four toggles directly, so `uv run swarmecho-train level=G01_static`
is sufficient. It starts each episode with one agent and permits voted
reinforcements up to the current stage cap. Loading weights still requires a
checkpoint compatible with adaptive size's action/observation contract.
For the filter alone on G01, explicitly set training.agent_count_curriculum.enabled=false. Existing
training.profile_timing=true can record the cost during an actual training run.
