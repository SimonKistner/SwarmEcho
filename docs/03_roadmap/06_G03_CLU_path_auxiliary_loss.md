# G03 CLU: path supervision of informed-agent GRUs

Status: implemented, opt-in; see [execution guide](../../ablations/cluster/path-probe/GEO_AUXILIARY.md).
Confirmed: train from scratch. Use twelve variants with the same two seeds,
including one clean policy-only variant. Existing G03 CLU baselines
remain useful historical references. Available experiment capacity: 24 slots.
Fresh independent probes stay in the manually executed offline probe suite;
training evaluates its existing auxiliary head only.

## Intended effect

An auxiliary decoder reads an informed, active agent's updated policy GRU and
predicts the existing base-to-target path mask. Its errors should shape the policy
representation, alongside the ordinary PPO gradients. It is training supervision,
not a reward term or an extra privileged observation. The decoder is optional at
deployment; the acting policy still uses its usual observations and memory.

Use the current 125-cell, base-anchored, yaw-canonical target and ordinary sigmoid
BCE initially. Reuse the exact route/label implementation, including clearance,
single-route tie-breaking and rasterization conventions. Do not independently
reimplement labels in the trainer.

For actor/encoder parameters, the objective is:

`existing PPO loss + lambda(update) * mean eligible path BCE`

Lambda=1 assigns the auxiliary loss a unit coefficient; it does not imply equal
scalar loss magnitude, equal gradient strength or a 50/50 optimizer update. PPO
uses its own surrogate/value/entropy terms and normalization. Compare gradient
norms on the shared representation to assess relative raw pressure.

Average BCE over the 125 cells first. Normalize eligible samples independently of
the total number of inactive or uninformed agents, so lambda does not implicitly
change with the fraction of agents that know the target.

Joint training is the initial proposal. The head learns to read the evolving GRU;
the GRU learns to support both action selection and path decoding. The head's
learning rate should be independently configurable: lambda controls the auxiliary
gradient entering the policy, not whether the head can keep learning. A detached
online head with lambda=0 is a useful optional diagnostic control. Actor/critic
PPO updates must remain unchanged in that case; head gradients must not enter the
PPO optimizer's clipping norm. The critic receives no direct path-loss gradient.

## Frequency and sample volume

In the current checked-in G03 CLU configuration:

- 6,000 environments, 7 agent slots, and 50 environment steps per rollout.
- One outer PPO update processes 300,000 environment transitions, then performs
  3 epochs over 6 environment minibatches (18 optimizer steps).
- One snapshot per currently knowing slot at an update boundary has an upper
  bound of 42,000 snapshots. A hypothetical 10% eligible fraction gives 4,200.
  This is a sample-count illustration, not a measured informed-agent fraction.
- Each selected sample is revisited in its minibatch for 3 epochs. These repeats
  are optimizer exposure, not new independent samples or targets.
- Thousands of same-episode states can share a path label. Count unique episodes
  and map/target pairs as well as snapshots; early training may have none eligible.

Recommended temporal selection: one uniformly selected eligible timestep per
environment/agent slot from the just-completed 50-step rollout. This remains
bounded by 42,000 selected snapshots, covers agents that knew the target and
decommissioned before the update boundary, and avoids always supervising the same
late phase. Each selected row retains its actual episode identity and target.
Boundary-only sampling is not implemented in this prototype.

Within an update, average selected snapshots per represented agent, then agents
per episode, then episodes. This is within-update balancing: a long episode can
appear in multiple updates. It does not claim the full-lifetime equal-episode
weighting of the offline dataset. No retrospective 20-row storage cap is needed
for this streaming training path.

Apply the auxiliary term in the same PPO minibatch/epoch updates as the selected
samples. Do not create separate policy-only auxiliary optimizer steps initially.
Keep the selected timestep mask fixed across a rollout's PPO epochs; recompute
hidden states with current parameters each epoch.

## Gradient and rollout integration

The active trainer is `src/swarmecho/training/ppo.py`, called by `train.py`;
`mappo_trainer.py` is not the current G03 integration point.

1. Extend rollout metadata/buffering with pre-actor knowledge eligibility,
   episode IDs and the map/base/target identity needed by selected samples.
   Use the same eligibility timing as the offline probe: active and target-known
   before the actor update, decoding the updated GRU that processed that input.
   Match the actor reset masks exactly. Auto-resets inside a rollout must not
   associate an old episode's GRU with the next episode's target.
2. Construct/cache one label per selected eligible episode, reusing it across
   agents, snapshots and PPO epochs. Initially batch CPU label construction on
   the existing host-buffer path; measure its cost before considering a device
   implementation. A label-cache key includes geometry/clearance/version and
   exact base/target endpoints. Bound and retire episode cache entries.
3. Extend the current differentiable recurrent replay to expose selected updated
   GRUs or accumulate auxiliary loss during the unroll. Saved rollout hidden
   arrays are detached data: training a head on those alone cannot shape the
   policy. Avoid copying/storing a full extra GRU dataset on the host.
4. Preserve the current 50-step truncated backpropagation boundary. Gradients can
   reach the actor encoder/GRU and communication computations inside that unroll;
   stored initial memory and stored base messages from earlier rollouts are
   boundary constants. Do not imply episode-length backpropagation.
5. A missing finite sparse-roadmap reference skips only that auxiliary label,
   with a counted reason. It must not replace/reject PPO episodes or change the
   environment's task distribution. Other label-construction errors fail clearly.
   This is an observed case, not a hypothetical fallback: two reproduced targets
   on `map_00275` had zero target attachments to the sparse reward roadmap and
   the actual reward distance query returned its 1e6 sentinel. Spawn checks use
   a different, denser relay-candidate/visual-contact graph. Passing those checks
   does not guarantee a finite path in the sparse reference graph. Audit label
   coverage by map and required-agent group. Improving the roadmap is a separate
   change that could alter reward behavior and baseline comparability.
6. Keep target geometry, route and mask out of actor/head inputs. They are labels
   and training-only metadata. No held-out-map samples enter either optimizer.

Move reusable label geometry from the experiment package into an appropriate
shared `src/swarmecho` module and retain a probe adapter. The trainer should not
depend on experiment runners or HTML/report code.

## Provisional first-experiment settings

These values are proposals for review, not established optima.

| Setting | Initial proposal | Purpose |
| --- | --- | --- |
| `enabled` | false globally; explicit G03 CLU experiment opt-in | Preserve existing baseline behavior. |
| `every_rollout_updates` | 1 | Use each newly collected PPO rollout. |
| `snapshots_per_agent_slot_per_rollout` | 1 | Bound head work without another environment rollout. |
| `temporal_selection` | stratified eligible timesteps | One sample is uniform; larger counts spread distinct samples across the rollout. |
| `eligibility` | active, target-known, valid reference | Includes finder and informed peers, regardless of eventual delivery/success. |
| `head_hidden_sizes` | [256, 128] | Match the current probe capacity initially; a linear-head ablation can follow. |
| `head_normalization` | none | Do not fit offline dataset-wide feature statistics on a drifting online GRU. |
| `head_learning_rate` | 0.0003 | Independent of policy lambda; start at the policy's default LR. |
| `lambda_target` | 0.03 | A small unvalidated starting coefficient included in the planned sweep. |
| `head_warmup_updates` | 10 updates with eligible data | Let the new decoder learn before it shapes the actor. |
| `lambda_ramp_updates` | 100 updates with eligible data | Increase actor pressure gradually from zero. |
| `positive_cell_weight` | 1 | Preserve the current ordinary probabilistic BCE target. |
| `metric_threshold` | 0.8 | Thresholded overlap and exact-mask accuracy; probability overlap and BCE have no threshold. |

Save head parameters, optimizer state, schedule counters and independent auxiliary
sampling RNG state with checkpoints. Fresh actor/critic initialization must match
the baseline seed; constructing the auxiliary head must not consume their RNG
stream. Disabled mode should not allocate head/buffer state or alter the policy
rollout RNG. Empty eligible minibatches leave head optimizer state unchanged.

## Scientific questions and evaluation

- **Information preservation and communication:** target-known propagates from
  a direct observer through peer/base contact, so optimizing GRUs and learned
  messages can preserve and transmit the finder's route information. This is the
  working hypothesis; all active knowing agents remain eligible. The simulator's
  flag records provenance/contact, not whether the learned message actually
  contains a decodable route. Distinguish a route the finder experienced from a
  guarantee of the globally shortest reference path: unseen shortcuts or geometry
  that sets the canonical orientation may remain ambiguous. Treat that narrower
  limitation as a hypothesis to measure, not a reason to exclude informed agents.
- **Task alignment:** a single reference route may penalize representations
  compatible with other equally effective relay layouts. Better decoding is not
  sufficient evidence of better task performance.
- **Loss imbalance:** many outputs are off-path, including padded cells. Low BCE
  can come from background prediction. Inspect overlap and complete matches;
  positive weighting would be an explicit later ablation with probability effects.
- **Gradient strength:** measure auxiliary versus PPO actor-gradient norms and,
  occasionally, whether they point in conflicting directions. PPO's clipped
  surrogate does not itself constrain an auxiliary encoder update. Keep policy KL,
  clipping and task metrics visible. Do not choose lambda by comparing scalar BCE
  with the scalar PPO loss.
- **Information propagation:** the first prototype keeps existing recurrent/base
  memory boundaries. Longer burn-in/backpropagation or explicit message supervision
  are separate design changes, not automatic consequences of adding this head.

Use the new matched policy-only variant as the primary PPO baseline. Evaluate
the same held-out maps/seeds, task success, delivery, recruitment/decommission
behavior and route efficiency versus both environment steps and wall-clock time.
Record label construction time, optimizer overhead, memory, unique labeled
episodes, selected knowing-agent samples, and missing-reference counts.

Show path similarity and exact-mask accuracy for the online head. After training,
the user can run the existing offline probe suite manually on frozen baseline
and auxiliary checkpoints with the same evaluation budget. A jointly trained
head alone cannot establish a general gain in independently decodable information.

## Proposed 24-slot experiment matrix

Use seeds 456789 and 456790 for every row. Let K be snapshots_per_agent.

| Variant | Actor auxiliary lambda | K |
| --- | --- | --- |
| Clean policy-only baseline | Disabled | — |
| Weak pressure | 0.01 | 1 |
| More snapshots, weak pressure | 0.01 | 5 |
| Default candidate | 0.03 | 1 |
| More snapshots, default weight | 0.03 | 5 |
| Lambda sweep | 0.1 | 1 |
| More snapshots, moderate pressure | 0.1 | 5 |
| Strong pressure | 0.3 | 1 |
| More snapshots, strong pressure | 0.3 | 5 |
| Strong pressure | 1.0 | 1 |
| More snapshots, strongest pressure | 1.0 | 5 |
| Intermediate snapshot count | 0.1 | 2 |

This is 12 configurations x 2 seeds = 24 trials, assuming one GPU slot per trial;
actual concurrency remains subject to available slots/resources. This compares
K=1/5 at five lambda values from 0.01 to 1, with K=1/2/5 at lambda=0.1. It is an
exploratory subset of possible lambda/snapshot settings. The clean
baseline has no auxiliary head, label collection,
optimizer or diagnostics; enabled-with-zero-lambda is a different optional
diagnostic mode and is not the clean baseline. Keep head LR/capacity, selection rule, warm-up/ramp,
training budget and map-selection seed fixed across the sweep. The small and
large coefficients deliberately test negligible through potentially disruptive
policy pressure; none is an established safe/optimal scale. Record gradient norms
and KL so the results explain the range rather than only naming a winning loss.

This plan does not modify `cluster/manifests/chain-efficiency/fresh-sweep.json` or submit any experiments.

## Efficient JAX integration and memory handling

Reuse the actor's existing differentiable lax.scan replay. Before optimization,
select fixed-shape timestep indices/masks with shape (B,N,K), and prepare target
references/weights. Within the scan, capture only selected GRUs into a fixed
(B,N,K,H) carry/output buffer. Selection is linear/masked so its backward rule
need not save each intermediate snapshot-buffer value. Verify the compiled memory
behavior; do not assume XLA necessarily eliminates every temporary. Avoid emitting
a second full (T,B,N,H) GRU history or rerunning the actor just for the auxiliary
head. Run the head vectorized over the selected states after recurrent replay.

K is static per trial. Invalid slots are masked/padded, including agents with no
eligible state, so varying knowledge/lifetimes do not cause shape recompilation.
Pass lambda as a device scalar so its warm-up/ramp does not cause recompilation.
Cache a target per selected episode and map graph/frame data; do not reconstruct
paths per agent, snapshot or PPO epoch. Keep host label construction outside JIT
and perform no host callbacks/device synchronization inside the recurrent scan.

An efficient way to retain independent head learning with one ordinary backward
pass is to scale the auxiliary input gradient:

```python
h_aux = stop_gradient(h) + lambda_effective * (h - stop_gradient(h))
loss_for_autodiff = ppo_loss + masked_mean_bce(path_head(h_aux), targets)
```

Forward h_aux equals h; its derivative into the actor is multiplied by lambda,
while head gradients retain full strength. This is intentional gradient scaling:
the numerical objective above is not the lambda-weighted scalar one would display
as policy loss. Preserve the existing PPO loss stats and store raw auxiliary BCE
separately. Use separate optimizer/LR/clipping groups for head versus actor/critic;
head gradients must not consume the PPO clipping budget, especially during warm-up.
Validate actor-gradient scaling and unchanged head gradients for lambda=0/1 and
intermediate values. Extra gradient diagnostics may require an additional reverse
pass on their sampled minibatch, not a second actor replay for every ordinary step.

For the current T=50, E=6000, six minibatches, N=7 and H=256, each optimizer
minibatch contains B=1000 environments. Float32 tensor payload estimates are:

| K | Selected GRUs | Head hidden layers and logits |
| --- | --- | --- |
| 1 | 6.84 MiB | 13.59 MiB |
| 2 | 13.67 MiB | 27.18 MiB |
| 5 | 34.18 MiB | 67.96 MiB |

These are forward tensor sizes, not an estimate of total incremental peak VRAM:
backpropagation residuals, XLA workspaces and aliasing affect the actual peak.
Head weights are approximately 0.44 MiB. An unnecessarily duplicated full hidden
sequence alone would be 341.8 MiB. Lambda magnitude does not change these shapes;
K, replay implementation and diagnostics do. Exact cluster GPU headroom must be
measured before judging OOM likelihood. Allocator reservation shown by nvidia-smi
is not equivalent to live tensor memory.

Use compiled memory analysis and device-memory profiling on the actual cluster
GPU, including the first active auxiliary step, gradient diagnostics and normal
full validation. The planned baseline and a K=5 trial can provide the initial
implementation checks before releasing the remaining trials; no extra search
configuration is required. Preserve scalar/feature RNG streams and restore state
if a dry run executes optimizer logic. Prefer one common execution microbatch
factor across the full comparison, selected from the worst K=5 case.

If optimizer memory does not fit, split each original PPO minibatch into execution
microbatches (1000 -> 2x500 -> 4x250) and accumulate gradients. Apply clipping and
Adam once per original minibatch, keeping E=6000, T=50, six PPO minibatches and
three epochs. Preserve global active-sample denominators, episode/agent auxiliary
weights, value normalization and per-sequence RNG assignments. Naively averaging
microbatch-normalized gradients with different eligible counts is incorrect.
Numerical reduction differences are possible; loss/gradient equivalence is tested
in full precision. Production retains the existing matrix precision setting.

The legacy launcher policies can change num_envs/num_steps/num_minibatches.
The auxiliary sweep explicitly enables the new optimizer_microbatches policy:
it changes execution shapes only, retaining requested K, lambda and scientific
PPO settings. A failed worker exits and recovery starts a fresh process from
the last complete checkpoint, including both Adam states and the lambda counter.
The journal persists execution shapes. Work after that checkpoint is repeated;
recovery cannot guarantee success if even a single sequence cannot fit.

Handle OOM by phase: optimizer microbatching does not fix rollout or validation
allocation failures. Validation may lower its existing parallel_envs without
changing num_episodes/cases. Any required rollout chunking must preserve the full
effective E/horizon and per-environment RNGs. Selective rematerialization of the
scan body is another memory/compute trade-off if activation profiling justifies it.
Do not silently lower snapshots_per_agent, lambda or scientific PPO settings.

## Runtime expectation

Cluster overhead has not been benchmarked yet. With shared replay, K=5
multiplies the small head's sample work by five, not the complete PPO update.
Expected overhead is modest to low tens of percent when label caching and the
selected-state replay work well; this is an engineering expectation, not a timing
result or guaranteed bound. CPU reference-route construction, gradient diagnostics,
rematerialization or inefficient microbatching can dominate and exceed it.

Use existing phase timings and perf/sps to measure baseline, K=1 and K=5 after
compilation, separately recording startup/JIT and validation costs. Synchronize
timed device work. Include auxiliary-specific preparation/retries in reported
compute rather than hiding them outside comparison budgets. Keep the same
four-hour budget and plot task validation against both steps and elapsed time.

JAX implementation references: [fixed-shape scan](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html),
[gradient stopping](https://docs.jax.dev/en/latest/_autosummary/jax.lax.stop_gradient.html),
[gradient checkpointing](https://docs.jax.dev/en/latest/gradient-checkpointing.html),
[profiling](https://docs.jax.dev/en/latest/201/profiling.html), and
[Optax gradient accumulation](https://optax.readthedocs.io/en/latest/_collections/examples/gradient_accumulation.html).

## Implemented configuration

This block belongs inside the existing training configuration. The first field
disables the entire feature. Fixed prototype semantics are BCE over 125 cells,
active/knowing eligibility, current canonical target construction, and the
specified episode/agent balancing; these do not need extra configuration switches.

```yaml
training:
  geo_probe_auxiliary:
    enabled: false # Bypass head, labels, optimizer and diagnostics entirely.
    lambda_target: 0.03 # Target path-gradient weight entering the policy; varied per trial.
    metric_threshold: 0.8 # Thresholded similarity/exact mask only; soft overlap and BCE have no cutoff.
    gradient_diagnostics: true # Reuse training.diagnostics_every and one PPO minibatch.

    head:
      hidden_sizes: [256, 128]
      learning_rate: 0.0003 # Head learns independently of the actor's lambda.

    sampling:
      every_updates: 1 # Outer PPO rollout/update, not one minibatch step.
      snapshots_per_agent: 1 # Up to K distinct eligible steps per slot/current rollout.
      selection: stratified_eligible_steps # One random step from each time-ordered band.

    schedule:
      warmup_updates: 10 # PPO continues; auxiliary gradient into policy is zero.
      ramp_updates: 100 # Valid-supervision cycles 11..110 linearly reach lambda_target.
```

Head normalization is initially none; its changing GRU inputs must not use
offline dataset-wide feature statistics. Validation inherits all existing task
validation settings: episode count, cases/maps, seed, cadence and batching. There
is no auxiliary episode cap, alternate validation population or extra rollout.
The auxiliary metric uses eligible informed states from those same cases, with
fixed independent snapshot RNG and one selected snapshot per agent/episode.
Cache reference labels for the fixed missions. Keep eligibility/failure counts
in saved artifacts, but do not add an episodes-scored-percentage W&B chart.

For snapshots_per_agent=5, sort all eligible timesteps for an environment/agent
slot in the current PPO rollout. Split them into min(5, available_count)
nonempty, approximately equal-count contiguous bands and sample one timestep
uniformly from each band. Samples are distinct; fewer than five eligible steps
means use all available steps without duplication. One requested snapshot is
uniform over the complete eligible set. This is within the current 50-step
rollout, not retrospective full-lifetime sampling. Each selected frame retains
its own episode/target identity if the slot spans an auto-reset. Average samples
per agent/episode before averaging episodes, so K=5 does not multiply lambda by 5.

Let u count auxiliary rollout/update cycles that contain valid supervision,
including the current cycle. The effective weight is:
`lambda_target * clip((u - 10) / 100, 0, 1)` for the proposed defaults. Cycles
1..10 have zero actor auxiliary weight; cycle 11 uses 1% of lambda_target;
cycle 110 reaches the full target. Empty-supervision cycles do not advance u.
The same weight applies throughout that rollout's PPO epochs/minibatches.

No new decoder is fitted automatically during training or finalization. Fresh
independent probe fitting remains a manual post-training probe-suite operation.

## Minimal W&B diagnostics

All eight new time series use the `geo_probe_auxiliary/` prefix. Keep both
thresholded path similarity and probability-based path overlap initially:
`soft_similarity = 100 * 2 * sum(p * target) / (sum(p) + sum(target))`.
This gives credit to probability mass on actual path cells and penalizes missing
mass/extra mass. Correct negatives earn no credit. It is threshold-free and
differs from the thresholded path similarity in the offline report; name it
explicitly. It is a diagnostic, not a replacement for the existing BCE objective
or a claim of probability calibration. Thresholded path similarity and exact-mask
accuracy use the fixed configured cutoff. Average per snapshot, then agent, then
episode. Keeping both overlap curves lets the experiment establish which is more
useful; no automatic threshold optimization is introduced.

| Suffix after `geo_probe_auxiliary/` | Cadence | Interpretation |
| --- | --- | --- |
| `train/soft_path_similarity_pct` | Existing completed-episode window | Is probability overlap improving on training trajectories? |
| `train/path_similarity_pct` | Same existing completed-episode window | Is overlap of threshold-selected cells improving? |
| `validation/soft_path_similarity_pct` | Normal task validation | Does probability overlap transfer to the configured held-out cases? |
| `validation/path_similarity_pct` | Normal task validation | Does threshold-selected overlap transfer? Compare with soft overlap. |
| `validation/exact_path_accuracy_pct` | Normal task validation | How often is the entire selected-cell mask correct? |
| `data/supervised_samples_per_update` | Existing per-update W&B logging | Number of distinct valid selected frames in the current rollout; do not count PPO-epoch reuse. |
| `gradient/aux_to_ppo_ratio` | training.diagnostics_every; one existing minibatch | Norm of lambda-weighted path gradient divided by PPO actor-gradient norm on shared encoder/GRU/communication parameters. Measures raw gradient pressure. |
| `gradient/cosine_similarity` | Same sampled measurement | Alignment of unweighted path and PPO gradients on those same parameters: positive agrees, negative opposes. |

Training score smoothing reuses the main completed-episode window. It currently
contains training.num_envs episodes (6,000 for G03 CLU), appended/evicted in the
same completion order as ordinary training success/return. Add optional auxiliary
summaries to those episode records rather than creating an independent deque
containing only episodes with probe samples. Accumulate per-agent sums/counts
for each live episode, finalize at completion and attach the score in that same
window; these live tallies are not another history window. Include auxiliary
predictions from the last PPO epoch once per selected frame, not every epoch.
Episodes without valid samples stay in the shared window with an unavailable
auxiliary field; exclude that field from the score mean without filling from
older episodes. Reuse the existing full-window publication rule.

Do not reuse the required-agent 500-episode category windows or adaptive
per-map/per-cell sampling windows: those have different populations and purposes.
No new window length or training-metric logging frequency is introduced.
W&B currently receives update-level logs; terminal cadence remains controlled by
logging.terminal_logging_frequency. That terminal setting is not a W&B interval.

Validation summarizes the current full validation call with the frozen current
policy/head, without a sliding history across checkpoints. Gradient metrics and
supervised-sample counts describe current optimizer inputs and should not be
smoothed over completed episodes. Use ordinary W&B display smoothing if desired.
No optimizer-update sliding window is needed.

Gradient measurements use the same existing minibatch sequences and model state for both
losses, before clipping/Adam. Exclude the path head, critic and downstream action
heads from both norms; they are not the shared representation being regularized.
Use a separate diagnostic RNG namespace. These are sampled gradient estimates,
not a measurement of the exact optimizer-step contribution. Omit undefined
cosines or ratios when their required norms/data are absent, rather than
inventing zero values. A genuine zero weighted gradient during warm-up has ratio
zero when the PPO denominator is valid. Gradient comparison is new work not
already supplied by PPO KL/value diagnostics; gradient_diagnostics toggles only
this extra computation. Reuse training.diagnostics_every (currently 1) and one
existing PPO minibatch, without an additional frequency or environment cap.
Implement by reusing the normal gradient computation where possible and measure
the added autodiff cost; sharing settings does not make that cost zero.

The effective lambda schedule is checkpointed, but nominal lambda, head settings,
thresholds, flags and seeds belong in W&B config, not constant time series.
Raw BCE stays available in local optimizer histories/debugging. Do not add weighted
BCE or total-combined-loss charts as proxies for auxiliary influence; their
arbitrary scales do not answer that question. Keep per-cell/per-agent charts and
gradient-tensor histograms out of the initial diagnostics.

One W&B run-summary value, `geo_probe_auxiliary/data/missing_reference_pct`, records
the percentage of distinct requested episode labels that were unavailable. Keep
counts and exact failures in local artifacts, emitting an event when failures
occur; no permanently zero missing-route chart is needed. Empty score populations
are unavailable, not zero decoding accuracy. Keep counts in artifacts rather
than publishing the removed coverage chart.

Reuse existing `validation/success_rate`, `train/success_rate`, PPO KL/clip metrics,
and `perf/sps`, `perf/global_step`, `perf/training_seconds`. Compare the two seeds
per variant using mean/spread of held-out task success against both environment
steps and wall time. First judge whether lambda improves the task, then use path
quality and gradient pressure/alignment to explain gains or damage. Keep task
checkpoint selection and early-stop criteria unchanged. Do not duplicate those
metrics under the auxiliary namespace or attempt to infer fused-backward overhead
from a timer that cannot isolate it.

Use validation results for comparing/selecting lambda. Keep a separate untouched
final-test mission seed set (preferably additional maps) for the final comparison;
repeatedly inspected validation results are not an unbiased final test. Do not
silently treat a missing-informed-samples checkpoint as a zero decoding score.

Required implementation checks: disabled/zero-policy-lambda equivalence,
nonzero actor and head gradients with valid supervision, inactive/uninformed and
invalid-reference masking, post-information timing, auto-reset alignment, no
held-out optimization, label equivalence with the offline probe, save/resume
equivalence, and measured training cost. Cluster submissions remain user-operated.

## Research context

Joint representation learning with auxiliary tasks has precedent in
[UNREAL](https://arxiv.org/abs/1611.05397). Auxiliary tasks can also conflict with
the main objective; [Du et al.](https://arxiv.org/abs/1812.02224) study weighting
through gradient similarity. These support the design questions, not a claim
that path supervision will improve G03 CLU without a matched comparison.
