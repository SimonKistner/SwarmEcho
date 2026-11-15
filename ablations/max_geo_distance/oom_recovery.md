# Training OOM recovery

G01_static now uses:

```yaml
training:
  num_envs: 4000
  oom_recovery:
    enabled: true

evaluation:
  eval_parallel_envs: null
```

The global recovery default is false. The global evaluation default remains
4000; null explicitly follows the effective training count, including periodic,
robust, map-suite and final evaluation and parallel replay capture. A numeric
evaluation count remains fixed even if training shrinks.

Run the usual command:

```bash
uv run swarmecho-train level=G01_static logging.run_name=G01_agent_curriculum
```

## Recovery rules

The command keeps a CPU-only supervisor and runs each enabled stage in a fresh
worker process. Recognized allocation exceptions, including the supplied JAX
RESOURCE_EXHAUSTED / Out of memory message and Python MemoryError, cause the
worker to exit. The supervisor waits for its exit before starting another
worker with exactly 500 fewer environments. This releases the failed CUDA
context rather than trying to reuse a damaged allocator or live device arrays.
Other exceptions fail the command. An unreported process termination, including
SIGKILL, is not assumed to be an OOM.

The retry keeps the run directory, run name, agent cap and W&B identity:

* With a completed checkpoint from this stage attempt, load its newest update
  in resume mode. Restore the adaptive target controller as well as the model.
* Before this stage writes a completed checkpoint, reuse its incoming checkpoint
  and original mode. For a curriculum handoff this is the preceding stage's
  final checkpoint in branch mode. For the first stage without an incoming
  checkpoint this is a fresh initialization.
  If adaptive spawn history was already written before that first checkpoint,
  append a restart session with a fresh success controller; retain the prior
  history and restore any completed startup cell mask from
  `adaptive_spawn/eligibility.json`. If discovery did not finish, use the reduced
  reset batch for discovery instead. Do not infer adaptive success history from
  a mask or from unsaved updates.
* Further OOMs repeat the same rule. Stop with a clear error if another
  500-environment decrement would leave no environments. Recurrent minibatches
  must divide both the starting count and the fixed decrement; enabled recovery
  validates that requirement without silently changing the PPO minibatch count.

In-progress work since the chosen checkpoint is discarded. Resume retains the
existing project semantics: model/value-normalizer and adaptive controller
state are restored, while optimizer, live environments, recurrent rollout state
and rolling/evaluation-hold windows are rebuilt. It is not a bit-identical
continuation. Unflushed logging from the failed worker can be lost, and existing
W&B resume behavior applies when rolling back to an older checkpoint.

Each OOM writes an append-only `oom_recovery.jsonl` inside its run directory,
including the error, old/new counts, checkpoint and loading mode. A normal
manual resume of that directory retains its recorded reductions. The effective
count is also written in the successful run's config/metrics and in the agent
curriculum journal (`agent_curriculum.json` in the first stage's run folder)
and stage row. Later agent stages inherit this count instead
of retrying 4000. The sequential multi-level curriculum also carries reductions
forward when recovery is enabled for the next level.

## Checkpoint and step accounting

Training saves model tensors, the step history and adaptive controller into a
pending sibling directory, then publishes the completed directory. Only a
published checkpoint with `recovery_ready.json` from this supervisor invocation
is automatically selected. A partial save is never mistaken for a checkpoint,
and an unrelated old run with the same name is not selected accidentally.
An explicitly supplied incoming legacy model checkpoint remains usable for
branch/init loading when its model contract matches. An old maskless
relay/adaptive controller checkpoint requires a new branch rather than resume.

Step history now stores actual accumulated environment steps and the batch size
used at each save. Reducing the count does not reinterpret all earlier updates
as smaller batches. Resume computes the remaining stage budget from completed
steps while keeping update indices monotonic; as before, only full rollout
updates are taken. Curriculum branch history preserves those actual totals.

If OOM happens during final evaluation/artifact creation after the final
checkpoint was published, retry retains its final/mastery metadata and reruns
final artifact creation without starting additional training updates. Allocation
failures in the final-evaluation warning handlers propagate to recovery.

Fresh workers also mean in-memory geometry/JIT caches are rebuilt for each
recovery-enabled stage and retry. Existing persistent map-bank caches remain
available. With recovery disabled, the previous in-process training path is
used.

## Verification status

Implementation has been inspected as source only. No project Python, compile
check, training, forced OOM or GPU experiment was run by the coding agent.
Actual OOM recovery, GPU memory release and W&B continuation still need the
user's runtime verification. The next normal G01 run exercises this path;
there is no added test suite or artificial allocation failure.
