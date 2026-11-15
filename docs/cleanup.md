# G03 CLU cleanup record

**Current branch** `final-ablation-archive`.

## Branch roles and order

1. **`geo_memory`** is the original pre-cleanup branch from which both branches were created.
2. **`final-ablation-archive`** is the thesis ablation branch, created from `geo_memory`. Keep the implementations, run recipes and evidence needed to reproduce thesis-relevant comparisons. Review uncertain cases together before removing or restructuring them. Remove material once its irrelevance to the thesis is established.
3. **`final-cleanup`** is the maintained G03 CLU branch, also created from `geo_memory`. After the ablation archive is curated, merge it into `final-cleanup` and remove the ablation implementations, options and tooling that the maintained workflow does not use. Use this branch as the thesis reference for the final system.

Use `final-ablation-archive` for thesis ablations and `final-cleanup` for the final system. Preserve the ablation implementations on their branch.

## Experiment provenance

- A run's saved `config.yaml` records the resolved training, model, environment, reward, validation and inspection settings. Frozen map selections, validation cases, metrics and checkpoint step history provide dataset and result context where available.
- Cluster recipes can record intended sweep structure, seed pairing, checkpoint parentage and jobs that never produced a local config. Generated submission receipts and run tracking can record Determined experiment IDs, dependency images and exact submitted overrides. Keep the original source snapshot when it differs from the branch used for reproduction.
- Thesis run selection is open. Retain distinct experiments and any unique provenance while deciding which comparisons to include. Ignored local artifacts are stored separately from the branches.

See the [saved config writer](../src/swarmecho/training/train.py#L1264), [checkpoint step history](../src/swarmecho/training/checkpoints.py#L159), and [cluster run tracking](../cluster/tools/tracking.py#L30).

## Archive organization and tests

Group standalone experiment reports, recipes and reproduction scripts under `ablations/<feature>/` when that makes the evidence easier to find. Keep wired runtime code in its existing modules.

Keep the regular test suite small and focused on the training, evaluation and validation entry points. Put standalone research checks with the ablation they support when they are needed to reproduce or interpret evidence.

## Cleanup log

Record high-level changes that distinguish `final-cleanup` from `final-ablation-archive` after the merge.
