# G03 CLU cleanup record

**Current branch** `final-cleanup`.

## Branch roles and order

1. **`geo_memory`** is the original pre-cleanup branch from which both branches were created.
2. **`final-ablation-archive`** is the thesis ablation branch, created from `geo_memory`. Keep the implementations, run recipes and evidence needed to reproduce thesis-relevant comparisons. Review uncertain cases together before removing or restructuring them. Remove material once its irrelevance to the thesis is established.
3. **`final-cleanup`** is the maintained G03 CLU branch, also created from `geo_memory`. The curated `final-ablation-archive` has been merged here. This branch now removes the ablation implementations, options and tooling outside the maintained workflow. Use this branch as the thesis reference for the final system.

Use `final-ablation-archive` for thesis ablations and `final-cleanup` for the final system. Preserve the ablation implementations on their branch.

## Experiment provenance

- A run's saved `config.yaml` records the resolved training, model, environment, reward, validation and inspection settings. Frozen map selections, validation cases, metrics and checkpoint step history provide dataset and result context where available.
- Cluster recipes can record intended sweep structure, seed pairing, checkpoint parentage and jobs that never produced a local config. Generated submission receipts and run tracking can record Determined experiment IDs, dependency images and exact submitted overrides. Keep the original source snapshot when it differs from the branch used for reproduction.
- Thesis run selection is open. Retain distinct experiments and any unique provenance while deciding which comparisons to include. Ignored local artifacts are stored separately from the branches.

See the [saved config writer](../../src/swarmecho/training/train.py#L1264), [checkpoint step history](../../src/swarmecho/training/checkpoints.py#L159), and [cluster run tracking](../../cluster/tools/tracking.py#L30).

## Archive organization and tests

Group standalone experiment reports, recipes and reproduction scripts under `ablations/<feature>/` when that makes the evidence easier to find. Keep wired runtime code in its existing modules.

Keep the regular test suite small and focused on the training, evaluation and validation entry points. Put standalone research checks with the ablation they support when they are needed to reproduce or interpret evidence.

## Cleanup log

- Aligned CLU G03’s training settings with the saved winner and recorded its provenance.
- Restored fresh starts and automatic bank selection. Moved the implementation inventory to [Removal candidates](cleanup/removal-candidates.md).
- Recorded planned population and radar ablations in [Tests to run](cleanup/tests-to-run.md). Both options remain available pending the results.

## Winning CLU G03 recipe

Use the [saved resolved config](../../outputs/000_best_run_so_far/config.yaml) and [submitted run provenance](../../outputs/000_best_run_so_far/provenance/cluster_run.json) for the thesis. The reference is Determined experiment **55489**, `CLU_G03_forward_aux_l0p3_eff25_seed456789_decom3000_branch_v3`. The saved best checkpoint records **95.8% held-out chain-holding success at 1.191 billion cumulative environment steps**, including its two ancestor stages. Use [validation state](../../outputs/000_best_run_so_far/checkpoints/best_val_checkpoint/validation_state.json) and [step history](../../outputs/000_best_run_so_far/checkpoints/best_val_checkpoint/step_history.json) for the result and lineage.

The [CLU G03 level](../../src/swarmecho/curriculum_config/levels/G/G03_CLU_random_buildings.yaml) uses the winning run’s forward target curriculum, auxiliary weight `0.3`, efficiency reward `25`, surplus decommission reward `3000`, seed `456789`, and PPO layout of 6,000 environments × 50 steps, three epochs and six minibatches. A fresh launch initializes weights and uses a compatible generated map bank or builds one as needed. To continue an interrupted run, use its checkpoint with `ckpt_loading_mode=resume`. Use `branch` for a new experiment that inherits checkpoint weights and matching curriculum state.

The historical run loaded parent checkpoint `ckpt_002520` in `branch` mode and used the [frozen map selection](../../outputs/000_best_run_so_far/checkpoints/best_val_checkpoint/adaptive_map_selection.json). The local copy contains the 400 held-out map YAMLs. The original training maps, compiled records, pool metadata, and parent checkpoint are needed only for exact historical retraining. Preserve the winning checkpoint, validation cases, and submitted source under `cluster/.submissions` as provenance. Use the ancestor stages’ own records when describing their training.
