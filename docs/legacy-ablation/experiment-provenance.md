# Experiments and provenance

Use the [maintained G03 guides](../01_training.md) for the current system. Use each experiment’s resolved config, source snapshot, map selection and metrics for claims about that experiment. The run links point into the local `outputs/` archive. Provide those artifacts alongside the source for a handoff to another checkout.

## Five-seed cleaned baseline

The W&B group `CLU_G03_baseline_val_50` corresponds to Determined experiments **57005–57009**. The local receipts identify the submitted G03 level, the seed-specific map pool setting, the use of the training seed for map selection, and validation every 50 updates. The local run folders currently contain submission receipts. Use the original logged curves and resolved configs for numerical results and retain per-seed map and validation case records for comparisons.

| Seed | Determined experiment | Local submission receipt |
| --- | ---: | --- |
| 456789 | 57005 | [Receipt](../../outputs/CLU_G03_baseline_val_50_seed456789/cluster_run.json) |
| 456790 | 57006 | [Receipt](../../outputs/CLU_G03_baseline_val_50_seed456790/cluster_run.json) |
| 456791 | 57007 | [Receipt](../../outputs/CLU_G03_baseline_val_50_seed456791/cluster_run.json) |
| 456792 | 57008 | [Receipt](../../outputs/CLU_G03_baseline_val_50_seed456792/cluster_run.json) |
| 456793 | 57009 | [Receipt](../../outputs/CLU_G03_baseline_val_50_seed456793/cluster_run.json) |

## Historical best checkpoint

Determined experiment **55489**, `CLU_G03_forward_aux_l0p3_eff25_seed456789_decom3000_branch_v3`, reached **95.8% chain-holding validation success at 1.191 billion cumulative environment steps**. The step count includes ancestor stages. This run branched from `ckpt_002520`. Its resolved config and frozen bank selection describe the historical setup.

The original checkpoint expects 141 observation values. The separate `best_val_checkpoint_140obs` copy removes the always-zero vote input from both encoder matrices and their optimizer moments. The current loader restored the converted model and optimizer on CPU, and a recurrent comparison checked 16 inference steps. The recorded 95.8% comes from the historical validation run.

[Resolved config](../../outputs/000_best_run_so_far/config.yaml) · [Submission provenance](../../outputs/000_best_run_so_far/provenance/cluster_run.json) · [Validation state](../../outputs/000_best_run_so_far/checkpoints/best_val_checkpoint/validation_state.json) · [Step history](../../outputs/000_best_run_so_far/checkpoints/best_val_checkpoint/step_history.json) · [Converted checkpoint report](../../outputs/000_best_run_so_far/checkpoints/best_val_checkpoint_140obs/checkpoint_conversion.json)

## Pending ablations

| Arm | Population feedback | Radar | Inputs per drone |
| --- | --- | --- | ---: |
| G03 baseline | On | Legacy | 140 |
| Population off | Off | Legacy | 139 |
| Counting radar | On | Band-separated counts | 332 |

For the population arm, set `env.observe_base_population=false`. For the radar arm, set `env.radar_mode=band_sep_counting` and keep the G03 settings of 32 directional bins, three distance bands and a count cap of three. Compare fresh runs with paired seeds under a shared map assignment and fixed validation cases. Report results at matched environment-step counts, including validation success by required-agent count. Record throughput when changing radar width.

For older experiments and their implementation context, use the [legacy ablation index](README.md).
