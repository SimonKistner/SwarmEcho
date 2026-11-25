# Historical curriculum and path-feedback experiments

## Removed progression rules

The reverse target curriculum began with targets requiring the full fleet and progressively admitted easier groups. Its gate used validation success on the exact current required-agent group, with 75% as the sweep threshold. The old selector also supported configurable starting budgets. The current [forward toggle and full-budget setting](../03_marl_and_training.md#forward-target-curriculum) define the supported alternatives now.

Classic agent-count orchestration instead trained stages with different fleet populations. It changed the available slots between stages and carried compatible policy state forward. That differs from changing target difficulty while keeping all configured slots available. Earlier G04 plans used larger 4 × 4 × 4 buildings, five-to-eight-agent stages and 7 × 7 × 7 path-mask heads. The dedicated population-stage orchestration was retired. General user-defined sequential level curricula remain documented in the training reference.

An older path-difficulty curriculum applied soft or hard gates while introducing wall/path complexity. It preceded the current relay-estimated target-budget progression. The historical C corridor suite similarly provided manually structured straight and turning relay tasks. Those presets and their generator were retired.

## Automatic-deployment sweep

The October/November 2026 design crossed no, forward and reverse target progression with auxiliary lambda values 0, 0.03, 0.3 and 1. Its proposed 21 runs used one seed for each zero-lambda arm and two for each nonzero arm. Lambda zero was a policy-gradient control with detached decoder fitting.

### One-map replay comparison, 3 November 2026

The following six completed runs supplied the original comparison. All use training seed 456789. The saved rates come from best-checkpoint recovery metadata on the original 400-map validation task.

| Curriculum | Auxiliary lambda | Experiment | Best update | Saved validation success |
| --- | ---: | ---: | ---: | ---: |
| None | 0 | 55374 | 2100 | 77.87% |
| None | 0.3 | 55377 | 2300 | 79.23% |
| Forward | 0 | 55381 | 1650 | 87.42% |
| Forward | 0.3 | 55384 | 1800 | 86.90% |
| Reverse | 0 | 55409 | 2200 | 36.14% |
| Reverse | 0.3 | 55418 | 2300 | 63.09% |

Reverse auxiliary entries use completed `curriculum_v3_cachefix` replacements. Earlier failed or cancelled attempts were excluded. The six corresponding `outputs/CLU_G03_auto7_*/` folders retained best checkpoints, resolved configs and a first frozen inspection map.

The prepared one-map protocol selected `random_eval_0000`, checked identical map bytes across runs, and requested 1,000 independent full-task targets per run with inspection seed 456789. The map SHA256 was `8b44e8feda9dddd3aad2c2fdf9d4f37f6c8ad0481fb8cc6c30becf32af635297`. Required-agent replay selection included failures. The source helper is [cluster/eval_g03_curriculum.py](../../cluster/eval_g03_curriculum.py). Its original preparation record established retrieval and input checks, with GPU evaluation still pending at that time. The table above records the separate full-validation scores.

The original [curriculum/auxiliary report](G03_curriculum_auxiliary_report_20261103.pdf) is retained with this comparison.

## Continued runs and best-checkpoint snapshots

Follow-up efficiency and retirement experiments branched from earlier checkpoints, using chain-efficiency coefficient 25 and surplus-decommission reward 3,000 in selected arms. Their cumulative step counts include parent training. Continuation seed, initial checkpoint, inherited optimizer state and additional budget all matter for attribution.

The 4 November snapshot identified four selected best checkpoints.

| Historical label | Continuation experiment | Best-checkpoint source experiment | Best update | Saved validation success |
| --- | ---: | ---: | ---: | ---: |
| Forward, auxiliary 0.3, efficiency 25 | 55559 | 55489 | 1450 | 95.80% |
| Reverse, auxiliary 0.3, efficiency 25 | 55521 | 55521 | 1700 | 89.68% |
| Reverse, auxiliary 0.3 | 55557 | 55518 | 5950 | 90.46% |
| Forward, auxiliary 0.3 | 55555 | 55555 | 5650 | 95.41% |

These are selected checkpoint snapshots from different continuation histories. Their recorded best updates and rates do not by themselves establish matched training budgets or uncertainty across policy seeds. The retained cluster manifests and submission receipts identify the actual parent/source chains. The [current experimental records](../04_validation_evaluation_and_inspection.md#experimental-records) separately identify the selected historical best and its converted checkpoint.

## Previous-step path-prediction feedback

An experimental actor consumed its own previous path prediction as an observation extension. The previous actor state produced a sigmoid path mask, gated to zero while inactive or uninformed. Stored feedback was detached across the observation boundary. In the 125-cell setup, actor width increased from the historical 141 inputs to 266, while critic input width stayed 141. A larger 343-cell head produced a 484-input actor in the corresponding larger-map design.

The fresh-run design crossed forward/no target curriculum with feedback off/on and two seeds. It retained the auxiliary decoder and selected efficiency/retirement rewards, allowing the feedback input itself to be studied. The old notes describe the design and prepared launches. Outcome claims need the associated completed run records. Previous-probe feedback is absent from the current actor and configuration.
