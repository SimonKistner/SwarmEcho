# G01 final uniform comparison and G03 cluster preparation

> Historical report. For the current cluster PPO tuning conclusions, see [the four-wave G03 report](../g03_refinement_analysis_20261031/G03_refinement_report.html).

All nine runs finished their cooperative 30-minute budgets. These are final uniform-from-start G01 results, not the earlier virtual-curriculum comparison. The seed is shared; requested and effective PPO counts match for every run. Scores are episode success over the fixed full-task validation cases. Final updates/measurements can slightly overrun 1800 seconds.

| Rank | Envs / steps / epochs / minibatches | Final validation | Best validation | Training transitions | Reported SPS | W&B |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | 4000 / 50 / 4 / 5 | 57.6% | 57.6% | 40.60M | 22770 | [run](https://wandb.ai/ias_master_thesis_sk/swarmecho/runs/du14ddln) |
| 2 | 4000 / 100 / 4 / 5 | 45.4% | 45.4% | 41.60M | 23332 | [run](https://wandb.ai/ias_master_thesis_sk/swarmecho/runs/umgv82l4) |
| 3 | 16000 / 25 / 1 / 10 | 43.8% | 43.8% | 78.80M | 44147 | [run](https://wandb.ai/ias_master_thesis_sk/swarmecho/runs/1905ooh4) |
| 4 | 1000 / 50 / 12 / 5 | 42.0% | 42.0% | 8.25M | 4615 | [run](https://wandb.ai/ias_master_thesis_sk/swarmecho/runs/4rio7pq6) |
| 5 | 2000 / 50 / 12 / 5 | 33.4% | 33.4% | 11.40M | 6380 | [run](https://wandb.ai/ias_master_thesis_sk/swarmecho/runs/fr5y22nc) |
| 6 | 4000 / 50 / 8 / 5 | 27.0% | 36.6% | 25.60M | 14294 | [run](https://wandb.ai/ias_master_thesis_sk/swarmecho/runs/vntbd3lz) |
| 7 | 2000 / 50 / 8 / 5 | 15.6% | 19.8% | 15.70M | 8813 | [run](https://wandb.ai/ias_master_thesis_sk/swarmecho/runs/yl3dzuev) |
| 8 | 4000 / 100 / 4 / 20 | 6.8% | 7.4% | 22.00M | 12291 | [run](https://wandb.ai/ias_master_thesis_sk/swarmecho/runs/3lfukhsv) |
| 9 | 2000 / 100 / 4 / 5 | 0.0% | 0.0% | 27.60M | 15386 | [run](https://wandb.ai/ias_master_thesis_sk/swarmecho/runs/1o4wel0o) |

## Findings

- Changing only minibatches from 20 to 5 at 4000/100/4 improved final validation from 6.8% to 45.4%, while reported SPS increased from 12.3k to 23.3k. Fewer, larger optimizer steps were productive in this run.
- At 4000 envs, 4 epochs and 5 minibatches, a 50-step rollout beat 100 steps (57.6% vs 45.4%) at similar total experience (40.6M vs 41.6M). Faster policy feedback is a stronger explanation than raw sampling volume alone.
- Doubling epochs from 4 to 8 at 4000/50/5 reduced the final score to 27.0% and lowered throughput. More reuse is not monotonically better.
- The 16000/25/1/10 run reached 43.8% with 78.8M transitions. It remains relevant to diverse G03 maps despite poorer sample efficiency.
- The 1000/50/12/5 run achieved 42.0% with 8.25M transitions. Its sample efficiency is attractive, but transfer to thousands of maps remains uncertain.
- The 2000/100/4/5 run recorded zero training and validation success within this budget; 2000/50/8/5 finished at 15.6%. Low env count alone does not explain good learning.
- These observations select candidates for transfer; one seed and one static map do not establish an optimal G03 configuration.

## New G03 contract

- Dedicated source: `src/swarmecho/curriculum_config/levels/G/G03_CLU_random_buildings.yaml`; the local G03 level is unchanged.
- One continuous run, no wall-clock cutoff, up to 2B training transitions, rounded down to complete rollouts. Normal full-task early stopping remains enabled.
- Seven recruitment slots from the beginning, one initial active drone. No classic agent-count stages or checkpoint handoffs.
- Virtual target budget starts at two and progresses to seven. Training preserves both adaptive map sampling and per-map adaptive cell sampling.
- Inspection uses frozen held-out inspection maps, uniform target proposals conditional on the current virtual budget, and no adaptive map/cell success weights.
- Inspection progression uses the inherited local G03 intermediate criterion (75%, one qualifying inspection); full-task validation stopping uses 99%.
- Validation keeps 400 held-out maps, 10000 fixed uniform full-seven-agent cases, batches of 500, and never receives the virtual budget or adaptive weights.
- The bank retains 3600 training maps independently of live env count; map identities survive OOM retries/resume. Pool generation already uses an inter-process generation lock.
- Inspection/validation cadence stays at local G03 defaults: 100/50 PPO updates. No run manifest overrides scheduling or shared task settings.
- OOM recovery uses the shared 0.75 environment factor and two retries; requested/effective PPO settings are recorded.

## Eleven candidates

| Manifest | PPO tuple | Reason |
| --- | --- | --- |
| `clu-g03-balanced-diversity` | 6000/50/3/5 | Intermediate map diversity and three passes balance the strongest G01 regimes |
| `clu-g03-baseline` | 4000/100/4/20 | Local PPO reference |
| `clu-g03-diversity-controlled` | 8000/50/4/10 | More map diversity per rollout with the winner's recurrent minibatch shape and four passes |
| `clu-g03-g01-rank1` | 4000/50/4/5 | G01 uniform final validation 57.6% |
| `clu-g03-g01-rank2` | 4000/100/4/5 | G01 uniform final validation 45.4% |
| `clu-g03-g01-rank3` | 16000/25/1/10 | G01 uniform final validation 43.8% |
| `clu-g03-g01-rank4` | 1000/50/12/5 | G01 uniform final validation 42.0% |
| `clu-g03-g01-rank5` | 2000/50/12/5 | G01 uniform final validation 33.4% |
| `clu-g03-long-context-fresh` | 4000/100/2/10 | Longer recurrent context with two passes and the winner's total sample presentations per rollout |
| `clu-g03-short-moderate-reuse` | 8000/25/2/10 | Short policy feedback interval with two passes rather than the high-throughput candidate's single pass |
| `clu-g03-wide-fresh` | 8000/50/2/5 | Wide recurrent minibatches and two passes favor fresh random-map experience |

## Error addressed and verification

The pasted failure was configuration validation rejecting `final_artifacts_only` for random maps. Adding a separate YAML alone would not fix it. The code now suppresses periodic map-suite media and creates final checkpoint heatmaps/replays for the frozen inspection maps. No change was made to evaluation/validation frequency.

CPU checks exercised the real strict level loader for all eleven candidates and the exact previous failing configuration, global multi-map curriculum advancement, compact checkpoint save/resume, budget-only inspection, single-map compatibility, and full-seven validation without adaptive inputs. An integration check exercised the actual map-suite/validation control flow with GPU functions mocked, including final artifacts and full-task stopping. Submission validation, unique PPO-derived names and source-package inclusion also passed.

No JAX/GPU training or cluster submission was executed. Actual 4000-map accelerator memory fit and throughput remain runtime checks.

Baseline command: `.\cluster\submit.ps1 -Run clu-g03-baseline`

The ten-candidate recipes are preserved under `cluster/manifests/historical/g03/`.
