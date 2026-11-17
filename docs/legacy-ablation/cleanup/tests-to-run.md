# Tests to run

## Shared setup

Use the maintained [CLU G03 level](../../../src/swarmecho/curriculum_config/levels/G/G03_CLU_random_buildings.yaml) as the baseline. Both observation options are [pending ablation results](removal-candidates.md#pending-ablation-results). Preserve the source revision and resolved configs with the results.

Train each arm from fresh weights. Pair the training seeds across arms, and use the same compiled map bank, 3,600 training maps, 400 held-out maps, and frozen validation cases. Keep the remaining G03 settings fixed. One fresh baseline per seed can serve both studies. Use distinct run names and report results at matched environment-step counts.

| Arm | `env.observe_base_population` | `env.radar_mode` | Observations per drone |
| --- | --- | --- | --- |
| G03 baseline | `true` | `legacy` | 141 |
| Population off | `false` | `legacy` | 140 |
| Counting radar | `true` | `band_sep_counting` | 333 |

## Population channel on vs. off

Compare the baseline with `env.observe_base_population=false`. The enabled input gives each active drone connected to the base the total active fleet size divided by 10. Leave staggered deployment, permanent decommissioning, the decommission-zone and chain-contributor inputs, and the disabled base vote-ratio channel as in G03.

Compare held-out chain-holding success and training success with surplus cleanup. Inspect decommission counts and success by required-agent group to see where removing fleet-size feedback changes behavior.

## Legacy radar vs. band-separated counting radar

Compare the baseline with `env.radar_mode=band_sep_counting`. Keep `env.drone_radar_bins=32`, `env.distance_bands=3`, `env.radar_count_cap=3`, and the 32 Fibonacci wall rays. Legacy radar encodes the strongest neighbor proximity in each direction for each of its three drone channels. Counting radar bins neighbors by direction and distance, then caps and normalizes the counts for the same three groups.

Compare held-out chain-holding success, success by required-agent group, and environment steps per second. The comparison measures the full radar representation change, including the larger input, so record observation width and throughput alongside success.

## Record results

Save the resolved config, map selection, validation cases, and per-seed validation curves for each arm. Report paired differences against the baseline at the same environment-step counts. Record any early stopping or execution changes that affect the comparison.
