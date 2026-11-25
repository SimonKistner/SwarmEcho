# Completed observation ablations

## Outcome and baseline choice

The population-observation and counting-radar comparisons were completed. The user reported no significant differences from the normal baseline in either comparison.

The baseline retains base-population feedback and `radar_mode=max_proximity`. Population feedback is an established part of the baseline setup. Counting radar remains implemented for ablation reproduction.

| Reference arm | Base-population feedback | Radar | Inputs per drone |
| --- | --- | --- | ---: |
| G03 baseline | On | `max_proximity` | 140 |
| Population observation off | `observe_base_population=false` | `max_proximity` | 139 |
| Counting radar | On | `band_sep_counting`, three bands, cap three | 332 |

## Population observation

The input supplies the total active fleet size divided by 10 to active drones connected to the base, including through relays. Disconnected drones receive zero. The comparison disabled this input while retaining staggered deployment, decommissioning, zone membership and selected-chain membership. Its reported outcome did not motivate changing the baseline.

## Counting radar

Counting radar groups visible peers by angular bin, equal-width distance band and peer category. The three categories are all peers, target-connected peers and base-connected peers. Counts saturate at `radar_count_cap` and are divided by that cap. Wall readings precede the flattened directional, band and category counts.

The baseline maximum-proximity radar uses the strongest proximity per direction and category. The counting comparison changes the representation and its input width. Its reported outcome did not motivate replacing the simpler baseline.

To reproduce the documented counting arm, explicitly set `env.radar_mode=band_sep_counting`, `env.distance_bands=3` and `env.radar_count_cap=3`, with 32 peer bins and 32 wall rays. Legacy-radar level files omit the counting-only settings. The schema defaults to one distance band, with zero accepted as an alias for one. A model must be constructed with the selected observation width.

The documented comparison plan called for fresh models, paired seeds, shared map assignments, frozen validation cases and matched environment-step counts. Per-run resolved configurations and metric records establish the executed protocol and quantitative results.

[Radar configuration](../../src/swarmecho/core/config.py) · [Observation width and radar implementation](../../src/swarmecho/env/environment.py)
