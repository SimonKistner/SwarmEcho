#!/usr/bin/env bash
# Run manually in WSL. This script does not launch automatically during analysis.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$REPO_ROOT/pyproject.toml" || ! -d "$REPO_ROOT/src/swarmecho" ]]; do
  if [[ "$REPO_ROOT" == / ]]; then
    echo "Cannot locate the SwarmEcho repository." >&2
    exit 1
  fi
  REPO_ROOT="$(dirname -- "$REPO_ROOT")"
done
cd "$REPO_ROOT"
RUN="outputs/CLU_G03_aux_l0p3_k5_seed456789_geo_aux_v1"
uv run swarmecho-evaluate \
  checkpoint="$RUN/checkpoints/ckpt_001650" \
  level="$RUN/inspection_level.yaml" \
  mode=parallel \
  replay_after=true replay_by_required_agents=true \
  inspection_evaluation.target_sampling=independent \
  inspection_evaluation.dynamic_max_geo_distance=true \
  inspection_evaluation.eval_parallel_envs=1000 \
  inspection_evaluation.map_suite_replay_limit=1 \
  eval_name=aux_peak_full_task
