# MARL and training

The maintained trainer is `training/train.py`, using `PPOTrainer` from
`training/ppo.py`. Production training requires recurrent actor and critic
memory. A shared decentralized actor emits three pre-squash action components;
tanh produces the normalized forces supplied to the environment.

The agent-centric critic produces one value per agent. Its observation mode
attends over team observations. Its optional privileged mode uses compact
features assembled in `env/critic.py`, retaining the active-agent attention
path. The actor's inputs do not acquire privileged state.

## Recurrent communication

TarMAC exchanges addressed signatures and values subject to communication
visibility and the configured episode-step schedule. Recurrent state and base
memory are reset and masked according to episode and agent activity. PPO
replays the corresponding sequences.

## Optimization and checkpoints

Actor clipping, value clipping, entropy mode, value normalization, and
diagnostics are documented in [PPO controls](05_ppo_controls.md). The
`legacy` entropy option remains a supported objective; its name is not a
separate spatial runtime.

Checkpoint restoration validates architecture and behavioral metadata.
Observation settings, critic representation, and value units must remain
compatible. Run lifecycle helpers maintain output directories, schedules,
history, and checkpoint accounting. Curriculum stages inherit checkpoints;
multi-train runs choose reproducible independent seeds.

`training/mappo_trainer.py` remains because the packaged workflow validator
uses its generic PPO implementation for recurrent and feed-forward checks.
It is not a second environment. The production trainer continues to use
`ppo.py`.

See [commands](../02_guide/03_commands.md) for training, resume, evaluation,
and validation.
