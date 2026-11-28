# MARL and training

The maintained trainer is `training/train.py`, using `PPOTrainer` from
`training/ppo.py`. Production training uses a GRU actor and a recurrent
critic. A shared decentralized actor emits three pre-squash action components;
tanh produces the normalized forces supplied to the environment.
With adaptive size enabled, a fourth discrete vote head shares the policy
trunk and PPO objective; the vote is never tanh-squashed. The base collects
majority approval through the communication graph. Per-drone lifetime
boundaries reset recurrent state and stop return bootstrapping. See the
[adaptive swarm reference](06_adaptive_swarm_size.md).

The agent-centric critic produces one value per agent and attends over team
observations. `network.critic_type` currently accepts only `observation`.

## Recurrent communication

TarMAC exchanges addressed signatures and values subject to communication
visibility and the configured episode-step schedule. Recurrent state and base
memory are reset and masked according to episode and agent activity. PPO
replays the corresponding sequences.

By default, each actor uses its previous hidden state both as its GRU carry
and to query incoming messages. Set `network.disable_actor_memory_carry=true`
to start the GRU from zero at every decision. With TarMAC enabled, the
current observation embedding supplies the attention query. At each step,
the actor weights the previous step's accessible sender values, then uses
that context and its current observation to produce its action and its next
signature and value. An eligible receiver can also attend to the token stored
at base, which may be older. With learned messages disabled, the actor
computes its action from its current observation alone. The critic retains
its recurrent state in both arms.

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
