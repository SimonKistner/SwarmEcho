# Memory T-Maze Preview

This diagnostic setup renders `memory_t_maze_8`, an eight-agent memory task:

- eight isolated T corridors in one static map;
- one fixed drone spawn per T;
- one fixed per-agent target per T;
- one fixed per-agent anti-target in the opposite branch;
- four left-cued and four right-cued branches;
- communication effectively disabled;
- normal SwarmEcho physics, reward, and renderer paths;
- a MEM_T8-only observation mask that removes non-local leak channels.

Each corridor is exactly `visual_radius * 2` wide, so one traversal reveals the corridor contents. The episode is intentionally short enough that an agent can commit to only one branch. Finding the target is therefore the positive memory outcome; seeing the anti-target is the wrong-branch outcome and gives `-finder_bonus` once to that drone.

The level sets `env.mem_test_mask_nonlocal_obs: true`. This keeps the normal observation dimension but zeros channels that would identify the fixed T location or bypass memory: base-relative vector, base/target connectivity flags, target-known and target-relative vector, local coverage probes, and base/target-connected teammate radar. Local velocity and wall radar remain available so the agent can navigate by geometry and the intended cue.

The preview command renders a simulated rollout of the memory layout:

```bash
uv run python src/visualize/render_preview.py memory_t_maze_8 --mode video --level MEM_T8_memory
```

The training level is `src/curriculum_config/levels/MEM_T8_memory.yaml`.

For this level, `train/target_found_rate` is averaged from the per-agent found fraction (`0/8` through `8/8`) instead of the normal single-target "found at least once" binary.
