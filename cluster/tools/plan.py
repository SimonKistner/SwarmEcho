"""Expand one editable cluster config into independent experiment manifests.

Standard library only; no training imports or cluster calls.
"""
from __future__ import annotations

import re


WORKSPACE_SLOT_LIMIT = 24


def merge_overrides(*layers):
    """A named variation replaces matching common keys; reject duplicates within a layer."""
    result = {}
    for layer in layers:
        if not isinstance(layer, list):
            raise ValueError("overrides and variation values must be lists of key=value strings.")
        seen = set()
        for value in layer:
            if not isinstance(value, str) or "=" not in value or "\n" in value or "\r" in value:
                raise ValueError("Overrides must be single-line key=value strings.")
            key = value.split("=", 1)[0]
            if key in seen:
                raise ValueError("Duplicated override: " + key)
            seen.add(key)
            result[key] = value
    return list(result.values())


def level_seed(root, level):
    """Read the explicit training seed from the same block YAML used by our levels."""
    paths = list((root / "src/swarmecho/curriculum_config/levels").rglob(level + ".yaml"))
    if len(paths) != 1:
        raise ValueError("Expected one source YAML for level " + level)
    in_training = False
    for line in paths[0].read_text(encoding="utf-8-sig").splitlines():
        content = line.split("#", 1)[0].rstrip()
        if not content.strip():
            continue
        if not content[0].isspace():
            in_training = content == "training:"
        elif in_training:
            match = re.fullmatch(r"  seed:\s*([0-9][0-9_]*)\s*", content)
            if match:
                return int(match.group(1).replace("_", ""))
    raise ValueError("Multiple seeds require an explicit training.seed in the level YAML, "
                     "or an explicit seeds list in the cluster config.")


def level_logging_value(root, level, key, default):
    """Read a logging scalar without importing YAML or training."""
    path = next((root / "src/swarmecho/curriculum_config/levels").rglob(level + ".yaml"))
    in_logging = False
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        content = line.split("#", 1)[0].rstrip()
        if not content.strip():
            continue
        if not content[0].isspace():
            in_logging = content == "logging:"
        elif in_logging:
            match = re.fullmatch(r"  " + re.escape(key) + r":\s*(.*?)\s*", content)
            if match:
                return match.group(1).strip("\"'")
    return default


def level_timestamp_postfix(root, level):
    """Use the level's timestamp setting for cluster names."""
    return level_logging_value(root, level, "use_timestamp_postfix", "false").lower() in {"true", "yes", "on"}


def level_run_name(root, level, overrides):
    """Resolve the level's configured run name, including its PPO label."""
    name = level_logging_value(root, level, "run_name", level.lower())
    if "{ppo}" in name:
        from swarmecho.core.run_names import ppo_name, submission_ppo_parameters

        path = next((root / "src/swarmecho/curriculum_config/levels").rglob(level + ".yaml"))
        name = name.replace("{ppo}", ppo_name(submission_ppo_parameters(path, overrides)))
    return name


def expand_config(config, root, stamp):
    """Return (name, run) pairs; a default single run has no training overrides."""
    allowed = {"level", "seeds", "wandb_mode", "overrides", "variations",
               "name", "description", "wandb_group", "run_postfix", "exclude_trials"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("Unknown cluster config keys; expected: " + ", ".join(sorted(allowed)))
    level = config.get("level")
    if not isinstance(level, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", level):
        raise ValueError("level must be the exact source YAML name, without its extension.")
    name = config.get("name")
    if "name" in config and (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", name)):
        raise ValueError("name must contain 1-100 letters, numbers, underscores or hyphens.")
    seeds = config.get("seeds", 1)
    inherit_seed = type(seeds) is int and seeds == 1
    if type(seeds) is int:
        if not 1 <= seeds <= WORKSPACE_SLOT_LIMIT:
            raise ValueError("seeds must be between 1 and the 24-slot workspace limit.")
        # Preserve the level seed exactly, without adding an override, for the default run.
        if inherit_seed:
            seeds = [None]
        else:
            first = level_seed(root, level)
            seeds = list(range(first, first + seeds))
    elif not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a positive count or a nonempty list of explicit integers.")
    if not inherit_seed:
        if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
            raise ValueError("Explicit/generated seeds must be integers between 0 and 2**32 - 1.")
        if len(set(seeds)) != len(seeds):
            raise ValueError("Seeds must be distinct.")
    variations = config.get("variations", {})
    if not isinstance(variations, dict):
        raise ValueError("variations must map descriptive names to override lists.")
    common = merge_overrides(config.get("overrides", []))
    postfix = config.get("run_postfix", "")
    use_timestamp = level_timestamp_postfix(root, level)
    plans = []
    exclusions = config.get("exclude_trials",[])
    if not isinstance(exclusions,list) or any(not isinstance(row,dict) or set(row)!={"variation","seed"}
            or not isinstance(row["variation"],str) or type(row["seed"]) is not int for row in exclusions):
        raise ValueError("exclude_trials must contain {variation: name, seed: integer} records.")
    excluded = {(row["variation"],row["seed"]) for row in exclusions}
    if len(excluded)!=len(exclusions):
        raise ValueError("exclude_trials contains duplicates.")
    matched = set()
    for variation, changes in (variations or {None: []}).items():
        if variation is not None and (not isinstance(variation, str)
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,60}", variation)):
            raise ValueError("Variation names must contain 1-60 letters, numbers, underscores or hyphens.")
        overrides = merge_overrides(common, changes)
        base_name = name if name is not None else level_run_name(root, level, overrides)
        variant_name = base_name + ("_" + variation if variation is not None else "")
        group_name = base_name + ("_" + postfix if postfix else "")
        if use_timestamp:
            group_name += "-" + stamp
        if not inherit_seed and any(value.startswith("training.seed=") for value in overrides):
            raise ValueError("Use the seeds field instead of training.seed overrides for multi-seed/explicit-seed runs.")
        group = config.get("wandb_group", "CLU_" + group_name)
        if not isinstance(group, str):
            raise ValueError("wandb_group must be a string.")
        if variation is not None:
            group += "_" + variation
        for seed in seeds:
            if (variation,seed) in excluded:
                matched.add((variation,seed))
                continue
            run = {"description": config.get("description", "Cluster training: " + variant_name),
                   "level": level, "wandb_mode": config.get("wandb_mode", "online"),
                   "wandb_group": group, "run_postfix": postfix,
                   "overrides": overrides[:] if seed is None else merge_overrides(overrides, ["training.seed=" + str(seed)])}
            plans.append((variant_name + ("_seed" + str(seed) if seed is not None else ""), run))
    if matched != excluded:
        raise ValueError("exclude_trials refers to a variation/seed absent from this batch.")
    if not plans:
        raise ValueError("The cluster config excludes every trial.")
    return plans


def check_capacity(plans, slots_per_trial):
    """Refuse batches that cannot fit even in an otherwise empty workspace."""
    requested = len(plans) * slots_per_trial
    if requested > WORKSPACE_SLOT_LIMIT:
        raise ValueError("Batch requests {} GPU slots; workspace limit is 24. "
                         "Split the config into smaller batches and account for occupied slots.".format(requested))
