"""PPO comparison labels shared by local runs and standard-library submission."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re


PPO_PARAMETERS = (
    ("num_envs", "numEnvs"),
    ("num_steps", "numSteps"),
    ("num_epochs", "numEpochs"),
    ("num_minibatches", "numMinibatches"),
)
BASELINE_LEVEL = Path(__file__).resolve().parents[1] / "curriculum_config/levels/G/G03_random_buildings.yaml"


def wandb_display_name(run_name):
    """Keep configuration/stage labels, omitting automatic run-identity suffixes."""
    name = re.sub(r"-\d{8}T\d{6}Z(?:-[0-9a-f]{8})?(?=_|$)", "", run_name)
    return re.sub(r"_\d{8}_\d{6}(?=_|$)", "", name)


def _positive_integer(value, label):
    if isinstance(value, str) and re.fullmatch(r"\+?[0-9]+(?:_[0-9]+)*", value.strip()):
        value = int(value.strip().replace("_", ""))
    if type(value) is not int or value < 1:
        raise ValueError(label + " must be a positive integer for PPO run naming.")
    return value


def read_ppo_parameters(path):
    """Read four explicit training integers without importing YAML or training.

    Submission stays Python 3.7+/stdlib-only. Reject unsupported YAML forms
    rather than guessing a label; these levels use ordinary block mappings.
    """
    values = {}
    in_training = False
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        content = line.split("#", 1)[0].rstrip()
        if not content.strip():
            continue
        if not content[0].isspace():
            in_training = content == "training:"
            continue
        match = re.fullmatch(r"  (num_envs|num_steps|num_epochs|num_minibatches):\s*(.*?)\s*", content)
        if in_training and match:
            key, value = match.groups()
            if key in values:
                raise ValueError("Duplicated training." + key + " in " + str(path))
            values[key] = _positive_integer(value, "training." + key)
    missing = {key for key, _ in PPO_PARAMETERS} - set(values)
    if missing:
        raise ValueError("PPO naming requires explicit two-space-indented training integers in "
                         + str(path) + ": " + ", ".join(sorted(missing)))
    return values


def submission_ppo_parameters(path, overrides):
    values = read_ppo_parameters(path)
    for override in overrides:
        key, value = override.split("=", 1)
        if key == "training":
            raise ValueError("PPO naming requires individual training.* overrides, not a whole training mapping.")
        if key.startswith("training.") and key[9:] in values:
            values[key[9:]] = _positive_integer(value, key)
    return values


def ppo_name(training, baseline=None):
    """Name only differences from current G03, in a stable parameter order."""
    if baseline is None:
        baseline = read_ppo_parameters(BASELINE_LEVEL)
    labels = []
    for key, label in PPO_PARAMETERS:
        value = _positive_integer(training[key] if isinstance(training, Mapping)
                                  else getattr(training, key), "training." + key)
        if value == baseline[key]:
            continue
        number = str(value)
        for divisor, unit in ((1_000_000, "M"), (1_000, "k")):
            if value % divisor == 0:
                number = str(value // divisor) + unit
                break
        labels.append(label + number)
    return "_".join(labels) or "baseline"
