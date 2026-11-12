"""Discover data-only definitions; active definitions all use the shared runner."""
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

from geo_experiments.shared.definitions import Definition


@dataclass(frozen=True)
class Probe:
    path: Path

    def definition(self):
        result = Definition(OmegaConf.to_container(OmegaConf.load(self.path), resolve=True))
        result.validate()
        return result

    def settings(self):
        return self.definition().runtime_config()

    def adapter(self):
        from geo_experiments.shared import runner
        return runner


DEFINITIONS = {}
for path in sorted((Path(__file__).parent / "probes").rglob("probe.yaml")):
    probe = Probe(path)
    name = probe.definition().data["id"]
    if name in DEFINITIONS:
        raise ValueError(f"Duplicate probe definition {name}")
    DEFINITIONS[name] = probe
PROBES = {name: probe for name, probe in DEFINITIONS.items() if probe.definition().data["status"] == "active"}


def get_probe(name):
    try:
        return DEFINITIONS[name]
    except KeyError:
        raise ValueError(f"Unknown probe {name!r}; definitions: {', '.join(DEFINITIONS)}.") from None


def archived_probe():
    from geo_experiments.archive.target_location import adapter
    class Archived:
        def adapter(self):
            return adapter
    return Archived()


def job_probe(job):
    if "definition" not in job["config"]:
        return archived_probe()
    # A frozen plan does not depend on the current definition files or registry.
    class Frozen:
        def adapter(self):
            from geo_experiments.shared import runner
            return runner
    return Frozen()
