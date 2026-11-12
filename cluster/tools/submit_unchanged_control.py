"""Submit the pre-copied original experiment; never repackage current source."""
from __future__ import annotations

from submit import CLUSTER, ROOT, read_json, submit


def main():
    name = "CLU_G03_forward_aux_l0p3_seed456789_exact55384_v1"
    output = CLUSTER / ".submissions" / name
    if not (output / "model/cluster/run.json").is_file():
        raise RuntimeError("The frozen control snapshot is missing: " + str(output))
    origin = read_json(output / "control_origin.json")
    if origin["source_experiment_id"] != 55384:
        raise RuntimeError("Unexpected control source experiment.")
    result = output / "result.json"
    if result.is_file() and read_json(result).get("experiment_id"):
        raise RuntimeError("This control already has an experiment ID; inspect its receipt before retrying.")
    if (ROOT / "outputs" / name / "cluster_run.json").exists():
        raise RuntimeError("This control is already registered; do not submit another copy.")
    summary = read_json(output / "summary.json")
    submit(summary["settings"], output / "experiment.yaml", output / "model", output)


if __name__ == "__main__":
    main()
