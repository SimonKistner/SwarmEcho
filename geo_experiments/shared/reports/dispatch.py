"""Report blocks selected explicitly by the frozen definition's settings."""


def render(run_dir, manifest, arrays, *, dpi=None):
    renderer = manifest["config"]["report"]["renderer"]
    if renderer == "path_mask":
        from geo_experiments.shared.reports.path_mask import generate_report
        return generate_report(run_dir, manifest, arrays, dpi=dpi)
    if renderer == "location":
        from geo_experiments.shared.reports.location import generate_target_report
        return generate_target_report(run_dir, manifest, arrays, dpi=dpi)
    raise ValueError(f"Report block {renderer!r} is not implemented.")
