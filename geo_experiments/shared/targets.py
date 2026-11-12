"""Target construction and output encoders selected by probe definitions."""
from geo_experiments.shared.cell_labels import suite_geometry, position_labels


def build_target(config, dataset, graph):
    target = config["definition"]["target"]
    if target["kind"] != "target_position":
        raise ValueError(f"Target builder is not implemented: {target['kind']}")
    return position_labels(dataset, graph, position_key=target["field"], task=target["task"])


def encode_output(config, labels, graph):
    """Return supervised labels and the saved output vocabulary for this block."""
    encoding = config["definition"]["output"]["encoding"]
    if encoding != "categorical_coordinates":
        raise ValueError(f"Output encoder is not implemented: {encoding}")
    return labels["coordinate_cell"], graph["coordinate_values"]


