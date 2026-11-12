"""Readout axes and friendly labels; input choice is separate from target choice."""
from geo_experiments.shared.inputs import input_names

SOURCES = ("finder_gru", "finder_message", "base_message")
TASKS = ("target_cell", "target_storey")
SOURCE_LABELS = {"finder_gru": "Finder GRU", "finder_message": "Finder message",
                 "base_message": "Base stored message", "agent_gru": "Agent GRU", "agent_message": "Agent message"}


def source_label(source):
    return SOURCE_LABELS.get(source, source.replace("_", " ").title())


def experiment_axes(config):
    if "definition" in config:
        return (config["definition"]["target"].get("task", "path"),), input_names(config)
    if config.get("experiment") == "own_cell":
        return ("own_cell",), ("agent_gru", "agent_message")
    return (TASKS if config.get("protocol", {}).get("target_storey", True) else ("target_cell",)), SOURCES
