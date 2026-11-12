"""Declarative target/input/model/output contracts and their runtime compilation."""
import copy
import re
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Definition:
    data: dict

    def validate(self, *, runnable=False):
        d = self.data
        required = {"schema", "id", "status", "target", "inputs", "model", "output", "sampling", "settings"}
        if set(d) != required or d["schema"] != 1:
            raise ValueError(f"Probe definition requires schema=1 and sections {sorted(required)}.")
        if d["status"] not in ("active", "planned", "archived"):
            raise ValueError("Probe status must be active, planned or archived.")
        if not d["inputs"] or not d["target"].get("kind") or not d["output"].get("encoding"):
            raise ValueError("Probe definitions need inputs, a target and an output encoding.")
        if d["sampling"].get("split_unit") != "episode":
            raise ValueError("Samples from the same episode must remain in the same split.")
        if not runnable:
            return
        if d["status"] != "active":
            raise ValueError(f"Probe {d['id']} is {d['status']} and cannot be scheduled.")
        if d["output"]["encoding"] == "binary_cell_mask":
            validate_path_definition(d)
            return
        from geo_experiments.shared.inputs import resolve_inputs
        resolve_inputs(d["inputs"])
        for section, keys in (("target", {"kind", "field", "task"}),
                              ("model", {"kind", "hidden_sizes", "normalization"}),
                              ("output", {"encoding", "vocabulary", "loss", "readout"})):
            if set(d[section]) != keys:
                raise ValueError(f"Unsupported options in the implemented {section} block: {sorted(set(d[section]) ^ keys)}")
        if (d["target"]["kind"], d["output"]["encoding"]) != ("target_position", "categorical_coordinates"):
            raise ValueError("This target/output building block is planned and not implemented yet.")
        if d["output"].get("loss") != "softmax_cross_entropy" or d["output"].get("readout") != "separate_per_input":
            raise ValueError("Categorical coordinates use softmax_cross_entropy and separate_per_input readouts.")
        if d["model"].get("kind") != "mlp" or d["model"].get("normalization") != "train_standardize":
            raise ValueError("The implemented model block is an MLP with train_standardize normalization.")
        if d["sampling"] != {"unit": "episode", "split_unit": "episode", "backend": "episode_events"}:
            raise ValueError("The implemented collector backend samples aligned episode events.")
        if d["target"].get("field") != "target_position" or d["target"].get("task") != "target_cell":
            raise ValueError("The target_position block emits the target_cell task from target_position.")
        if d["output"].get("vocabulary") != "pooled_absolute_xyz_cells":
            raise ValueError("Categorical coordinates use pooled_absolute_xyz_cells.")

    def runtime_config(self):
        self.validate(runnable=True)
        d = copy.deepcopy(self.data)
        config = d["settings"]
        path = d["output"]["encoding"] == "binary_cell_mask"
        config["experiment"] = "path" if path else "target"
        config["protocol"]["decoder"] = "path_mask" if path else "shared_cell"
        config["protocol"]["target_storey"] = False
        config["probe"]["hidden_sizes"] = d["model"]["hidden_sizes"]
        config["definition"] = copy.deepcopy(self.data)
        return config


def validate_runtime(config):
    """Keep the current artifact layout while definitions select the building blocks."""
    Definition(config["definition"]).validate(runnable=True)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", config["run"]["name"]):
        raise ValueError("run.name must be 1–101 filename-safe characters.")
    for section, keys in (("dataset", ("train_samples", "validation_samples", "test_samples", "eval_batch_size", "max_episodes")),
                          ("probe", ("epochs", "batch_size", "patience")), ("report", ("dpi",)),
                          ("audit", ("samples", "every", "batch_size"))):
        for key in keys:
            if type(config[section][key]) is not int or config[section][key] < 1:
                raise ValueError(f"{section}.{key} must be a positive integer.")
    if config["dataset"]["max_episodes"] < sum(config["dataset"][f"{s}_samples"] for s in ("train", "validation", "test")):
        raise ValueError("dataset.max_episodes must cover every requested sample.")
    for key in ("seed", "split_seed"):
        if type(config["dataset"][key]) is not int or config["dataset"][key] < 0:
            raise ValueError(f"dataset.{key} must be a nonnegative integer.")
    widths, seeds = config["probe"]["hidden_sizes"], config["probe"]["seeds"]
    if not widths or any(type(w) is not int or w < 1 for w in widths):
        raise ValueError("MLP hidden sizes must be positive integers.")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Decoder seeds must be distinct nonnegative integers.")
    for key in ("learning_rate", "weight_decay", "min_delta"):
        value = config["probe"][key]
        if not np.isfinite(value) or value < 0 or (key == "learning_rate" and value == 0):
            raise ValueError(f"Invalid probe.{key}.")
    if config["policy"].get("map_source", "training") not in ("training", "validation", "compatible_pool"):
        raise ValueError("Unknown policy map_source.")
    if config["protocol"]["eligibility"] not in ("success", "delivered", "informed"):
        raise ValueError("Unknown episode eligibility rule.")
    expected_renderer = "path_mask" if config.get("experiment") == "path" else "location"
    if config["report"].get("renderer") != expected_renderer:
        raise ValueError(f"This definition requires report.renderer={expected_renderer}.")
    if type(config["audit"]["enabled"]) is not bool or config["audit"]["mode"] not in ("full", "conditional"):
        raise ValueError("Invalid audit enabled/mode settings.")
    if not 0 <= config["audit"]["duplicate_threshold"] <= 1:
        raise ValueError("audit.duplicate_threshold must be in [0, 1].")
    clearance = config["distance"]["clearance_m"]
    if clearance is not None and (not np.isfinite(clearance) or clearance < 0):
        raise ValueError("distance.clearance_m must be null or nonnegative.")
    if not config["policy"]["from_run_config"] or config["dataset"]["path"] is not None:
        raise ValueError("Suite probes use saved checkpoint config; use resume to reuse a collection.")
    if config.get("experiment") == "path":
        threshold = config["report"].get("threshold", .5)
        if not isinstance(threshold, (float, int)) or not 0 <= threshold <= 1:
            raise ValueError("report.threshold must be a probability in [0,1].")
        if config["audit"]["enabled"]:
            raise ValueError("The legacy location-collision audit is not defined for path masks.")
        if config["distance"]["clearance_m"] is not None:
            raise ValueError("Path references use the checkpoint's reward clearance; clearance_m must be null.")
        for section, key in (("probe", "updates_per_epoch"),):
            if type(config[section][key]) is not int or config[section][key] < 1:
                raise ValueError(f"{section}.{key} must be a positive integer.")
        if config["protocol"]["eligibility"] != "informed":
            raise ValueError("The path prototype retains informed episodes, including eventual failures.")
    elif config["audit"]["enabled"] and set(config["definition"]["inputs"]) != {
            "finder_gru", "finder_message", "base_message"}:
        raise ValueError("The location audit block needs all three event inputs; disable it for other requests.")


def validate_path_definition(d):
    from geo_experiments.shared.inputs import resolve_inputs
    resolve_inputs(d["inputs"])
    expected = {"finder_gru", "finder_message", "all_knowing_after_info", "all_knowing_till_end", "base"}
    if set(d["inputs"]) != expected:
        raise ValueError("The path prototype requires its five defined input groups.")
    ordinary = {"finder_gru": ("gru", "finder", "first_communication_after_discovery"),
                "finder_message": ("message", "finder", "first_communication_after_discovery"),
                "base": ("stored_message", "base", "first_receipt")}
    for name, values in ordinary.items():
        spec = d["inputs"][name]
        if tuple(spec[k] for k in ("kind", "agents", "event")) != values:
            raise ValueError(f"Unsupported capture rule for {name}.")
    for name in ("all_knowing_after_info", "all_knowing_till_end"):
        if d["inputs"][name]["kind"] != "gru" or d["inputs"][name]["agents"] != "active_target_knowing":
            raise ValueError("All-knowing input groups use each active informed agent's GRU.")
    first, repeated = d["inputs"]["all_knowing_after_info"], d["inputs"]["all_knowing_till_end"]
    if first.get("event") != "first_actor_update_processing_target_information" or first.get("capture") != "once_per_agent_episode":
        raise ValueError("First-informed capture must use the updated GRU once per agent/episode.")
    if repeated.get("event") != "actor_update_after_target_information" or repeated.get("stop") != "episode_end_or_agent_inactive" or repeated.get("include_first") is not True:
        raise ValueError("Repeated capture must begin with the first informed update and end at inactivity or episode end.")
    for value in (d["inputs"]["all_knowing_till_end"]["every_steps"], d["sampling"]["repeated_snapshot_cap_per_agent"]):
        if type(value) is not int or value < 1:
            raise ValueError("Candidate stride and retained snapshot cap must be positive integers.")
    if d["target"]["kind"] != "optimal_roadmap_path" or d["target"]["endpoints"] != "base_to_target":
        raise ValueError("The path target is the base-to-target shortest roadmap route.")
    if d["target"]["reference"] != "reward_geodesic_distance" or d["target"]["route_selection"] != "single_shortest":
        raise ValueError("Only a single reward-reference shortest route is implemented.")
    output = d["output"]
    if output["size"] != 125 or output["shape"] != [5, 5, 5] or output["origin"] != "base_cell" or output["rotations"] != "yaw_quarter_turns":
        raise ValueError("Path masks require the 125-cell base-relative yaw-canonical representation.")
    if output["loss"] != "sigmoid_binary_cross_entropy" or output["readout"] != "separate_per_input":
        raise ValueError("Path masks use sigmoid BCE and separate input readouts.")
    constants = {"relative_cell_range": [-2, 2], "origin_index": [2, 2, 2],
        "frame": "canonical_base_anchored_geometry", "cardinality": "multiple", "cell_order": "z_y_x_x_fastest",
        "symmetry_tiebreaker": "base_target_configuration_then_canonical_shortest_route"}
    if any(output.get(key) != value for key, value in constants.items()):
        raise ValueError("Unsupported path coordinate/encoding convention.")
    if d["model"]["kind"] != "mlp" or d["model"]["normalization"] != "train_standardize":
        raise ValueError("The path prototype uses a standardized vector MLP.")
    if d["sampling"]["repeated_snapshot_selection"] != "uniform_over_candidate_indices":
        raise ValueError("Retained candidates must be evenly spread over the stride grid.")
    if d["sampling"]["training_sampler"] != "uniform_episode_then_agent_then_snapshot" or d["sampling"]["evaluation_aggregation"] != "mean_snapshot_then_agent_then_episode":
        raise ValueError("Path training/evaluation must use the specified episode/agent balancing.")
    if d["sampling"]["backend"] != "mixed_episode_and_agent_events" or d["sampling"]["unit"] != "input_specific_event":
        raise ValueError("Path inputs require mixed episode/agent events.")
