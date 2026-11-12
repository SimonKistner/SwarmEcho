"""Input capabilities requested by definitions, independent of prediction targets."""
from dataclasses import dataclass


@dataclass(frozen=True)
class InputRequest:
    fields: tuple[tuple[str, str], ...]

    @property
    def needs_finder(self):
        return any(field.startswith("finder_") for _, field in self.fields)

    @property
    def needs_base(self):
        return any(field == "base_message" for _, field in self.fields)

    @property
    def has_agent_events(self):
        return any(field in ("first_informed_gru", "candidate_gru") for _, field in self.fields)


# New capabilities belong here and in the shared rollout engine, never in probes.
EVENT_INPUTS = {
    ("gru", "finder", "first_communication_after_discovery"): "finder_gru",
    ("message", "finder", "first_communication_after_discovery"): "finder_message",
    ("stored_message", "base", "first_receipt"): "base_message",
    ("gru", "active_target_knowing", "first_actor_update_processing_target_information"): "first_informed_gru",
    ("gru", "active_target_knowing", "actor_update_after_target_information"): "candidate_gru",
}


def resolve_inputs(inputs):
    if not inputs:
        raise ValueError("A probe must request at least one input.")
    fields = []
    for name, spec in inputs.items():
        if not name.isidentifier():
            raise ValueError(f"Input name must be an identifier: {name!r}")
        expected = {"kind", "agents", "event"}
        if spec.get("event") == "first_actor_update_processing_target_information":
            expected.add("capture")
            if spec.get("capture") != "once_per_agent_episode":
                raise ValueError("First informed GRU capture is once per agent/episode.")
        if spec.get("event") == "actor_update_after_target_information":
            expected.update(("every_steps", "include_first", "stop"))
            if type(spec.get("every_steps")) is not int or spec["every_steps"] < 1:
                raise ValueError("Informed GRU candidate stride must be a positive integer.")
        if set(spec) != expected:
            raise ValueError(f"Invalid options for input {name!r}; expected {sorted(expected)}.")
        if name in {"target_position", "base_position", "map_id", "episode_id", "finder_step", "base_step", "complete_sample"}:
            raise ValueError(f"Input name collides with sample metadata: {name!r}")
        key = (spec["kind"], spec["agents"], spec["event"])
        if key not in EVENT_INPUTS:
            raise ValueError(f"Input capability {key} is planned or unavailable; implement it in the shared collector.")
        fields.append((name, EVENT_INPUTS[key]))
    return InputRequest(tuple(fields))


def input_request(config):
    if "definition" in config:
        return resolve_inputs(config["definition"]["inputs"])
    return None  # Historical workflows retain their three-input contract.


def input_names(config):
    request = input_request(config)
    return tuple(name for name, _ in request.fields) if request else ("finder_gru", "finder_message", "base_message")


def project_event_inputs(output, request):
    """Project one batch without coupling sample validity to unrequested events."""
    if request.has_agent_events:
        raise ValueError("Agent-event requests require mixed row assembly, not episode-only feature projection.")
    features = {name: output[field] for name, field in request.fields}
    metadata = {key: value for key, value in output.items()
                if key not in ("finder_gru", "finder_message", "base_message")}
    metadata.update(features)
    metadata["complete_sample"] = ((output["finder_step"] >= 0) if request.needs_finder else True) & (
        (output["base_step"] >= 0) if request.needs_base else True)
    return metadata
