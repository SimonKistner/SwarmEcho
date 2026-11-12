# Own cell

The planned [`cell-mask/probe.yaml`](cell-mask/probe.yaml) predicts each agent's
position as a binary map-cell mask, exactly one positive cell. Input is each
active agent's GRU. It uses the same planned agent-event collector and binary-mask
building blocks as path prediction, with its own sampling and configuration.

The definition is not runnable. Decide cadence, phase restrictions, balancing,
cell ordering and variable-map handling before implementation. Samples from an
episode must stay together across splits. See [the implementation plan](../../CONCEPT.md).

The historical own-cell **classifier** is archived in `../../archive/legacy/`;
it is not being converted to this output representation.
