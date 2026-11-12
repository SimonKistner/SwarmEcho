# Target location

This family asks where the target is. Its three subprobes differ in how the
output space and decoder routing represent that target:

- `shared-coordinates/`: active shared absolute-XYZ categorical decoder.
- `per-map-decoder/`: archived map-local decoder experts.
- `joint-map-and-cell/`: archived joint map-and-cell classifier.

Each has a `probe.yaml`. Only shared coordinates is executed through the new
shared pipeline. Probe folders contain no collection, fitting or report code.
Finder GRU, finder message and base stored message are configurable input requests,
each producing a separate readout in the active example.

See [the concept](../../CONCEPT.md) and [archived implementations](../../archive/README.md).
