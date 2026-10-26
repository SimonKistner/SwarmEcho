"""Strict snapshot identity checks with explicit, additive schema migrations."""
from __future__ import annotations


# Reviewed changes only: training-style logging, additive capture-time population
# fields and an opt-in early stop used solely by observation-history replays.
# No reset, action, snapshot timing, labels or episode-selection semantics changed.
SOURCE_MIGRATIONS = {
    "collection.py": ("dab786a8e3438153ce696e4fcd1459c81b9576876e3a910715c24270253cd0d1",
                      "6b1ddc64914d91d91df1f5dd9843d9aed5f26d66447fccf13cb97427cdf69ea4"),
    "target_collection.py": ("da74ffe42a71759d7e9f01503feae8a76758d113cc4f31b7610495cff86fa627",
                             "43beb2f5bb5c4e152a12fc0ef82b9394704d3fe6c01bde84649b6f702fe3f8b8"),
    "policy.py": ("4d80d07b7b672d69ae29e5f4ef9a12c181b28d0123cd5444bc5c2fc691c8c7c4",
                  "3dd59fed2b5df1f35b6c92007b90094ce477a1ce50e46a043dd885c224412d84"),
}


def snapshot_identity_matches(saved, current):
    if not saved or not current:
        return False
    ignored = {"collector_sha256", "probe_collection_files"}
    if {k: v for k, v in saved.items() if k not in ignored} != {k: v for k, v in current.items() if k not in ignored}:
        return False

    def compatible(name, previous, now):
        return previous == now or SOURCE_MIGRATIONS.get(name) == (previous, now)

    if not compatible("collection.py", saved.get("collector_sha256"), current.get("collector_sha256")):
        return False
    before, after = saved.get("probe_collection_files", {}), current.get("probe_collection_files", {})
    # Audit implementation is analysis of snapshots, not part of their generation.
    names = (set(before) | set(after)) - {"auditing.py"}
    return all(name in before and name in after and compatible(name, before[name], after[name]) for name in names)
