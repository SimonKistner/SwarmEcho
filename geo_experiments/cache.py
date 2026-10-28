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

# Opt-in validation banks leave the reviewed authored-map paths unchanged.
# These exact source revisions are accepted only for the original <=5-map
# protocol; validation-bank archives still require exact generation sources.
AUTHORED_SOURCE_MIGRATIONS = {
    "collection.py": ({"dab786a8e3438153ce696e4fcd1459c81b9576876e3a910715c24270253cd0d1",
                       "6b1ddc64914d91d91df1f5dd9843d9aed5f26d66447fccf13cb97427cdf69ea4"},
                      "3375deb092fbe4398b975689cdf6df4c6465e3dec954bdd199764ed6fa6513bb"),
    "target_collection.py": ({"da74ffe42a71759d7e9f01503feae8a76758d113cc4f31b7610495cff86fa627",
                              "43beb2f5bb5c4e152a12fc0ef82b9394704d3fe6c01bde84649b6f702fe3f8b8"},
                             "373644e69ec308c03a19111cbc75f69a99adf2f6f2985164d2a724713ff37397"),
    "policy.py": ({"4d80d07b7b672d69ae29e5f4ef9a12c181b28d0123cd5444bc5c2fc691c8c7c4",
                   "3dd59fed2b5df1f35b6c92007b90094ce477a1ce50e46a043dd885c224412d84"},
                  "75129cfa641add4a839c03deaad178ee4984852a2cece0031a6cd1806325d705"),
    "common.py": ({"96aae88c628e3c5eb1c1439e50e9a8b7a8554c5c68f2e083386f7ced132ab7b5"},
                  "68ef05f42a8e93186d849de7a76ab8d3e2a7232414b1cb16f493861517b2c257"),
    "labels.py": ({"b7ae9d12ac74fa540808979a891c46396d334ae18f0526d76a9a9a2b691e9e86"},
                  "f4303beaa5f5c2a20cae05f5d432069419f99e6c652254698548228bf845de67"),
}


def snapshot_identity_matches(saved, current):
    if not saved or not current:
        return False
    ignored = {"collector_sha256", "probe_collection_files"}
    if {k: v for k, v in saved.items() if k not in ignored} != {k: v for k, v in current.items() if k not in ignored}:
        return False

    def compatible(name, previous, now):
        if previous == now or SOURCE_MIGRATIONS.get(name) == (previous, now):
            return True
        authored = current.get("map_source", "training") == "training" and len(current.get("maps", [])) <= 5
        reviewed, expected = AUTHORED_SOURCE_MIGRATIONS.get(name, (set(), None))
        return authored and previous in reviewed and now == expected

    if not compatible("collection.py", saved.get("collector_sha256"), current.get("collector_sha256")):
        return False
    before, after = saved.get("probe_collection_files", {}), current.get("probe_collection_files", {})
    # Audit implementation is analysis of snapshots, not part of their generation.
    names = (set(before) | set(after)) - {"auditing.py"}
    return all(name in before and name in after and compatible(name, before[name], after[name]) for name in names)
