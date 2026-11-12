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


# Exact before/after hashes for the package-only reorganization.
RELOCATION_SOURCE_MIGRATIONS = {'collection.py': ('5cd8c690097997f18fe9ed4e9841f9405fe9e2ccf8552ce7fc8c8d9d78912bdf',
                   'eb565efa3eb6a038198a5a65dbe43a582191c523a845b1d43e80cf4ca4fd3a84'),
 'common.py': ('ea136f80bc73bc9e11d8e2f74527b709e120fd82ac811dea7157804905c2d3a9',
               'a6e0e92e4d3526e6dea1704c8354bfaf295d39c699c8cfab36885ac36c843d96'),
 'geometry.py': ('69152e1ac3d19f0bcf19659abe4b5fb45c497b051b2f13e693b08392a82b8693',
                 '69152e1ac3d19f0bcf19659abe4b5fb45c497b051b2f13e693b08392a82b8693'),
 'labels.py': ('f4303beaa5f5c2a20cae05f5d432069419f99e6c652254698548228bf845de67',
               '41eec98b63fec267625833273efc4e3daacda2c725398fc8af48f853e79b35d0'),
 'policy.py': ('0555002c6227b740c14fe8f52f64d57f5944fbb28ab014f8ab69a748e20f9fd9',
               'f9e2b7a7df8018c00605354161b2db90318df64f7f55769796a840f7e9e25b60'),
 'target_collection.py': ('4ee07b32a26a13975fce7e33a409b3a9cdef1f7393280f43c0e6d68c9fabdd32',
                          '756f1149568afc4135fb5ecb5400ba3f7c477b21c28ceee1762537044f7bf9e4'),
 'validation_maps.py': ('1b7e8b6379df61c20ee6d62635bff179f4e0ffcdd094657bc269b3c2f71e7b09',
                        '71feb9e0da673ffacfc6ffe452b7ec7eb0deaef7db48fafc581f2c1c0bb90608')}

def snapshot_identity_matches(saved, current):
    if not saved or not current:
        return False
    ignored = {"collector_sha256", "probe_collection_files"}
    if {k: v for k, v in saved.items() if k not in ignored} != {k: v for k, v in current.items() if k not in ignored}:
        return False

    def compatible(name, previous, now):
        if previous == now:
            return True
        # Compose previously reviewed migrations with this exact relocation.
        relocation = RELOCATION_SOURCE_MIGRATIONS.get(name)
        if relocation and now == relocation[1]:
            now = relocation[0]
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
