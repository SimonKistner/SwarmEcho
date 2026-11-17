"""Relay feasibility based on the global candidate graph."""
from swarmecho.core.config import DynamicMaxGeoDistanceConfig
from swarmecho.env.relay_candidates import prepare_record as prepare_global_record
from swarmecho.env.relay_global import make_calculator as make_global_calculator


def prepare_record(record, cfg, building, *, options=None, randomize_base=False, cache_dir=None):
    return prepare_global_record(record, cfg, building, options or DynamicMaxGeoDistanceConfig(),
                                 randomize_base=randomize_base, cache_dir=cache_dir)


def make_calculator(cfg, geometry, building, bank=None, *, return_witness=False,
                    options=None, randomize_base=False, cache_dir=None):
    return make_global_calculator(cfg, geometry, building, bank, options or DynamicMaxGeoDistanceConfig(),
                                  randomize_base=randomize_base, return_witness=return_witness, cache_dir=cache_dir)
