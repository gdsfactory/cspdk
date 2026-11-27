"""gflvs LVS rules for cspdk: ``gflvs.lvs(netlist, lib, DEFAULT_LVS_CONFIG)``.

Importing this package imports gflvs; ``import cspdk`` never does.
"""

from cspdk.lvs.rules import (
    DEFAULT_LVS_CONFIG,
    DEVICE_TEMPLATES,
    LAYER_CONNECTIVITY,
    PARAMETER_EXTRACTORS,
    PCELL_GENERATORS,
    build_lvs_config,
)

__all__ = [
    "DEFAULT_LVS_CONFIG",
    "DEVICE_TEMPLATES",
    "LAYER_CONNECTIVITY",
    "PARAMETER_EXTRACTORS",
    "PCELL_GENERATORS",
    "build_lvs_config",
]
