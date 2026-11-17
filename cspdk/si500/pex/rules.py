"""Default gfpex parasitic-extraction rules for the CORNERSTONE 500 nm SOI (Si500) platform.

Usage::

    from cspdk.si500.pex.rules import DEFAULT_PEX_RULES_CONFIG

The techfile is built by ``cspdk._pex.build_heater_pad_techfile``, which documents the stack convention (z = 0 at the
substrate top), the provenance tags and the units. This module holds the platform's sourced thicknesses.
"""

from __future__ import annotations

from cspdk._pex import build_heater_pad_techfile
from cspdk.si500.tech import LAYER, LAYER_STACK

__all__ = ["DEFAULT_PEX_RULES_CONFIG"]

BOX_THICKNESS_UM = 3.0  # FOUNDRY CORNERSTONE-42nd-Call-Design-Guidelines.pdf p.3-4
CORE_THICKNESS_UM = 0.5  # FOUNDRY CORNERSTONE-42nd-Call-Design-Guidelines.pdf p.3-4 (Si core; optical only, etched away under the field)
CLADDING_THICKNESS_UM = (
    2.0  # FOUNDRY CORNERSTONE-42nd-Call-Design-Guidelines.pdf p.3-4 (SiO2 top cladding)
)
# TODO(foundry): filament and pad thicknesses are not in the design guideline; these are the cspdk placeholders
# TODO(repo-mismatch): LAYER_STACK puts the pad at the filament bottom (zmin_metal = zmin_heater); here it lands on the filament top
HEATER_THICKNESS_UM = LAYER_STACK.layers[
    "heater"
].thickness  # ASSUMED cspdk.si500.tech.get_layer_stack placeholder
PAD_THICKNESS_UM = LAYER_STACK.layers[
    "metal"
].thickness  # ASSUMED cspdk.si500.tech.get_layer_stack placeholder

DEFAULT_PEX_RULES_CONFIG = build_heater_pad_techfile(
    name="cspdk-si500",
    heater_gds=tuple(LAYER.HEATER),
    pad_gds=tuple(LAYER.PAD),
    box_thickness_um=BOX_THICKNESS_UM,
    core_thickness_um=CORE_THICKNESS_UM,
    cladding_thickness_um=CLADDING_THICKNESS_UM,
    heater_thickness_um=HEATER_THICKNESS_UM,
    pad_thickness_um=PAD_THICKNESS_UM,
)
