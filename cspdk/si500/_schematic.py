"""Schematic closures for cspdk.si500 cells.

No SAX models live under ``cspdk.si500`` yet, so every closure emits an
empty ``models`` list. Adding SAX models is tracked separately.
"""

from __future__ import annotations

from cspdk._schematic import (
    _1X2,
    _2X2,
    _GRATING,
    _LEFT_BOTTOM,
    _LEFT_RIGHT,
    _MZI_1X2,
    _PAD,
    _WIRE_BEND,
    schematic,
)

straight_schematic = schematic("straight", ["waveguide"], _LEFT_RIGHT)
bend_euler_schematic = schematic("bend", ["bend", "euler"], _LEFT_BOTTOM)
bend_s_schematic = schematic("sbend", ["bend", "s"], _LEFT_RIGHT)
wire_corner_schematic = schematic("wire-corner", ["wire", "corner"], _WIRE_BEND)
taper_schematic = schematic("taper", ["taper"], _LEFT_RIGHT)
mmi1x2_schematic = schematic("mmi-1x2", ["mmi", "1x2"], _1X2)
mmi2x2_schematic = schematic("mmi-2x2", ["mmi", "2x2"], _2X2)
coupler_schematic = schematic("coupler", ["coupler"], _2X2)
coupler_straight_schematic = schematic("coupler", ["coupler", "straight"], _2X2)
grating_coupler_rectangular_schematic = schematic(
    "grating-coupler", ["grating-coupler", "rectangular"], _GRATING
)
grating_coupler_elliptical_schematic = schematic(
    "grating-coupler", ["grating-coupler", "elliptical"], _GRATING
)
mzi_schematic = schematic("mzi", ["mzi"], _MZI_1X2)
pad_schematic = schematic("pad", ["pad"], _PAD)
