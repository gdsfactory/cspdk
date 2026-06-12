"""Schematic closures for cspdk.si_sus cells."""

from __future__ import annotations

from cspdk._schematic import (
    _GRATING,
    _LEFT_BOTTOM,
    _LEFT_RIGHT,
    _PAD,
    schematic,
)

straight_schematic = schematic("straight", ["waveguide"], _LEFT_RIGHT)
bend_euler_schematic = schematic("bend", ["bend", "euler"], _LEFT_BOTTOM)
bend_circular_schematic = schematic("bend", ["bend", "circular"], _LEFT_BOTTOM)
bend_s_schematic = schematic("sbend", ["bend", "s"], _LEFT_RIGHT)
taper_schematic = schematic("taper", ["taper"], _LEFT_RIGHT)
grating_coupler_rectangular_schematic = schematic(
    "grating-coupler", ["grating-coupler", "rectangular"], _GRATING
)
pad_schematic = schematic("pad", ["pad"], _PAD)
