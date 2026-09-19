"""Schematic closures for cspdk.ge_on_si cells."""

from __future__ import annotations

from cspdk._schematic import (
    _LEFT_RIGHT,
    _LEFT_TOP,
    _PAD,
    schematic,
)

straight_schematic = schematic("straight", ["waveguide"], _LEFT_RIGHT)
bend_euler_schematic = schematic("bend", ["bend", "euler"], _LEFT_TOP)
bend_s_schematic = schematic("sbend", ["bend", "s"], _LEFT_RIGHT)
taper_schematic = schematic("taper", ["taper"], _LEFT_RIGHT)
pad_schematic = schematic("pad", ["pad"], _PAD)
