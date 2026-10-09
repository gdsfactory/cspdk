"""Schematic closures for cspdk.si_sus cells, linked to SAX models."""

from __future__ import annotations

from cspdk._schematic import (
    _GRATING,
    _LEFT_RIGHT,
    _LEFT_TOP,
    sax_model,
    schematic,
)

_MODULE = "cspdk.si_sus.models"
_TWO_PORTS = ["o1", "o2"]

straight_schematic = schematic(
    "straight",
    ["waveguide"],
    _LEFT_RIGHT,
    models=[sax_model("straight", _MODULE, _TWO_PORTS, params={"length": "length"})],
)
bend_euler_schematic = schematic(
    "bend",
    ["bend", "euler"],
    _LEFT_TOP,
    models=[sax_model("bend_euler", _MODULE, _TWO_PORTS)],
)
bend_circular_schematic = schematic(
    "bend",
    ["bend", "circular"],
    _LEFT_TOP,
    models=[sax_model("bend_circular", _MODULE, _TWO_PORTS)],
)
bend_s_schematic = schematic(
    "sbend",
    ["bend", "s"],
    _LEFT_RIGHT,
    models=[sax_model("bend_s", _MODULE, _TWO_PORTS)],
)
taper_schematic = schematic(
    "taper",
    ["taper"],
    _LEFT_RIGHT,
    models=[sax_model("taper", _MODULE, _TWO_PORTS)],
)
grating_coupler_rectangular_schematic = schematic(
    "grating-coupler",
    ["grating-coupler", "rectangular"],
    _GRATING,
    models=[sax_model("grating_coupler_rectangular", _MODULE, _TWO_PORTS)],
)
