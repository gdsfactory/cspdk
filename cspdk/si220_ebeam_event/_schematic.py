"""Schematic closures for cspdk.si220_ebeam_event cells, linked to SAX models."""

from __future__ import annotations

from cspdk._schematic import (
    _2X2,
    _GRATING,
    _LEFT_RIGHT,
    _LEFT_TOP,
    _RING_COUPLER,
    sax_model,
    schematic,
)

_MODULE = "cspdk.si220_ebeam_event.models"
_TWO_PORTS = ["o1", "o2"]
_FOUR_PORTS = ["o1", "o2", "o3", "o4"]

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
coupler_schematic = schematic(
    "coupler",
    ["coupler"],
    _2X2,
    models=[sax_model("coupler", _MODULE, _FOUR_PORTS)],
)
coupler_ring_schematic = schematic(
    "coupler-ring",
    ["coupler", "ring"],
    _RING_COUPLER,
    models=[sax_model("coupler_ring", _MODULE, _FOUR_PORTS)],
)
# Composite: simulated from its netlist (coupler_ring, straights and bends).
ring_single_schematic = schematic("ring-single", ["ring", "single"], _LEFT_RIGHT)
grating_coupler_schematic = schematic(
    "grating-coupler",
    ["grating-coupler", "rectangular"],
    _GRATING,
    models=[sax_model("grating_coupler", _MODULE, _TWO_PORTS)],
)
