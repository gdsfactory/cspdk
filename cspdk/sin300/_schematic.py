"""Schematic closures for cspdk.sin300 cells, linked to SAX models."""

from __future__ import annotations

from cspdk._schematic import (
    _1X2,
    _2X2,
    _GRATING,
    _LEFT_BOTTOM,
    _LEFT_RIGHT,
    _MZI_1X2,
    _PAD,
    _WIRE_STRAIGHT,
    sax_model,
    schematic,
)

_MODULE = "cspdk.sin300.models"


straight_schematic = schematic(
    "straight",
    ["waveguide"],
    _LEFT_RIGHT,
    models=[sax_model("straight", _MODULE, ["o1", "o2"], params={"length": "length"})],
)
bend_euler_schematic = schematic(
    "bend",
    ["bend", "euler"],
    _LEFT_BOTTOM,
    models=[sax_model("bend_euler", _MODULE, ["o1", "o2"])],
)
bend_s_schematic = schematic(
    "sbend",
    ["bend", "s"],
    _LEFT_RIGHT,
    models=[sax_model("bend_s", _MODULE, ["o1", "o2"])],
)
wire_corner_schematic = schematic("wire-corner", ["wire", "corner"], _WIRE_STRAIGHT)
taper_schematic = schematic(
    "taper",
    ["taper"],
    _LEFT_RIGHT,
    models=[sax_model("taper", _MODULE, ["o1", "o2"])],
)
mmi1x2_schematic = schematic(
    "mmi-1x2",
    ["mmi", "1x2"],
    _1X2,
    models=[sax_model("mmi1x2", _MODULE, ["o1", "o2", "o3"])],
)
mmi2x2_schematic = schematic(
    "mmi-2x2",
    ["mmi", "2x2"],
    _2X2,
    models=[sax_model("mmi2x2", _MODULE, ["o1", "o2", "o3", "o4"])],
)
coupler_schematic = schematic(
    "coupler",
    ["coupler"],
    _2X2,
    models=[sax_model("coupler", _MODULE, ["o1", "o2", "o3", "o4"])],
)
coupler_straight_schematic = schematic(
    "coupler",
    ["coupler", "straight"],
    _2X2,
    models=[sax_model("coupler_straight", _MODULE, ["o1", "o2", "o3", "o4"])],
)
grating_coupler_rectangular_schematic = schematic(
    "grating-coupler",
    ["grating-coupler", "rectangular"],
    _GRATING,
    models=[sax_model("grating_coupler_rectangular", _MODULE, ["o1", "o2"])],
)
grating_coupler_elliptical_schematic = schematic(
    "grating-coupler",
    ["grating-coupler", "elliptical"],
    _GRATING,
    models=[sax_model("grating_coupler_elliptical", _MODULE, ["o1", "o2"])],
)
mzi_schematic = schematic("mzi", ["mzi"], _MZI_1X2)
pad_schematic = schematic("pad", ["pad"], _PAD)
