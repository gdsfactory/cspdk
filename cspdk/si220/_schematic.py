"""Schematic closures for cspdk.si220 cells, linked to SAX models."""

from __future__ import annotations

from functools import partial

from cspdk._schematic import (
    _1X2,
    _2X2,
    _CROSSING,
    _GRATING,
    _HEATER_FULL,
    _HEATER_TOP,
    _LEFT_RIGHT,
    _LEFT_TOP,
    _PAD,
    _RING_COUPLER,
    _RING_DOUBLE,
    _WIRE_BEND,
    _WIRE_STRAIGHT,
    _die_ports,
    circulax_model,
    sax_model,
    schematic,
)
from cspdk.si220.tech import get_band

_MODULE = "cspdk.si220.models"
_ACTIVE_MODULE = "cspdk.si220.active_models"


def _heater_ports(
    cross_section: object = "strip_cband", **_ignored: object
) -> list[dict]:
    """Return the electrical ports exposed by the selected band."""
    return _HEATER_FULL if get_band(cross_section) == "oband" else _HEATER_TOP


# Waveguides
straight_schematic = schematic(
    "straight",
    ["waveguide"],
    _LEFT_RIGHT,
    models=[sax_model("straight", _MODULE, ["o1", "o2"], params={"length": "length"})],
)
straight_strip_schematic = schematic(
    "straight",
    ["waveguide", "strip"],
    _LEFT_RIGHT,
    models=[
        sax_model("straight_strip", _MODULE, ["o1", "o2"], params={"length": "length"})
    ],
)
straight_rib_schematic = schematic(
    "straight",
    ["waveguide", "rib"],
    _LEFT_RIGHT,
    models=[
        sax_model("straight_rib", _MODULE, ["o1", "o2"], params={"length": "length"})
    ],
)
bend_euler_schematic = schematic(
    "bend",
    ["bend", "euler"],
    _LEFT_TOP,
    models=[sax_model("bend_euler", _MODULE, ["o1", "o2"])],
)
bend_circular_schematic = schematic(
    "bend",
    ["bend", "circular"],
    _LEFT_TOP,
    models=[sax_model("bend_circular", _MODULE, ["o1", "o2"])],
)
bend_s_schematic = schematic(
    "sbend",
    ["bend", "s"],
    _LEFT_RIGHT,
    models=[sax_model("bend_s", _MODULE, ["o1", "o2"])],
)

# Electrical wires (no SAX models for routing metal)
wire_corner_schematic = schematic("wire-corner", ["wire", "corner"], _WIRE_BEND)
wire_corner45_schematic = schematic("wire-corner", ["wire", "corner"], _WIRE_BEND)
wire_corner45_straight_schematic = schematic(
    "wire-corner", ["wire", "corner"], _WIRE_BEND
)
straight_metal_schematic = schematic("wire", ["wire", "straight"], _WIRE_STRAIGHT)
bend_metal_schematic = schematic("wire-bend", ["wire", "bend"], _WIRE_BEND)
bend_s_metal_schematic = schematic("wire-sbend", ["wire", "sbend"], _WIRE_STRAIGHT)

# Tapers
taper_schematic = schematic(
    "taper",
    ["taper"],
    _LEFT_RIGHT,
    models=[sax_model("taper", _MODULE, ["o1", "o2"])],
)
taper_metal_schematic = schematic("taper", ["taper", "metal"], _WIRE_STRAIGHT)
transition_schematic = schematic(
    "taper",
    ["taper", "transition"],
    _LEFT_RIGHT,
    models=[sax_model("taper_strip_to_ridge", _MODULE, ["o1", "o2"])],
)
taper_mm2sm_schematic = schematic(
    "taper",
    ["taper", "mm2sm"],
    _LEFT_RIGHT,
    models=[sax_model("taper_mm2sm", _MODULE, ["o1", "o2"])],
)
taper_rib_to_strip_schematic = schematic(
    "taper",
    ["taper", "transition"],
    _LEFT_RIGHT,
    models=[sax_model("taper_rib_to_strip", _MODULE, ["o1", "o2"])],
)

# MMIs
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

# Couplers
coupler_schematic = schematic(
    "coupler",
    ["coupler"],
    _2X2,
    models=[sax_model("coupler", _MODULE, ["o1", "o2", "o3", "o4"])],
)
coupler_ring_schematic = schematic(
    "coupler-ring",
    ["coupler", "ring"],
    _RING_COUPLER,
    models=[sax_model("coupler_ring", _MODULE, ["o1", "o2", "o3", "o4"])],
)

# Rings (no top-level SAX model; composites)
ring_single_schematic = schematic("ring-single", ["ring", "single"], _LEFT_RIGHT)
ring_double_schematic = schematic("ring-double", ["ring", "double"], _RING_DOUBLE)

# MZI (composite; cband mzi uses 2x2 splitter so 4 ports)
mzi_schematic = schematic("mzi-2x2", ["mzi"], _2X2)

# Foundry MZI: the input faces east, the two outputs west, the heater pads north.
_MZI_HEATER_LOOP = [
    {"name": "o1", "side": "right", "type": "photonic"},
    {"name": "o2", "side": "left", "type": "photonic"},
    {"name": "o3", "side": "left", "type": "photonic"},
    {"name": "e1", "side": "top", "type": "electric"},
    {"name": "e2", "side": "top", "type": "electric"},
]
_MZI_HEATER_LOOP_PORTS = ["o1", "o2", "o3", "e1", "e2"]
mzi_heater_loop_schematic = schematic(
    "ckt",
    ["mzi", "heater"],
    _MZI_HEATER_LOOP,
    models=[sax_model("mzi_heater_loop", _MODULE, _MZI_HEATER_LOOP_PORTS)],
)

# Spirals (composite)
spiral_schematic = schematic("spiral", ["spiral"], _LEFT_RIGHT)

# Heaters
# Rendered as a generic model-driven box ("ckt") so all heater ports show.
# The Mosaic "modulator" symbol uses hardcoded conn and cannot represent a
# multi-port heater; see the model-driven-modulator follow-up issue.
# SAX model lists all 8 electrical ports, but Circulax active model only
# implements o1, o2, l_e2, r_e2; these entries use the Circulax model.
_HEATER_MODELS = [
    circulax_model(
        "ThermalPhaseShifter",
        _ACTIVE_MODULE,
        ["o1", "o2", "l_e2", "r_e2"],
        params={"length": "length"},
    )
]
straight_heater_metal_schematic = schematic(
    "ckt", ["heater", "modulator"], _heater_ports, models=_HEATER_MODELS
)
straight_heater_meander_schematic = schematic(
    "ckt", ["heater", "modulator", "meander"], _heater_ports, models=_HEATER_MODELS
)
# Heater filament with two pads (no optical ports); placed over a waveguide.
_HEATER_PADS = [
    {"name": "e1", "side": "top", "type": "electric"},
    {"name": "e2", "side": "top", "type": "electric"},
]
heater_schematic = schematic(
    "ckt",
    ["heater"],
    _HEATER_PADS,
    models=[sax_model("heater", _MODULE, ["e1", "e2"])],
)

# Grating couplers
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

# Crossings
crossing_schematic = schematic(
    "crossing",
    ["crossing"],
    _CROSSING,
    models=[sax_model("crossing", _MODULE, ["o1", "o2", "o3", "o4"])],
)
crossing_rib_schematic = schematic(
    "crossing",
    ["crossing", "rib"],
    _CROSSING,
    models=[sax_model("crossing_rib", _MODULE, ["o1", "o2", "o3", "o4"])],
)

# Run 50 fixed Y-splitter (o1 in, o2 top and o3 bottom out, like mmi1x2)
ysplitter_schematic = schematic(
    "mmi-1x2",
    ["splitter", "1x2", "y-branch"],
    _1X2,
    models=[sax_model("ysplitter", _MODULE, ["o1", "o2", "o3"])],
)

# Pads / via stacks
pad_schematic = schematic("pad", ["pad"], _PAD)
via_stack_schematic = schematic("pad", ["via", "stack"], _PAD)

# Die: symbol "die" makes editors render it as a die (no background box).
# Ports are derived from the factory kwargs and follow the GDS orientations; the
# default of 16 gratings per side is the Run-50 packaging template's.
die_schematic = schematic("die", ["die"], partial(_die_ports, ngratings=16))
