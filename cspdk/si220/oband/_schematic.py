"""Schematic closures for cspdk.si220.oband cells, linked to SAX models."""

from __future__ import annotations

from cspdk._schematic import (
    _1X2,
    _2X2,
    _CROSSING,
    _GRATING,
    _HEATER_FULL,
    _HEATER_TOP,
    _LEFT_BOTTOM,
    _LEFT_RIGHT,
    _PAD,
    _WIRE_BEND,
    _WIRE_STRAIGHT,
    sax_model,
    schematic,
)

_MODULE = "cspdk.si220.oband.models"


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
    models=[sax_model("straight_strip", _MODULE, ["o1", "o2"], params={"length": "length"})],
)
straight_rib_schematic = schematic(
    "straight",
    ["waveguide", "rib"],
    _LEFT_RIGHT,
    models=[sax_model("straight_rib", _MODULE, ["o1", "o2"], params={"length": "length"})],
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

# Electrical wires
wire_corner_schematic = schematic("wire-corner", ["wire", "corner"], _WIRE_BEND)
wire_corner45_schematic = schematic("wire-corner", ["wire", "corner"], _WIRE_BEND)
wire_corner45_straight_schematic = schematic("wire-corner", ["wire", "corner"], _WIRE_STRAIGHT)
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
    _2X2,
    models=[sax_model("coupler_ring", _MODULE, ["o1", "o2", "o3", "o4"])],
)

# Rings / MZI / spirals — composite, no top-level SAX model
ring_single_schematic = schematic("ring-single", ["ring", "single"], _LEFT_RIGHT)
ring_double_schematic = schematic("ring-double", ["ring", "double"], _2X2)
mzi_schematic = schematic("mzi", ["mzi"], _2X2)
spiral_schematic = schematic("spiral", ["spiral"], _LEFT_RIGHT)

# Heaters — oband exposes all via-stack electrical ports
straight_heater_metal_schematic = schematic(
    "modulator",
    ["heater", "modulator"],
    _HEATER_FULL,
    models=[
        sax_model(
            "straight_heater_metal",
            _MODULE,
            [
                "o1", "o2",
                "l_e1", "l_e2", "l_e3", "l_e4",
                "r_e1", "r_e2", "r_e3", "r_e4",
            ],
            params={"length": "length"},
        )
    ],
)
# straight_heater_meander inherits the same full-port layout; no SAX model.
straight_heater_meander_schematic = schematic(
    "modulator", ["heater", "modulator", "meander"], _HEATER_FULL
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

# Crossings — oband has no SAX models for these
crossing_schematic = schematic("crossing", ["crossing"], _CROSSING)
crossing_rib_schematic = schematic("crossing", ["crossing", "rib"], _CROSSING)

# Pads / via stacks
pad_schematic = schematic("pad", ["pad"], _PAD)
via_stack_schematic = schematic("pad", ["via", "stack"], _PAD)
