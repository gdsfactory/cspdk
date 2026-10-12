"""Heater components."""

import math
from functools import partial

import gdsfactory as gf
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, LayerSpec, Size

from cspdk.si220._schematic import (
    heater_schematic,
    straight_heater_meander_schematic,
    straight_heater_metal_schematic,
)
from cspdk.si220.tech import LAYER, get_band, strip_heater_metal


@gf.cell(tags=["heaters"], schematic_function=heater_schematic)
def heater(
    length: float = 200.0,
    width: float = 2.0,
    lead_length: float = 50.0,
    lead_width: float = 50.0,
    pad_size: Size = (100.0, 100.0),
    pad_pitch: float = 120.0,
    pad_taper_length: float = 30.0,
    layer_heater: LayerSpec = "HEATER",
    layer_pad: LayerSpec = "PAD",
) -> gf.Component:
    """Return a heater filament with a contact pad at each end.

    Drawn like the Cornerstone ``Heater`` cell, which the defaults reproduce. The
    filament runs along x from 0 to ``length``, centred on y = 0, so it sits on a
    waveguide placed on the x axis. At each end a lead fans out from the filament
    (``width`` wide) to ``lead_width`` over ``lead_length``; it is drawn on both
    the heater and the pad layer so the two overlap. A taper of
    ``pad_taper_length`` then widens the lead to the pad. As on the foundry cell,
    a disc of radius ``width / 2`` sits on the inner corner where each lead
    leaves the filament.

    Args:
        length: filament length (um).
        width: filament width (um).
        lead_length: length of the lead from the filament to the pad taper (um).
        lead_width: width of the lead at the pad taper (um).
        pad_size: pad width and height (um).
        pad_pitch: distance between the pad centres (um), centred on the filament.
        pad_taper_length: length of the taper from the lead to the pad (um); 0
            puts the pad on the lead.
        layer_heater: filament layer.
        layer_pad: pad layer.
    """
    pad_width, pad_height = pad_size
    y_stub = width / 2
    y_lead = y_stub + lead_length
    y_pad = y_lead + pad_taper_length
    radius = width / 2

    c = gf.Component()
    c.add_polygon(
        [(0, -y_stub), (0, y_stub), (length, y_stub), (length, -y_stub)],
        layer=layer_heater,
    )
    for sign, x_end, name in ((1, 0.0, "e1"), (-1, length, "e2")):
        x_lead = x_end + sign * width / 2
        x_pad = length / 2 - sign * pad_pitch / 2
        lead = [
            (x_end, y_stub),
            (x_end + sign * width, y_stub),
            (x_lead + sign * lead_width / 2, y_lead),
            (x_lead - sign * lead_width / 2, y_lead),
        ]
        corner = (x_end + sign * width, y_stub)
        round_corner = [
            (
                corner[0] + radius * math.cos(math.radians(2.5 * i)),
                corner[1] + radius * math.sin(math.radians(2.5 * i)),
            )
            for i in range(144)
        ]
        c.add_polygon(lead, layer=layer_heater)
        c.add_polygon(round_corner, layer=layer_heater)
        c.add_polygon(lead, layer=layer_pad)
        if pad_taper_length:
            c.add_polygon(
                [
                    (x_lead - lead_width / 2, y_lead),
                    (x_lead + lead_width / 2, y_lead),
                    (x_pad + pad_width / 2, y_pad),
                    (x_pad - pad_width / 2, y_pad),
                ],
                layer=layer_pad,
            )
        c.add_polygon(
            [
                (x_pad - pad_width / 2, y_pad),
                (x_pad - pad_width / 2, y_pad + pad_height),
                (x_pad + pad_width / 2, y_pad + pad_height),
                (x_pad + pad_width / 2, y_pad),
            ],
            layer=layer_pad,
        )
        c.add_port(
            name,
            center=(x_pad, y_pad + pad_height),
            orientation=90,
            width=pad_width,
            layer=layer_pad,
            port_type="electrical",
        )
    c.info["length"] = length
    return c


@gf.cell(tags=["heaters"], schematic_function=straight_heater_metal_schematic)
def straight_heater_metal(
    length: float = 320.0,
    length_undercut_spacing: float = 6.0,
    length_undercut: float = 30.0,
    length_straight: float = 0.1,
    length_straight_input: float = 15.0,
    with_undercut: bool = False,
    port_orientation1: int | None = None,
    port_orientation2: int | None = None,
    cross_section: CrossSectionSpec = "strip_cband",
) -> gf.Component:
    """Returns a thermal phase shifter.

    dimensions from https://doi.org/10.1364/OE.27.010456

    Args:
        length: phase shifter length.
        length_undercut_spacing: spacing between the waveguide and the undercut.
        length_undercut: undercut length.
        length_straight: straight length.
        length_straight_input: straight length input.
        with_undercut: isolation trenches for higher efficiency.
        port_orientation1: orientation of the first port. None for all orientations.
        port_orientation2: orientation of the second port. None for all orientations.
        cross_section: optical cross-section selecting the operating band.
    """
    if get_band(cross_section) == "cband":
        port_orientation1 = 90 if port_orientation1 is None else port_orientation1
        port_orientation2 = 90 if port_orientation2 is None else port_orientation2
    optical_width = gf.get_cross_section(cross_section).width
    heated_cross_section = partial(strip_heater_metal, width=optical_width)
    undercut_cross_section = partial(
        gf.cross_section.strip_heater_metal_undercut,
        width=optical_width,
        layer="WG",
        layer_heater="HEATER",
        layer_trench=LAYER.SLAB,
    )
    return gf.c.straight_heater_metal_undercut(
        length=length,
        length_undercut_spacing=length_undercut_spacing,
        length_undercut=length_undercut,
        length_straight=length_straight,
        length_straight_input=length_straight_input,
        with_undercut=with_undercut,
        port_orientation1=port_orientation1,
        port_orientation2=port_orientation2,
        via_stack="via_stack_heater_mtop",
        heater_taper_length=5.0,
        ohms_per_square=None,
        cross_section=cross_section,
        cross_section_waveguide_heater=heated_cross_section,
        cross_section_heater_undercut=undercut_cross_section,
    )


@gf.cell(tags=["heaters"], schematic_function=straight_heater_meander_schematic)
def straight_heater_meander(
    length: float = 320.0,
    heater_width: float = 2.5,
    spacing: float = 2,
    cross_section: CrossSectionSpec = "strip_cband",
    layer_heater: LayerSpec = "HEATER",
    via_stack: ComponentSpec | None = "via_stack_heater_mtop",
    n: int | None = 3,
    port_orientation1: float | None = None,
    port_orientation2: float | None = None,
    radius: float | None = None,
) -> gf.Component:
    """Returns a meander based heater.

    based on SungWon Chung, Makoto Nakai, and Hossein Hashemi,
    Low-power thermo-optic silicon modulator for large-scale photonic integrated systems
    Opt. Express 27, 13430-13459 (2019)
    https://www.osapublishing.org/oe/abstract.cfm?URI=oe-27-9-13430

    Args:
        length: phase shifter length.
        heater_width: width of the heater.
        spacing: waveguide spacing (center to center).
        cross_section: for waveguide.
        layer_heater: for top heater, if None, it does not add a heater.
        via_stack: for the heater to via_stack metal.
        n: number of meanders.
        port_orientation1: orientation of the first port. None for all orientations.
        port_orientation2: orientation of the second port. None for all orientations.
        radius: radius of the meander.
    """
    band = get_band(cross_section)
    if band == "cband":
        port_orientation1 = 90 if port_orientation1 is None else port_orientation1
        port_orientation2 = 90 if port_orientation2 is None else port_orientation2
    return gf.c.straight_heater_meander(
        spacing=spacing,
        cross_section=cross_section,
        layer_heater=layer_heater,
        via_stack=via_stack,
        length=length,
        heater_width=heater_width,
        extension_length=15.0,
        port_orientation1=port_orientation1,
        port_orientation2=port_orientation2,
        heater_taper_length=10.0,
        straight_widths=None,
        taper_length=10.0,
        n=n,
        radius=radius,
    )
