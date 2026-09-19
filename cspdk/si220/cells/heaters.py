"""Heater components."""

from functools import partial

import gdsfactory as gf
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, LayerSpec

from cspdk.si220._schematic import (
    straight_heater_meander_schematic,
    straight_heater_metal_schematic,
)
from cspdk.si220.tech import LAYER, get_band, strip_heater_metal


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
