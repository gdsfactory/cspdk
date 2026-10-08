"""Evanescent Couplers."""

import gdsfactory as gf
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from cspdk.si220._schematic import coupler_ring_schematic, coupler_schematic
from cspdk.si220.tech import TECH, get_band, is_rib


@gf.cell(tags=["couplers"], schematic_function=coupler_schematic)
def coupler(
    length: float | None = None,
    gap: float = TECH.gap_strip,
    cross_section: CrossSectionSpec = "strip_cband",
) -> gf.Component:
    """Returns Symmetric coupler.

    Args:
        length: of coupling region in um.
        gap: of coupling region in um.
        cross_section: optical cross-section selecting the operating band.
    """
    if is_rib(cross_section):
        # rib bends need a longer, shallower S-bend to respect radius_min
        return coupler_rib(length=length, gap=gap, cross_section=cross_section)
    if length is None:
        length = (
            TECH.length_coupler_oband
            if get_band(cross_section) == "oband"
            else TECH.length_coupler
        )
    return gf.c.coupler(
        length=length,
        gap=gap,
        dy=TECH.dy_coupler,
        dx=TECH.dx_coupler,
        cross_section=cross_section,
        allow_min_radius_violation=False,
    )


@gf.cell(tags=["couplers"], schematic_function=coupler_schematic)
def coupler_rib(
    length: float | None = None,
    gap: float = TECH.gap_rib,
    cross_section: CrossSectionSpec = "rib_cband",
) -> gf.Component:
    """Returns Symmetric coupler.

    Args:
        length: of coupling region in um.
        gap: of coupling region in um.
        cross_section: rib cross-section selecting the operating band.
    """
    length = length if length is not None else TECH.length_coupler_rib
    return gf.c.coupler(
        length=length,
        gap=gap,
        dy=TECH.dy_coupler_rib,
        dx=TECH.dx_coupler_rib,
        cross_section=cross_section,
        allow_min_radius_violation=False,
    )


@gf.cell(tags=["couplers"], schematic_function=coupler_ring_schematic)
def coupler_ring(
    length_x: float = 4,
    gap: float = TECH.gap_strip,
    radius: float = TECH.radius_strip,
    bend: ComponentSpec = "bend_euler",
    straight: ComponentSpec = "straight",
    cross_section: str = "strip_cband",
    length_extension: float = 10,
) -> gf.Component:
    """Returns Coupler for ring.

    Args:
        length_x: length of the parallel coupled straight waveguides.
        gap: gap between for coupler.
        radius: for the bend and coupler.
        bend: 90 degrees bend spec.
        straight: straight spec.
        cross_section: cross_section spec.
        length_extension: length extension for the coupler.
    """
    return gf.c.coupler_ring(
        length_x=length_x,
        gap=gap,
        radius=radius,
        bend=bend,
        straight=straight,
        cross_section=cross_section,
        cross_section_bend=None,
        length_extension=length_extension,
    )
