"""Primitives."""

import gdsfactory as gf
from gdsfactory.add_pins import add_electrical_pins
from gdsfactory.cross_section import port_names_electrical, port_types_electrical
from gdsfactory.typings import CrossSectionSpec, LayerSpec, Size

from cspdk.si220._schematic import FOUNDRY_SCHEMATICS as _FOUNDRY
from cspdk.si220._schematic import (
    bend_circular_schematic,
    bend_euler_schematic,
    bend_metal_schematic,
    bend_s_metal_schematic,
    bend_s_schematic,
    straight_metal_schematic,
    straight_rib_schematic,
    straight_schematic,
    straight_strip_schematic,
    wire_corner45_schematic,
    wire_corner45_straight_schematic,
    wire_corner_schematic,
)


@gf.cell(tags=["waveguides"], schematic_function=straight_schematic)
def straight(
    length: float = 10,
    cross_section: CrossSectionSpec = "strip_cband",
    width: float | None = None,
    npoints: int = 2,
) -> gf.Component:
    """Returns a Straight waveguide.

    Args:
        length: straight length (um).
        cross_section: specification (CrossSection, string or dict).
        width: width of the waveguide. If None, it will use the width of the cross_section.
        npoints: number of points.
    """
    return gf.c.straight(
        length=length, cross_section=cross_section, width=width, npoints=npoints
    )


@gf.cell(tags=["waveguides"], schematic_function=straight_strip_schematic)
def straight_strip(
    length: float = 10,
    cross_section: CrossSectionSpec = "strip_cband",
    width: float | None = None,
    npoints: int = 2,
) -> gf.Component:
    """Returns a Straight waveguide.

    Args:
        length: straight length (um).
        cross_section: specification (CrossSection, string or dict).
        width: width of the waveguide. If None, it will use the width of the cross_section.
        npoints: number of points.
    """
    return gf.c.straight(
        length=length, cross_section=cross_section, width=width, npoints=npoints
    )


@gf.cell(tags=["waveguides"], schematic_function=straight_rib_schematic)
def straight_rib(
    length: float = 10,
    cross_section: CrossSectionSpec = "rib_cband",
    width: float | None = None,
) -> gf.Component:
    """Returns a Straight waveguide.

    Args:
        length: straight length (um).
        cross_section: specification (CrossSection, string or dict).
        width: width of the waveguide. If None, it will use the width of the cross_section.
    """
    return gf.c.straight(
        length=length, cross_section=cross_section, width=width, npoints=2
    )


@gf.cell(tags=["waveguides"], schematic_function=bend_euler_schematic)
def bend_euler(
    radius: float | None = None,
    angle: float = 90,
    p: float = 0.5,
    width: float | None = None,
    cross_section: CrossSectionSpec = "strip_cband",
    allow_min_radius_violation: bool = False,
) -> gf.Component:
    """Regular degree euler bend.

    Args:
        radius: in um. Defaults to cross_section_radius.
        angle: total angle of the curve.
        p: Proportion of the curve that is an Euler curve.
        width: width to use. Defaults to cross_section.width.
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
        allow_min_radius_violation: if True allows radius to be smaller than cross_section radius.
    """
    return gf.c.bend_euler(
        radius=radius,
        angle=angle,
        p=p,
        width=width,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
        with_arc_floorplan=True,
        npoints=None,
        layer=None,
    )


def _slab_as_sections(cross_section: CrossSectionSpec) -> gf.CrossSection:
    """``cross_section`` with its slab (bbox) layers drawn along the path instead.

    A bbox slab is a rectangle around the whole component; on a bend the foundry
    slab follows the core at a constant offset.
    """
    xs = gf.get_cross_section(cross_section)
    if not xs.bbox_layers:
        return xs
    slabs = tuple(
        gf.Section(width=xs.width + 2 * offset, layer=layer, name=f"slab_{i}")
        for i, (layer, offset) in enumerate(zip(xs.bbox_layers, xs.bbox_offsets))
    )
    return xs.copy(sections=(*xs.sections, *slabs), bbox_layers=(), bbox_offsets=())


@gf.cell(tags=["waveguides"], schematic_function=bend_circular_schematic)
def bend_circular(
    radius: float | None = None,
    angle: float = 90,
    npoints: int | None = None,
    width: float | None = None,
    cross_section: CrossSectionSpec = "strip_cband",
    allow_min_radius_violation: bool = False,
) -> gf.Component:
    """Circular bend, like the Cornerstone 90 degree bends.

    A rib cross-section's slab follows the bend at a constant offset.

    Args:
        radius: in um. Defaults to the cross-section radius.
        angle: total angle of the curve.
        npoints: number of points along the arc. Defaults to gdsfactory's.
        width: width to use. Defaults to cross_section.width.
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
        allow_min_radius_violation: if True allows radius to be smaller than cross_section radius.
    """
    xs = gf.get_cross_section(cross_section)
    if width is not None:
        xs = xs.copy(width=width)
    return gf.c.bend_circular(
        radius=radius if radius is not None else xs.radius,
        angle=angle,
        npoints=npoints,
        cross_section=_slab_as_sections(xs),
        allow_min_radius_violation=allow_min_radius_violation,
    )


@gf.cell(
    tags=["waveguides", "foundry"],
    schematic_function=_FOUNDRY["SOI220_cband_TE_strip_90deg_bend"],
)
def SOI220_cband_TE_strip_90deg_bend() -> gf.Component:
    """Cornerstone C-band strip 90 degree bend, 5 um radius (Run-50 template cell)."""
    return bend_circular(
        radius=5,
        npoints=31,
        cross_section="strip_cband",
        allow_min_radius_violation=True,
    )


@gf.cell(
    tags=["waveguides", "foundry"],
    schematic_function=_FOUNDRY["SOI220_oband_TE_strip_90deg_bend"],
)
def SOI220_oband_TE_strip_90deg_bend() -> gf.Component:
    """Cornerstone O-band strip 90 degree bend, 5 um radius (Run-50 template cell).

    The 50th-call components PDF quotes a 30 um radius; the template cell is 5 um.
    """
    return bend_circular(
        radius=5,
        npoints=31,
        cross_section="strip_oband",
        allow_min_radius_violation=True,
    )


@gf.cell(
    tags=["waveguides", "foundry"],
    schematic_function=_FOUNDRY["SOI220_cband_TE_rib_90deg_bend"],
)
def SOI220_cband_TE_rib_90deg_bend() -> gf.Component:
    """Cornerstone C-band rib 90 degree bend, 25 um radius (Run-50 template cell)."""
    return bend_circular(radius=25, npoints=156, cross_section="rib_cband")


@gf.cell(tags=["waveguides"], schematic_function=bend_s_schematic)
def bend_s(
    size: Size = (11, 1.8),
    cross_section: CrossSectionSpec = "strip_cband",
    width: float | None = None,
    allow_min_radius_violation: bool = False,
) -> gf.Component:
    """Return S bend with bezier curve.

    stores min_bend_radius property in self.info['min_bend_radius']
    min_bend_radius depends on height and length

    Args:
        size: in x and y direction.
        cross_section: spec.
        width: width of the waveguide. If None, it will use the width of the cross_section.
        allow_min_radius_violation: allows min radius violations.
    """
    return gf.c.bend_s(
        size=size,
        cross_section=cross_section,
        npoints=99,
        allow_min_radius_violation=allow_min_radius_violation,
        width=width,
    )


@gf.cell(tags=["waveguides"], schematic_function=wire_corner_schematic)
def wire_corner(
    cross_section: CrossSectionSpec = "metal_routing",
    width: float | None = None,
    radius: float | None = None,
) -> gf.Component:
    """Returns 45 degrees electrical corner wire.

    Args:
        cross_section: spec.
        width: optional width. Defaults to cross_section width.
        radius: ignored.
    """
    return gf.c.wire_corner(
        cross_section=cross_section,
        width=width,
        port_names=port_names_electrical,
        port_types=port_types_electrical,
        radius=radius,
    )


@gf.cell(tags=["waveguides"], schematic_function=wire_corner45_schematic)
def wire_corner45(
    cross_section: CrossSectionSpec = "metal_routing",
    radius: float = 10,
    width: float | None = None,
    layer: LayerSpec | None = None,
    with_corner90_ports: bool = True,
) -> gf.Component:
    """Returns 90 degrees electrical corner wire.

    Args:
        cross_section: spec.
        radius: in um.
        width: optional width. Defaults to cross_section width.
        layer: optional layer.
        with_corner90_ports: if True, adds ports at 90 degrees.
    """
    return gf.c.wire_corner45(
        cross_section=cross_section,
        radius=radius,
        width=width,
        layer=layer,
        with_corner90_ports=with_corner90_ports,
    )


@gf.cell(tags=["waveguides"], schematic_function=wire_corner45_straight_schematic)
def wire_corner45_straight(
    width: float | None = None,
    radius: float | None = None,
    cross_section: CrossSectionSpec = "metal_routing",
) -> gf.Component:
    """Returns 90 degrees electrical corner wire.

    Args:
        width: of the wire.
        radius: of the corner. Defaults to width.
        cross_section: metal_routing.
    """
    return gf.c.wire_corner45_straight(
        width=width,
        radius=radius,
        cross_section=cross_section,
    )


####################
# Metal waveguides
####################


@gf.cell(tags=["waveguides"], schematic_function=straight_metal_schematic)
def straight_metal(
    length: float = 10,
    cross_section: CrossSectionSpec = "metal_routing",
    width: float | None = None,
) -> gf.Component:
    """Returns a Straight waveguide.

    Args:
        length: straight length (um).
        cross_section: specification (CrossSection, string or dict).
        width: width of the waveguide. If None, it will use the width of the cross_section.
    """
    c = gf.c.straight(
        length=length, cross_section=cross_section, width=width, npoints=2
    )
    add_electrical_pins(c)
    return c


@gf.cell(tags=["waveguides"], schematic_function=bend_metal_schematic)
def bend_metal(
    radius: float | None = None,
    angle: float = 90,
    width: float | None = None,
    cross_section: CrossSectionSpec = "metal_routing",
) -> gf.Component:
    """Return a circular metal-routing bend.

    Args:
        radius: Bend radius. Uses the cross-section radius or width when omitted.
        angle: Bend angle in degrees.
        width: Optional metal width override.
        cross_section: Metal-routing cross-section specification.
    """
    if radius is None:
        if width:
            xs = gf.get_cross_section(cross_section=cross_section, width=width)
        else:
            xs = gf.get_cross_section(cross_section=cross_section)
        radius = xs.radius or xs.width
    c = gf.c.bend_circular(
        radius=radius,
        angle=angle,
        width=width,
        cross_section=cross_section,
        allow_min_radius_violation=True,
        npoints=None,
        layer=None,
    )
    add_electrical_pins(c)
    return c


@gf.cell(tags=["waveguides"], schematic_function=bend_s_metal_schematic)
def bend_s_metal(
    size: Size = (11, 1.8),
    cross_section: CrossSectionSpec = "metal_routing",
    width: float | None = None,
    allow_min_radius_violation: bool = True,
) -> gf.Component:
    """Return S bend with bezier curve.

    stores min_bend_radius property in self.info['min_bend_radius']
    min_bend_radius depends on height and length

    Args:
        size: in x and y direction.
        cross_section: spec.
        width: width of the waveguide. If None, it will use the width of the cross_section.
        allow_min_radius_violation: allows min radius violations.
    """
    c = gf.c.bend_s(
        size=size,
        cross_section=cross_section,
        npoints=99,
        allow_min_radius_violation=allow_min_radius_violation,
        width=width,
    )
    add_electrical_pins(c)
    return c
