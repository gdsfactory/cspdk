"""This module contains the building blocks for the CSPDK PDK."""

from functools import partial

import gdsfactory as gf
from gdsfactory.cross_section import CrossSection
from gdsfactory.typings import (
    ComponentSpec,
    CrossSectionSpec,
    Ints,
    LayerSpec,
    Size,
)

from cspdk.si220.config import PATH
from cspdk.si220.tech import LAYER, Tech

################
# Straights
################


@gf.cell
def straight(
    length: float = 10.0,
    cross_section: CrossSectionSpec = "xs_sc",
    **kwargs,
) -> gf.Component:
    """A straight waveguide.

    Args:
        length: the length of the waveguide.
        cross_section: a cross section or its name or a function generating a cross section.
        kwargs: additional arguments to pass to the straight function.
    """
    return gf.c.straight(length=length, cross_section=cross_section, **kwargs)


straight_sc = partial(straight, cross_section="xs_sc")
straight_so = partial(straight, cross_section="xs_so")
straight_rc = partial(straight, cross_section="xs_rc")
straight_ro = partial(straight, cross_section="xs_ro")
straight_metal = partial(straight, cross_section="metal_routing")

################
# Bends
################


@gf.cell
def wire_corner(cross_section="metal_routing", **kwargs) -> gf.Component:
    """A wire corner.

    A wire corner is a bend for electrical routes.
    """
    return gf.components.wire_corner(cross_section=cross_section, **kwargs)


@gf.cell
def bend_s(
    size: tuple[float, float] = (11.0, 1.8),
    cross_section: CrossSectionSpec = "xs_sc",
    **kwargs,
) -> gf.Component:
    """An S-bend.

    Args:
        size: the width and height of the s-bend
        cross_section: a cross section or its name or a function generating a cross section.
        kwargs: additional arguments to pass to the bend_s function.
    """
    return gf.components.bend_s(size=size, cross_section=cross_section, **kwargs)


@gf.cell
def bend_euler(
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.5,
    width: float | None = None,
    cross_section: CrossSectionSpec = "xs_sc",
) -> gf.Component:
    """An euler bend.

    Args:
        radius: the effective radius of the bend.
        angle: the angle of the bend (usually 90 degrees).
        p: the fraction of the bend that's represented by a polar bend.
        width: the width of the waveguide forming the bend.
        cross_section: a cross section or its name or a function generating a cross section.
    """
    return gf.components.bend_euler(
        radius=radius,
        angle=angle,
        p=p,
        with_arc_floorplan=True,
        npoints=None,
        layer=None,
        width=width,
        cross_section=cross_section,
        allow_min_radius_violation=False,
    )


bend_euler_sc = partial(bend_euler, cross_section="xs_sc")
bend_euler_so = partial(bend_euler, cross_section="xs_so")
bend_euler_rc = partial(bend_euler, cross_section="xs_rc")
bend_euler_ro = partial(bend_euler, cross_section="xs_ro")
bend_metal = partial(bend_euler_sc, cross_section="metal_routing")

################
# Transitions
################


@gf.cell
def taper(
    length: float = 10.0,
    width1: float = Tech.width_sc,
    width2: float | None = None,
    port: gf.Port | None = None,
    cross_section: CrossSectionSpec = "xs_sc",
    **kwargs,
) -> gf.Component:
    """A taper.

    A taper is a transition between two waveguide widths

    Args:
        length: the length of the taper.
        width1: the input width of the taper.
        width2: the output width of the taper (if not given, use port).
        port: the port (with certain width) to taper towards (if not given, use width2).
        cross_section: a cross section or its name or a function generating a cross section.
        kwargs: additional arguments to pass to the taper function.
    """
    c = gf.c.taper(
        length=length,
        width1=width1,
        width2=width2,
        port=port,
        cross_section=cross_section,
        **kwargs,
    )
    return c


taper_sc = partial(
    taper,
    cross_section="xs_sc",
    length=10.0,
    width1=Tech.width_sc,
    width2=None,
)
taper_so = partial(
    taper,
    cross_section="xs_so",
    length=10.0,
    width1=Tech.width_so,
    width2=None,
)
taper_rc = partial(
    taper,
    cross_section="xs_rc",
    length=10.0,
    width1=Tech.width_rc,
    width2=None,
)
taper_ro = partial(
    taper,
    cross_section="xs_ro",
    length=10.0,
    width1=Tech.width_ro,
    width2=None,
)


@gf.cell
def taper_strip_to_ridge(
    length: float = 10.0,
    width1: float = 0.5,
    width2: float = 0.5,
    w_slab1: float = 0.2,
    w_slab2: float = 10.45,
    cross_section: CrossSectionSpec = "xs_sc",
) -> gf.Component:
    """A taper between strip and ridge.

    This is a transition between two distinct cross sections

    Args:
        length: the length of the taper.
        width1: the input width of the taper.
        width2: the output width of the taper.
        w_slab1: the input slab width of the taper.
        w_slab2: the output slab width of the taper.
        cross_section: a cross section or its name or a function generating a cross section.
    """
    return gf.c.taper_strip_to_ridge(
        length=length,
        width1=width1,
        width2=width2,
        w_slab1=w_slab1,
        w_slab2=w_slab2,
        cross_section=cross_section,
        layer_wg=LAYER.WG,
        layer_slab=LAYER.SLAB,
    )


trans_sc_rc10 = partial(taper_strip_to_ridge, length=10)
trans_sc_rc20 = partial(taper_strip_to_ridge, length=20)
trans_sc_rc50 = partial(taper_strip_to_ridge, length=50)


################
# MMIs
################


@gf.cell
def mmi1x2(
    width: float | None = None,
    width_taper=1.5,
    length_taper=20.0,
    length_mmi: float = 5.5,
    width_mmi=6.0,
    gap_mmi: float = 0.25,
    cross_section: CrossSectionSpec = "xs_sc",
) -> gf.Component:
    """An mmi1x2.

    An mmi1x2 is a splitter that splits a single input to two outputs

    Args:
        width: the width of the waveguides connecting at the mmi ports.
        width_taper: the width at the base of the mmi body.
        length_taper: the length of the tapers going towards the mmi body.
        length_mmi: the length of the mmi body.
        width_mmi: the width of the mmi body.
        gap_mmi: the gap between the tapers at the mmi body.
        cross_section: a cross section or its name or a function generating a cross section.
    """
    return gf.c.mmi1x2(
        width=width,
        width_taper=width_taper,
        length_taper=length_taper,
        length_mmi=length_mmi,
        width_mmi=width_mmi,
        gap_mmi=gap_mmi,
        taper=taper,
        straight=straight,
        cross_section=cross_section,
    )


mmi1x2_sc = partial(mmi1x2, length_mmi=31.8, gap_mmi=1.64, cross_section="xs_sc")
mmi1x2_so = partial(mmi1x2, length_mmi=40.1, gap_mmi=1.55, cross_section="xs_so")
mmi1x2_rc = partial(mmi1x2, length_mmi=32.7, gap_mmi=1.64, cross_section="xs_rc")
mmi1x2_ro = partial(mmi1x2, length_mmi=40.8, gap_mmi=1.55, cross_section="xs_ro")


@gf.cell
def mmi2x2(
    width: float | None = None,
    width_taper: float = 1.5,
    length_taper: float = 20.0,
    length_mmi: float = 5.5,
    width_mmi: float = 6.0,
    gap_mmi: float = 0.25,
    cross_section: CrossSectionSpec = "xs_sc",
) -> gf.Component:
    """An mmi2x2.

    An mmi2x2 is a 2x2 splitter

    Args:
        width: the width of the waveguides connecting at the mmi ports
        width_taper: the width at the base of the mmi body
        length_taper: the length of the tapers going towards the mmi body
        length_mmi: the length of the mmi body
        width_mmi: the width of the mmi body
        gap_mmi: the gap between the tapers at the mmi body
        cross_section: a cross section or its name or a function generating a cross section.
    """
    return gf.c.mmi2x2(
        width=width,
        width_taper=width_taper,
        length_taper=length_taper,
        length_mmi=length_mmi,
        width_mmi=width_mmi,
        gap_mmi=gap_mmi,
        taper=taper,
        straight=straight,
        cross_section=cross_section,
    )


mmi2x2_sc = partial(mmi2x2, length_mmi=42.5, gap_mmi=0.5, cross_section="xs_sc")
mmi2x2_so = partial(mmi2x2, length_mmi=53.5, gap_mmi=0.53, cross_section="xs_so")
mmi2x2_rc = partial(mmi2x2, length_mmi=44.8, gap_mmi=0.53, cross_section="xs_rc")
mmi2x2_ro = partial(mmi2x2, length_mmi=55.0, gap_mmi=0.53, cross_section="xs_ro")


##############################
# Evanescent couplers
##############################


@gf.cell
def coupler_straight(
    length: float = 20.0,
    gap: float = 0.27,
    cross_section: CrossSectionSpec = "xs_sc",
) -> gf.Component:
    """The straight part of a coupler.

    Args:
        length: the length of the straight part of the coupler.
        gap: the gap between the waveguides forming the straight part of the coupler.
        cross_section: a cross section or its name or a function generating a cross section.
    """
    return gf.c.coupler_straight(
        length=length,
        gap=gap,
        cross_section=cross_section,
    )


@gf.cell
def coupler_symmetric(
    gap: float = 0.234,
    dy: float = 4.0,
    dx: float = 10.0,
    cross_section: CrossSectionSpec = "xs_sc",
) -> gf.Component:
    """The part of the coupler that diverges away from each other with s-bends.

    Args:
        gap: the gap between the s-bends when closest together.
        dy: the height of the s-bend.
        dx: the length of the s-bend.
        cross_section: a cross section or its name or a function generating a cross section..
    """
    return gf.c.coupler_symmetric(
        bend="bend_s",
        gap=gap,
        dy=dy,
        dx=dx,
        cross_section=cross_section,
    )


@gf.cell
def coupler(
    gap: float = 0.234,
    length: float = 20.0,
    dy: float = 4.0,
    dx: float = 10.0,
    cross_section: CrossSectionSpec = "xs_sc",
) -> gf.Component:
    """A coupler.

    a coupler is a 2x2 splitter

    Args:
        gap: the gap between the waveguides forming the straight part of the coupler
        length: the length of the coupler
        dy: the height of the s-bend
        dx: the length of the s-bend
        cross_section: a cross section or its name or a function generating a cross section.
    """
    return gf.c.coupler(
        gap=gap,
        length=length,
        dy=dy,
        dx=dx,
        cross_section=cross_section,
    )


coupler_sc = partial(
    coupler,
    cross_section="xs_sc",
)
coupler_so = partial(
    coupler,
    cross_section="xs_so",
)
coupler_rc = partial(
    coupler,
    cross_section="xs_rc",
    dx=15,
    dy=4.0,
)
coupler_ro = partial(
    coupler,
    cross_section="xs_ro",
    dx=15,
    dy=4.0,
)


##############################
# grating couplers Rectangular
##############################


@gf.cell
def grating_coupler_rectangular(
    period=0.315 * 2,
    n_periods: int = 30,
    length_taper: float = 350.0,
    wavelength: float = 1.55,
    cross_section="xs_sc",
) -> gf.Component:
    """A grating coupler with straight and parallel teeth.

    Args:
        period: the period of the grating
        n_periods: the number of grating teeth
        length_taper: the length of the taper tapering up to the grating
        wavelength: the center wavelength for which the grating is designed
        cross_section: a cross section or its name or a function generating a cross section.
    """
    return gf.c.grating_coupler_rectangular(
        n_periods=n_periods,
        period=period,
        fill_factor=0.5,
        width_grating=11.0,
        length_taper=length_taper,
        polarization="te",
        wavelength=wavelength,
        taper=taper,
        layer_slab=LAYER.WG,
        layer_grating=LAYER.GRA,
        fiber_angle=10.0,
        slab_xmin=-1.0,
        slab_offset=0.0,
        cross_section=cross_section,
    )


grating_coupler_rectangular_sc = partial(
    grating_coupler_rectangular,
    wavelength=1.55,
    cross_section="xs_sc",
    n_periods=60,
)

grating_coupler_rectangular_so = partial(
    grating_coupler_rectangular,
    wavelength=1.31,
    cross_section="xs_so",
    n_periods=80,
    period=0.250 * 2,
)

grating_coupler_rectangular_rc = partial(
    grating_coupler_rectangular,
    period=0.5,
    cross_section="xs_rc",
    n_periods=60,
)

grating_coupler_rectangular_ro = partial(
    grating_coupler_rectangular,
    period=0.5,
    cross_section="xs_ro",
    n_periods=80,
)


##############################
# grating couplers elliptical
##############################


@gf.cell
def grating_coupler_elliptical(
    wavelength: float = 1.55,
    grating_line_width=0.315,
    cross_section="xs_sc",
) -> gf.Component:
    """A grating coupler with curved but parallel teeth.

    Args:
        wavelength: the center wavelength for which the grating is designed
        grating_line_width: the line width of the grating
        cross_section: a cross section or its name or a function generating a cross section.
    """
    return gf.c.grating_coupler_elliptical_trenches(
        polarization="te",
        wavelength=wavelength,
        grating_line_width=grating_line_width,
        taper_length=16.0,
        taper_angle=30.0,
        trenches_extra_angle=9.0,
        fiber_angle=15.0,
        neff=2.638,
        ncladding=1.443,
        layer_trench=LAYER.GRA,
        p_start=26,
        n_periods=30,
        end_straight_length=0.2,
        cross_section=cross_section,
    )


grating_coupler_elliptical_sc = partial(
    grating_coupler_elliptical,
    grating_line_width=0.315,
    wavelength=1.55,
    cross_section="xs_sc",
)

grating_coupler_elliptical_so = partial(
    grating_coupler_elliptical,
    grating_line_width=0.250,
    wavelength=1.31,
    cross_section="xs_so",
)


################
# MZI
################

# TODO: (needs gdsfactory fix) currently function arguments need to be
# supplied as ComponentSpec strings, because when supplied as function they get
# serialized weirdly in the netlist


@gf.cell
def mzi(
    delta_length: float = 10.0,
    bend="bend_euler_sc",
    straight="straight_sc",
    splitter="mmi1x2_sc",
    combiner="mmi2x2_sc",
    cross_section: CrossSectionSpec = "xs_sc",
) -> gf.Component:
    """A Mach-Zehnder Interferometer.

    Args:
        delta_length: the difference in length between the upper and lower arms of the mzi
        bend: the name of the default bend of the mzi
        straight: the name of the default straight of the mzi
        splitter: the name of the default splitter of the mzi
        combiner: the name of the default combiner of the mzi
        cross_section: a cross section or its name or a function generating a cross section.
    """
    return gf.c.mzi(
        delta_length=delta_length,
        length_y=1.0,
        length_x=0.1,
        straight_y=None,
        straight_x_top=None,
        straight_x_bot=None,
        with_splitter=True,
        port_e1_splitter="o2",
        port_e0_splitter="o3",
        port_e1_combiner="o3",
        port_e0_combiner="o4",
        nbends=2,
        cross_section=cross_section,
        cross_section_x_top=None,
        cross_section_x_bot=None,
        mirror_bot=False,
        add_optical_ports_arms=False,
        min_length=10e-3,
        auto_rename_ports=True,
        bend=bend,
        straight=straight,
        splitter=splitter,
        combiner=combiner,
    )


mzi_sc = partial(
    mzi,
    straight="straight_sc",
    bend="bend_euler_sc",
    splitter="mmi1x2_sc",
    combiner="mmi2x2_sc",
    cross_section="xs_sc",
)

mzi_so = partial(
    mzi,
    straight="straight_so",
    bend="bend_euler_so",
    splitter="mmi1x2_so",
    combiner="mmi2x2_so",
    cross_section="xs_so",
)

mzi_rc = partial(
    mzi,
    straight="straight_rc",
    bend="bend_euler_rc",
    splitter="mmi1x2_rc",
    combiner="mmi2x2_rc",
    cross_section="xs_rc",
)

mzi_ro = partial(
    mzi,
    straight="straight_ro",
    bend="bend_euler_ro",
    splitter="mmi1x2_ro",
    combiner="mmi2x2_ro",
    cross_section="xs_ro",
)


@gf.cell
def coupler_ring_sc(
    gap: float = 0.2,
    radius: float = 5.0,
    length_x: float = 4.0,
    bend: ComponentSpec = "bend_euler_sc",
    straight: ComponentSpec = "straight_sc",
    cross_section: CrossSectionSpec = "xs_sc",
    cross_section_bend: CrossSectionSpec | None = None,
    length_extension: float = 3,
) -> gf.Component:
    r"""Coupler for ring.

    Args:
        gap: spacing between parallel coupled straight waveguides.
        radius: of the bends.
        length_x: length of the parallel coupled straight waveguides.
        bend: 90 degrees bend spec.
        straight: straight spec.
        cross_section: cross_section spec.
        cross_section_bend: optional bend cross_section spec.
        length_extension: for the ports.

    .. code::

          o2            o3
           |             |
            \           /
             \         /
           ---=========---
        o1    length_x   o4

    """
    return gf.components.coupler_ring(
        gap,
        radius,
        length_x,
        bend,
        straight,
        cross_section,
        cross_section_bend,
        length_extension,
    )


coupler_ring_so = partial(
    coupler_ring_sc,
    cross_section="xs_so",
    bend="bend_euler_so",
    straight="straight_so",
)


@gf.cell
def ring_single_sc(
    gap: float = 0.2,
    radius: float = 10.0,
    length_x: float = 4.0,
    length_y: float = 0.6,
    bend: ComponentSpec = "bend_euler_sc",
    straight: ComponentSpec = "straight_sc",
    coupler_ring: ComponentSpec = "coupler_ring_sc",
    cross_section: CrossSectionSpec = "xs_sc",
) -> gf.Component:
    """Returns a single ring.

    ring coupler (cb: bottom) connects to two vertical straights (sl: left, sr: right),
    two bends (bl, br) and horizontal straight (wg: top)

    Args:
        gap: gap between for coupler.
        radius: for the bend and coupler.
        length_x: ring coupler length.
        length_y: vertical straight length.
        coupler_ring: ring coupler spec.
        bend: 90 degrees bend spec.
        straight: straight spec.
        coupler_ring: ring coupler spec.
        cross_section: cross_section spec.

    .. code::

                    xxxxxxxxxxxxx
                xxxxx           xxxx
              xxx                   xxx
            xxx                       xxx
           xx                           xxx
           x                             xxx
          xx                              xx▲
          xx                              xx│length_y
          xx                              xx▼
          xx                             xx
           xx          length_x          x
            xx     ◄───────────────►    x
             xx                       xxx
               xx                   xxx
                xxx──────▲─────────xxx
                         │gap
                 o1──────▼─────────o2
    """
    return gf.components.ring_single(
        gap, radius, length_x, length_y, bend, straight, coupler_ring, cross_section
    )


ring_single_so = partial(
    ring_single_sc,
    cross_section="xs_so",
    bend="bend_euler_so",
    straight="straight_so",
    coupler_ring="coupler_ring_so",
)


@gf.cell
def via_stack_heater_mtop(size=(10, 10)) -> gf.Component:
    """Returns a via stack for the heater."""
    return gf.c.via_stack(
        size=size,
        layers=("HEATER", "PAD"),
        vias=(None, None),
    )


@gf.cell
def straight_heater_metal_sc(
    length: float = 320.0,
    length_undercut_spacing: float = 6.0,
    length_undercut: float = 30.0,
    length_straight: float = 0.1,
    length_straight_input: float = 15.0,
    cross_section: CrossSectionSpec = "xs_sc",
    cross_section_heater: CrossSectionSpec = "heater_metal",
    cross_section_waveguide_heater: CrossSectionSpec = "xs_sc_heater",
    cross_section_heater_undercut: CrossSectionSpec = "xs_sc_heater",
    with_undercut: bool = True,
    via_stack: ComponentSpec | None = "via_stack_heater_mtop",
    port_orientation1: int | None = 180,
    port_orientation2: int | None = 0,
    heater_taper_length: float | None = 5.0,
    ohms_per_square: float | None = None,
) -> gf.Component:
    """Returns a thermal phase shifter.

    dimensions from https://doi.org/10.1364/OE.27.010456

    Args:
        length: of the waveguide.
        length_undercut_spacing: from undercut regions.
        length_undercut: length of each undercut section.
        length_straight: length of the straight waveguide.
        length_straight_input: from input port to where trenches start.
        cross_section: for waveguide ports.
        cross_section_heater: for heated sections. heater metal only.
        cross_section_waveguide_heater: for heated sections.
        cross_section_heater_undercut: for heated sections with undercut.
        with_undercut: isolation trenches for higher efficiency.
        via_stack: via stack.
        port_orientation1: left via stack port orientation.
        port_orientation2: right via stack port orientation.
        heater_taper_length: minimizes current concentrations from heater to via_stack.
        ohms_per_square: to calculate resistance.
    """
    return gf.components.straight_heater_metal_undercut(
        length,
        length_undercut_spacing,
        length_undercut,
        length_straight,
        length_straight_input,
        cross_section,
        cross_section_heater,
        cross_section_waveguide_heater,
        cross_section_heater_undercut,
        with_undercut,
        via_stack,
        port_orientation1,
        port_orientation2,
        heater_taper_length,
        ohms_per_square,
    )


straight_heater_metal_so = partial(
    straight_heater_metal_sc,
    cross_section="xs_so",
    cross_section_waveguide_heater="xs_so_heater",
    cross_section_heater_undercut="xs_so_heater",
)


################
# Packaging
################


@gf.cell
def pad() -> gf.Component:
    """An electrical pad."""
    return gf.c.pad(layer=LAYER.PAD, size=(100.0, 100.0))


@gf.cell
def rectangle(
    size: Size = (4.0, 2.0),
    layer: str = "FLOORPLAN",
    centered: bool = False,
    **kwargs,
) -> gf.Component:
    """A rectangle.

    Args:
        size: Width and height of rectangle.
        layer: Specific layer to put polygon geometry on.
        centered: True sets center to (0, 0), False sets south-west to (0, 0).
        kwargs: additional arguments to pass to the rectangle function.
    """
    return gf.c.rectangle(
        size=size,
        layer=layer,
        centered=centered,
        **kwargs,
    )


@gf.cell
def compass(
    size: Size = (4.0, 2.0),
    layer: LayerSpec = "PAD",
    port_type: str | None = "electrical",
    port_inclusion: float = 0.0,
    port_orientations: Ints | None = (180, 90, 0, -90),
    auto_rename_ports: bool = True,
) -> gf.Component:
    """Rectangle with ports on each edge (north, south, east, and west).

    Args:
        size: rectangle size.
        layer: tuple (int, int).
        port_type: optical, electrical.
        port_inclusion: from edge.
        port_orientations: list of port_orientations to add. None does not add ports.
        auto_rename_ports: auto rename ports.
    """
    return gf.c.compass(
        size=size,
        layer=layer,
        port_type=port_type,
        port_inclusion=port_inclusion,
        port_orientations=port_orientations,
        auto_rename_ports=auto_rename_ports,
    )


@gf.cell
def grating_coupler_array(
    pitch: float = 127.0,
    n: int = 6,
    centered=True,
    grating_coupler=None,
    port_name="o1",
    with_loopback=False,
    rotation=-90,
    straight_to_grating_spacing=10.0,
    radius: float | None = None,
    cross_section="xs_sc",
) -> gf.Component:
    """An array of grating couplers.

    Args:
        pitch: the pitch of the grating couplers
        n: the number of grating couplers
        centered: if True, centers the array around the origin.
        grating_coupler: the name of the grating coupler to use in the array.
        port_name: port name
        with_loopback: if True, adds a loopback between edge GCs. Only works for rotation = 90 for now.
        rotation: rotation angle for each reference.
        straight_to_grating_spacing: spacing between the last grating coupler and the loopback.
        radius: optional radius for routing the loopback.
        cross_section: a cross section or its name or a function generating a cross section.
    """
    if grating_coupler is None:
        if isinstance(cross_section, str):
            xs = cross_section
        elif callable(cross_section):
            xs = cross_section().name
        elif isinstance(cross_section, CrossSection):
            xs = cross_section.name
        else:
            xs = ""
        gcs = {
            "xs_sc": "grating_coupler_rectangular_sc",
            "xs_so": "grating_coupler_rectangular_so",
            "xs_rc": "grating_coupler_rectangular_rc",
            "xs_ro": "grating_coupler_rectangular_ro",
        }
        grating_coupler = gcs.get(xs, "grating_coupler_rectangular")

    if grating_coupler is None:
        raise ValueError("grating_coupler is None")
    return gf.c.grating_coupler_array(
        grating_coupler=grating_coupler,
        pitch=pitch,
        n=n,
        with_loopback=with_loopback,
        rotation=rotation,
        straight_to_grating_spacing=straight_to_grating_spacing,
        port_name=port_name,
        centered=centered,
        cross_section=cross_section,
        radius=radius,
    )


@gf.cell
def die(
    size: Size = (11470.0, 4900.0),
    edge_to_grating_distance: float = 150.0,
    edge_to_pad_distance: float = 150.0,
    grating_coupler=None,
    grating_pitch: float = 250.0,
    layer_floorplan=LAYER.FLOORPLAN,
    ngratings: int = 14,
    npads: int = 31,
    pad: str = "pad",
    pad_pitch: float = 300.0,
    cross_section="xs_sc",
) -> gf.Component:
    """A die template.

    Args:
        size: the size of the die.
        edge_to_grating_distance: the distance from the edge to the grating couplers.
        edge_to_pad_distance: the distance from the edge to the pads.
        grating_coupler: the name of the grating coupler to use in the array.
        grating_pitch: the pitch of the grating couplers.
        layer_floorplan: the layer to use for the floorplan.
        ngratings: the number of grating couplers.
        npads: the number of pads.
        pad: the name of the pad to use in the array.
        pad_pitch: the pitch of the pads.
        cross_section: a cross section or its name or a function generating a cross section.
    """
    if isinstance(cross_section, str):
        xs = cross_section
    elif callable(cross_section):
        xs = cross_section().name
    elif isinstance(cross_section, CrossSection):
        xs = cross_section.name
    else:
        xs = ""
    gcs = {
        "xs_sc": "grating_coupler_rectangular_sc",
        "xs_so": "grating_coupler_rectangular_so",
        "xs_rc": "grating_coupler_rectangular_rc",
        "xs_ro": "grating_coupler_rectangular_ro",
    }
    grating_coupler = grating_coupler or gcs.get(xs, "grating_coupler_rectangular")
    return gf.c.die_with_pads(
        size=size,
        edge_to_grating_distance=edge_to_grating_distance,
        edge_to_pad_distance=edge_to_pad_distance,
        grating_coupler=grating_coupler,
        grating_pitch=grating_pitch,
        layer_floorplan=layer_floorplan,
        ngratings=ngratings,
        npads=npads,
        pad=pad,
        pad_pitch=pad_pitch,
        cross_section=cross_section,
    )


die_sc = partial(die, cross_section="xs_sc")
die_so = partial(die, cross_section="xs_so")
die_rc = partial(die, cross_section="xs_rc")
die_ro = partial(die, cross_section="xs_ro")


################
# Imported from Cornerstone MPW SOI 220nm GDSII Template
################


@gf.cell
def heater() -> gf.Component:
    """Heater fixed cell."""
    return gf.import_gds(PATH.gds / "Heater.gds")


@gf.cell
def crossing_so() -> gf.Component:
    """SOI220nm_1310nm_TE_STRIP_Waveguide_Crossing fixed cell."""
    c = gf.import_gds(PATH.gds / "SOI220nm_1310nm_TE_STRIP_Waveguide_Crossing.gds")
    xc = 493.47
    dx = 8.47 / 2
    x = xc - dx
    xl = xc - 2 * dx
    xr = xc
    yb = -dx
    yt = +dx

    c.add_port("o1", orientation=180, center=(xl, 0), width=0.4, layer=LAYER.WG)
    c.add_port("o2", orientation=90, center=(x, yt), width=0.4, layer=LAYER.WG)
    c.add_port("o3", orientation=0, center=(xr, 0), width=0.4, layer=LAYER.WG)
    c.add_port("o4", orientation=270, center=(x, yb), width=0.4, layer=LAYER.WG)
    return c


@gf.cell
def crossing_rc() -> gf.Component:
    """SOI220nm_1550nm_TE_RIB_Waveguide_Crossing fixed cell."""
    c = gf.import_gds(PATH.gds / "SOI220nm_1550nm_TE_RIB_Waveguide_Crossing.gds")
    xc = 404.24
    dx = 9.24 / 2
    x = xc - dx
    xl = xc - 2 * dx
    xr = xc
    yb = -dx
    yt = +dx
    width = 0.45

    c.add_port("o1", orientation=180, center=(xl, 0), width=width, layer=LAYER.WG)
    c.add_port("o2", orientation=90, center=(x, yt), width=width, layer=LAYER.WG)
    c.add_port("o3", orientation=0, center=(xr, 0), width=width, layer=LAYER.WG)
    c.add_port("o4", orientation=270, center=(x, yb), width=width, layer=LAYER.WG)
    return c


@gf.cell
def crossing_sc() -> gf.Component:
    """SOI220nm_1550nm_TE_STRIP_Waveguide_Crossing fixed cell."""
    c = gf.import_gds(PATH.gds / "SOI220nm_1550nm_TE_STRIP_Waveguide_Crossing.gds")
    xc = 494.24
    yc = 800
    dx = 9.24 / 2
    x = xc - dx
    xl = xc - 2 * dx
    xr = xc
    yb = yc - dx
    yt = yc + dx
    width = 0.45

    c.add_port("o1", orientation=180, center=(xl, yc), width=width, layer=LAYER.WG)
    c.add_port("o2", orientation=90, center=(x, yt), width=width, layer=LAYER.WG)
    c.add_port("o3", orientation=0, center=(xr, yc), width=width, layer=LAYER.WG)
    c.add_port("o4", orientation=270, center=(x, yb), width=width, layer=LAYER.WG)
    return c


@gf.cell
def array(
    component="pad",
    columns: int = 6,
    rows: int = 1,
    add_ports: bool = True,
    size=None,
    centered: bool = False,
    column_pitch: float = 150,
    row_pitch: float = 150,
) -> gf.Component:
    """An array of components.

    Args:
        component: the component of which to create an array
        columns: the number of components to place in the x-direction
        rows: the number of components to place in the y-direction
        add_ports: add ports to the component
        size: Optional x, y size. Overrides columns and rows.
        centered: center the array around the origin.
        row_pitch: the pitch between rows.
        column_pitch: the pitch between columns.
    """
    return gf.c.array(
        component=component,
        columns=columns,
        rows=rows,
        size=size,
        centered=centered,
        add_ports=add_ports,
        column_pitch=column_pitch,
        row_pitch=row_pitch,
    )


pack_doe = gf.c.pack_doe
pack_doe_grid = gf.c.pack_doe_grid


if __name__ == "__main__":
    c = crossing_so()
    c.pprint_ports()
    c.show()
