"""This module contains the building blocks of the cspdk.si500 library.

Foundry-matched defaults follow the CORNERSTONE 500 nm SOI standard
components (42nd call) and the reference GDS files in ``cspdk/si500/gds``.

The 500 nm platform is C-band (1550 nm) only. The ``_ro`` (O-band rib)
variants have no foundry basis: they reuse the C-band component geometry on
a 400 nm wide rib and are kept only for backwards compatibility.
"""

from functools import partial

import gdsfactory as gf
from gdsfactory.cross_section import CrossSection
from gdsfactory.typings import (
    CrossSectionSpec,
    Ints,
    LayerSpec,
    Size,
)

from cspdk._cells import _grating_coupler_rectangular
from cspdk.si500._schematic import (
    bend_circular_schematic,
    bend_euler_schematic,
    bend_s_schematic,
    coupler_schematic,
    coupler_straight_schematic,
    grating_coupler_elliptical_schematic,
    grating_coupler_rectangular_schematic,
    mmi1x2_schematic,
    mmi2x2_schematic,
    mzi_schematic,
    pad_schematic,
    straight_schematic,
    taper_schematic,
    wire_corner_schematic,
)
from cspdk.si500.tech import LAYER, Tech

################
# Straights
################


@gf.cell(tags=["cells"], schematic_function=straight_schematic)
def straight(
    length: float = 10.0,
    cross_section: CrossSectionSpec = "xs_rc500",
    **kwargs,
) -> gf.Component:
    """A straight waveguide.

    Args:
        length: the length of the waveguide.
        cross_section: a cross section or its name or a function generating a cross section.
        kwargs: additional arguments to pass to the straight function.
    """
    return gf.c.straight(length=length, cross_section=cross_section, **kwargs)


straight_rc = partial(straight, cross_section="xs_rc500")
straight_ro = partial(straight, cross_section="xs_ro500")


################
# Bends
################


@gf.cell(tags=["cells"], schematic_function=wire_corner_schematic)
def wire_corner(cross_section="metal_routing", **kwargs) -> gf.Component:
    """A wire corner.

    A wire corner is a bend for electrical routes.

    Args:
        cross_section: "metal_routing".
        **kwargs: additional arguments.
    """
    return gf.components.wire_corner(cross_section=cross_section, **kwargs)


@gf.cell(tags=["cells"], schematic_function=bend_s_schematic)
def bend_s(
    size: tuple[float, float] = (20.0, 1.8),
    cross_section: CrossSectionSpec = "xs_rc500",
    allow_min_radius_violation: bool = True,
    width: float | None = None,
) -> gf.Component:
    """An S-bend.

    Args:
        size: the width and height of the s-bend.
        cross_section: a cross section or its name or a function generating a cross section.
        allow_min_radius_violation: if True, allows the s-bend to have a smaller radius than the minimum radius.
        width: waveguide width; defaults to the cross-section width.
    """
    return gf.components.bend_s(
        size=size,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
        width=width,
    )


@gf.cell(tags=["cells"], schematic_function=bend_euler_schematic)
def bend_euler(
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.5,
    width: float | None = None,
    cross_section: CrossSectionSpec = "xs_rc500",
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


bend_euler_rc = partial(bend_euler, cross_section="xs_rc500")
bend_euler_ro = partial(bend_euler, cross_section="xs_ro500")


@gf.cell(tags=["cells"], schematic_function=bend_circular_schematic)
def bend_circular(
    radius: float | None = None,
    angle: float = 90.0,
    width: float | None = None,
    cross_section: CrossSectionSpec = "xs_rc500",
) -> gf.Component:
    """A circular bend, as the foundry 90 degree bend (R = 25 um).

    SOI500nm_1550nm_TE_RIB_90_Degree_Bend is a circular arc, so this is the
    default bend for routing and the MZI. An Euler bend (``bend_euler``) with
    the same effective radius curves more tightly than the 25 um minimum.

    Args:
        radius: the radius of the bend; defaults to the cross-section radius.
        angle: the angle of the bend (usually 90 degrees).
        width: the width of the waveguide forming the bend.
        cross_section: a cross section or its name or a function generating a cross section.
    """
    return gf.components.bend_circular(
        radius=radius,
        angle=angle,
        width=width,
        cross_section=cross_section,
        allow_min_radius_violation=False,
    )


bend_circular_rc = partial(bend_circular, cross_section="xs_rc500")
bend_circular_ro = partial(bend_circular, cross_section="xs_ro500")

################
# Transitions
################


@gf.cell(tags=["cells"], schematic_function=taper_schematic)
def taper(
    length: float = 10.0,
    width1: float | None = None,
    width2: float | None = None,
    port: gf.Port | None = None,
    cross_section: CrossSectionSpec = "xs_rc500",
) -> gf.Component:
    """A taper.

    A taper is a transition between two waveguide widths

    Args:
        length: the length of the taper.
        width1: the input width of the taper (defaults to the cross-section width).
        width2: the output width of the taper (if not given, use port).
        port: the port (with certain width) to taper towards (if not given, use width2).
        cross_section: a cross section or its name or a function generating a cross section.
    """
    if width1 is None:
        width1 = gf.get_cross_section(cross_section).width
    return gf.c.taper(
        length=length,
        width1=width1,
        width2=width2,
        port=port,
        cross_section=cross_section,
    )


taper_rc = partial(
    taper,
    cross_section="xs_rc500",
    length=10.0,
    width1=Tech.width_rc,
    width2=None,
)
taper_ro = partial(
    taper,
    cross_section="xs_ro500",
    length=10.0,
    width1=Tech.width_ro,
    width2=None,
)

################
# MMIs
################


@gf.cell(tags=["cells"], schematic_function=mmi1x2_schematic)
def mmi1x2(
    width: float | None = None,
    width_taper: float = 1.6,
    length_taper: float = 20.0,
    length_mmi: float = 37.5,
    width_mmi: float = 6.0,
    gap_mmi: float = 1.47,
    cross_section: CrossSectionSpec = "xs_rc500",
) -> gf.Component:
    """An mmi1x2.

    An mmi1x2 is a splitter that splits a single input to two outputs.
    Defaults match SOI500nm_1550nm_TE_RIB_2x1_MMI without its 10 um port
    straights: 37.5 x 6 um body, 20 um tapers to 1.6 um.

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
        straight=straight_rc,
        cross_section=cross_section,
    )


mmi1x2_rc = partial(mmi1x2, cross_section="xs_rc500")
mmi1x2_ro = partial(mmi1x2, cross_section="xs_ro500")


@gf.cell(tags=["cells"], schematic_function=mmi2x2_schematic)
def mmi2x2(
    width: float | None = None,
    width_taper: float = 1.6,
    length_taper: float = 20.0,
    length_mmi: float = 50.2,
    width_mmi: float = 6.0,
    gap_mmi: float = 0.4,
    cross_section: CrossSectionSpec = "xs_rc500",
) -> gf.Component:
    """An mmi2x2.

    An mmi2x2 is a 2x2 splitter. Defaults match SOI500nm_1550nm_TE_RIB_2x2_MMI
    without its 10 um port straights: 50.2 x 6 um body, 20 um tapers to
    1.6 um, 0.4 um gap.

    Args:
        width: the width of the waveguides connecting at the mmi ports.
        width_taper: the width at the base of the mmi body.
        length_taper: the length of the tapers going towards the mmi body.
        length_mmi: the length of the mmi body.
        width_mmi: the width of the mmi body.
        gap_mmi: the gap between the tapers at the mmi body.
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


mmi2x2_rc = partial(mmi2x2, cross_section="xs_rc500")
mmi2x2_ro = partial(mmi2x2, cross_section="xs_ro500")

##############################
# Evanescent couplers
##############################


@gf.cell(tags=["cells"], schematic_function=coupler_straight_schematic)
def coupler_straight(
    length: float = 20.0,
    gap: float = 0.236,
    cross_section: CrossSectionSpec = "xs_rc500",
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


@gf.cell(tags=["cells"], schematic_function=coupler_schematic)
def coupler(
    gap: float = 0.236,
    length: float = 20.0,
    dy: float = 4.0,
    dx: float = 15.0,
    cross_section: CrossSectionSpec = "xs_rc500",
) -> gf.Component:
    """A coupler.

    a coupler is a 2x2 splitter

    Args:
        gap: the gap between the waveguides forming the straight part of the coupler.
        length: the length of the coupler.
        dy: the height of the s-bend.
        dx: the length of the s-bend.
        cross_section: a cross section or its name or a function generating a cross section.
    """
    return gf.c.coupler(
        gap=gap,
        length=length,
        dy=dy,
        dx=dx,
        cross_section=cross_section,
    )


coupler_rc = partial(coupler, cross_section="xs_rc500")
coupler_ro = partial(coupler, cross_section="xs_ro500")


##############################
# grating couplers Rectangular
##############################


@gf.cell(tags=["cells"], schematic_function=grating_coupler_rectangular_schematic)
def grating_coupler_rectangular(
    period: float = 0.53,
    n_periods: int = 60,
    fill_factor: float = 0.5283,
    length_taper: float = 350.0,
    width_grating: float = 10.0,
    length_grating: float = 42.0,
    grating_offset: float = 7.116,
    teeth_overhang: float = 0.5,
    wavelength: float = 1.55,
    cross_section: CrossSectionSpec = "xs_rc500",
) -> gf.Component:
    """A grating coupler with straight and parallel teeth.

    Defaults reproduce SOI500nm_1550nm_TE_RIB_Grating_Coupler: a 350 um
    linear taper from the waveguide to a 10 um wide, 42 um long grating
    section with 60 teeth, each 0.28 um etched (GDS layer 6, 160 nm deep) at
    a 0.53 um period and 11 um tall, the first 7.116 um after the taper.

    Args:
        period: the period of the grating.
        n_periods: the number of grating teeth.
        fill_factor: etched tooth width (GDS layer 6) as a fraction of the period.
        length_taper: the length of the taper tapering up to the grating.
        width_grating: the width of the waveguide under the grating.
        length_grating: the length of the full-width waveguide after the taper.
        grating_offset: distance from the end of the taper to the first tooth.
        teeth_overhang: how far the teeth extend beyond each side of the waveguide.
        wavelength: the center wavelength for which the grating is designed.
        cross_section: a cross section or its name or a function generating a cross section.
    """
    return _grating_coupler_rectangular(
        period=period,
        n_periods=n_periods,
        fill_factor=fill_factor,
        length_taper=length_taper,
        width_grating=width_grating,
        length_grating=length_grating,
        grating_offset=grating_offset,
        teeth_overhang=teeth_overhang,
        wavelength=wavelength,
        cross_section=cross_section,
        layer_grating=LAYER.GRA,
    )


grating_coupler_rectangular_rc = partial(
    grating_coupler_rectangular,
    cross_section="xs_rc500",
)

# No foundry basis: the C-band grating (0.53 um period) fed by a 400 nm
# rib, not an O-band design. Kept for backwards compatibility.
grating_coupler_rectangular_ro = partial(
    grating_coupler_rectangular,
    cross_section="xs_ro500",
)


##############################
# grating couplers elliptical
##############################


@gf.cell(tags=["cells"], schematic_function=grating_coupler_elliptical_schematic)
def grating_coupler_elliptical(
    wavelength: float = 1.55,
    grating_line_width=0.315,
    cross_section="xs_rc500",
) -> gf.Component:
    """A grating coupler with curved but parallel teeth.

    Args:
        wavelength: the center wavelength for which the grating is designed.
        grating_line_width: the line width of the grating.
        cross_section: a cross section or its name or a function generating a cross section.
    """
    return gf.c.grating_coupler_elliptical_trenches(
        polarization="te",
        wavelength=wavelength,
        grating_line_width=grating_line_width,
        taper_length=16.6,
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


grating_coupler_elliptical_rc = partial(
    grating_coupler_elliptical,
    grating_line_width=0.315,
    wavelength=1.55,
    cross_section="xs_rc500",
)

grating_coupler_elliptical_ro = partial(
    grating_coupler_elliptical,
    grating_line_width=0.250,
    wavelength=1.31,
    cross_section="xs_ro500",
)

################
# MZI
################

# TODO: (needs gdsfactory fix) currently function arguments need to be
# supplied as gf.ComponentSpec strings, because when supplied as function they get
# serialized weirdly in the netlist


@gf.cell(tags=["cells"], schematic_function=mzi_schematic)
def mzi(
    delta_length: float = 10.0,
    bend="bend_circular_rc",
    straight="straight_rc",
    splitter="mmi1x2_rc",
    combiner="mmi2x2_rc",
    cross_section: CrossSectionSpec = "xs_rc500",
) -> gf.Component:
    """A Mach-Zehnder Interferometer.

    Args:
        delta_length: the difference in length between the upper and lower arms of the mzi.
        bend: the name of the default bend of the mzi.
        straight: the name of the default straight of the mzi.
        splitter: the name of the default splitter of the mzi.
        combiner: the name of the default combiner of the mzi.
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


mzi_rc = partial(
    mzi,
    straight="straight_rc",
    bend="bend_circular_rc",
    splitter="mmi1x2_rc",
    combiner="mmi2x2_rc",
    cross_section="xs_rc500",
)

mzi_ro = partial(
    mzi,
    straight="straight_ro",
    bend="bend_circular_ro",
    splitter="mmi1x2_ro",
    combiner="mmi2x2_ro",
    cross_section="xs_ro500",
)


################
# Packaging
################


@gf.cell(tags=["cells"], schematic_function=pad_schematic)
def pad(size: Size = (100.0, 100.0)) -> gf.Component:
    """An electrical pad.

    Args:
        size: the pad size in um (the packaging template uses 150 x 200 um).
    """
    return gf.c.pad(layer=LAYER.PAD, size=size)


@gf.cell(tags=["cells"])
def rectangle(layer=LAYER.FLOORPLAN, **kwargs) -> gf.Component:
    """A rectangle.

    Args:
        layer: LAYER.FLOORPLAN.
        **kwargs: additional arguments.
    """
    return gf.c.rectangle(layer=layer, **kwargs)


@gf.cell(tags=["cells"])
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


@gf.cell(tags=["cells"])
def grating_coupler_array(
    pitch: float = 127.0,
    n: int = 6,
    cross_section="xs_rc500",
    centered=True,
    grating_coupler=None,
    port_name="o1",
    with_loopback=False,
    rotation=-90,
    straight_to_grating_spacing=10.0,
    radius: float | None = None,
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
        cross_section: a cross section or its name or a function generating a cross section.
        radius: optional radius for routing the loopback.
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
            "xs_rc500": "grating_coupler_rectangular_rc",
            "xs_ro500": "grating_coupler_rectangular_ro",
        }
        grating_coupler = gcs.get(xs, "grating_coupler_rectangular")
    assert grating_coupler is not None
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


# Coordinates of Cell0_SOI500_Full_1550nm_Packaging_Template.gds (um).
_DIE_SIZE = (11470.0, 4900.0)
_DIE_GRATING_X = 4922.173  # |x| of the grating waveguide ports (o1)
_DIE_GRATING_YS = tuple(-1850.0 + 250.0 * i for i in range(16))  # per side
_DIE_LOOPBACK_RADIUS = 30.0
_DIE_LOOPBACK_X_TURN = 4842.173  # |x| of the inner loopback turns
_DIE_LOOPBACK_X_OUT = 5402.173  # |x| of the outer vertical loopback waveguide
_DIE_LOOPBACK_DY = 60.0  # outer gratings to the horizontal loopback waveguides
_DIE_PAD_SIZE = (150.0, 200.0)
_DIE_PAD_XS = tuple(-4615.0 + 300.0 * i for i in range(31))  # per side
_DIE_PAD_Y = 2300.0  # |y| of the pad centres


@gf.cell(tags=["cells"])
def die(cross_section: CrossSectionSpec = "xs_rc500") -> gf.Component:
    """A die matching the CORNERSTONE SOI500 1550 nm packaging template.

    Cell0_SOI500_Full_1550nm_Packaging_Template: 11.47 x 4.9 mm floorplan,
    16 grating couplers per side at a 250 um pitch (y = -1850 ... 1900 um)
    pointing outwards, the outermost pair on each side joined by a loopback,
    and 31 pads of 150 x 200 um per side at a 300 um pitch
    (x = -4615 ... 4385 um, y = +-2300 um).

    Args:
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
        "xs_rc500": "grating_coupler_rectangular_rc",
        "xs_ro500": "grating_coupler_rectangular_ro",
    }
    gc = gf.get_component(gcs.get(xs, "grating_coupler_rectangular"))

    c = gf.Component()
    c << gf.c.rectangle(
        size=_DIE_SIZE, layer=LAYER.FLOORPLAN, centered=True, port_type=None
    )
    ys = _DIE_GRATING_YS
    for side, sign in (("W", -1), ("E", 1)):
        ports = []
        for y in ys:
            ref = c << gc
            if sign < 0:
                ref.drotate(180)
            ref.dmove(ref.ports["o1"].center, (sign * _DIE_GRATING_X, y))
            ports.append(ref.ports["o1"])
        x_turn = sign * _DIE_LOOPBACK_X_TURN
        x_out = sign * _DIE_LOOPBACK_X_OUT
        y_bot = ys[0] - _DIE_LOOPBACK_DY
        y_top = ys[-1] + _DIE_LOOPBACK_DY
        gf.routing.route_single(
            c,
            ports[0],
            ports[-1],
            waypoints=[
                (x_turn, ys[0]),
                (x_turn, y_bot),
                (x_out, y_bot),
                (x_out, y_top),
                (x_turn, y_top),
                (x_turn, ys[-1]),
            ],
            radius=_DIE_LOOPBACK_RADIUS,
            cross_section=cross_section,
            bend="bend_circular",
            straight="straight",
        )
        for i, port in enumerate(ports[1:-1]):
            c.add_port(name=f"{side}{i}", port=port)

    pad_component = pad(size=_DIE_PAD_SIZE)
    for side, sign, port_name in (("N", 1, "e4"), ("S", -1, "e2")):
        for i, x in enumerate(_DIE_PAD_XS):
            ref = c << pad_component
            ref.dcenter = (x, sign * _DIE_PAD_Y)
            c.add_port(name=f"{side}{i}", port=ref.ports[port_name])
    return c


die_rc = partial(die, cross_section="xs_rc500")
die_ro = partial(die, cross_section="xs_ro500")


@gf.cell(tags=["cells"])
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
