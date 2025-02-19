"""This module contains the building blocks for the SiN300 technology."""

from functools import partial

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.cross_section import CrossSection
from gdsfactory.typings import (
    CrossSectionSpec,
    Ints,
    LayerSpec,
    Size,
)

from cspdk.sin300.tech import LAYER, Tech

################
# Straights
################


@gf.cell
def straight(
    length: float = 10.0,
    cross_section: CrossSectionSpec = "xs_nc",
    **kwargs,
) -> Component:
    """A straight waveguide.

    Args:
        length: the length of the waveguide.
        cross_section: a cross section or its name or a function generating a cross section.
        kwargs: additional arguments to pass to the straight function.
    """
    return gf.c.straight(length=length, cross_section=cross_section, **kwargs)


straight_nc = partial(straight, cross_section="xs_nc")
straight_no = partial(straight, cross_section="xs_no")

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
    size: tuple[float, float] = (15.0, 1.8),
    cross_section: CrossSectionSpec = "xs_nc",
) -> Component:
    """An S-bend.

    Args:
        size: the width and height of the s-bend
        cross_section: a cross section or its name or a function generating a cross section.
    """
    return gf.components.bend_s(
        size=size,
        cross_section=cross_section,
        allow_min_radius_violation=True,  # TODO: fix without this flag
    )


@gf.cell
def bend_euler(
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.5,
    width: float | None = None,
    cross_section: CrossSectionSpec = "xs_nc",
) -> Component:
    """An euler bend.

    Args:
        radius: the effective radius of the bend
        angle: the angle of the bend (usually 90 degrees)
        p: the fraction of the bend that's represented by a polar bend
        width: the width of the waveguide forming the bend
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


bend_euler_nc = partial(bend_euler, cross_section="xs_nc")
bend_euler_no = partial(bend_euler, cross_section="xs_no")

################
# Transitions
################


@gf.cell
def taper(
    length: float = 10.0,
    width1: float = Tech.width_nc,
    width2: float | None = None,
    port: gf.Port | None = None,
    cross_section: CrossSectionSpec = "xs_nc",
) -> Component:
    """A taper.

    A taper is a transition between two waveguide widths

    Args:
        length: the length of the taper
        width1: the input width of the taper
        width2: the output width of the taper (if not given, use port)
        port: the port (with certain width) to taper towards (if not given, use width2)
        cross_section: a cross section or its name or a function generating a cross section.
    """
    return gf.c.taper(
        length=length,
        width1=width1,
        width2=width2,
        port=port,
        cross_section=cross_section,
    )


taper_nc = partial(
    taper,
    cross_section="xs_nc",
    length=10.0,
    width1=Tech.width_nc,
    width2=None,
)
taper_no = partial(
    taper,
    cross_section="xs_no",
    length=10.0,
    width1=Tech.width_no,
    width2=None,
)


################
# MMIs
################


@gf.cell
def mmi1x2(
    width: float | None = None,
    width_mmi: float = 12.0,
    width_taper: float = 5.5,
    length_taper: float = 50.0,
    length_mmi: float = 64.7,
    gap_mmi: float = 0.4,
    cross_section: CrossSectionSpec = "xs_nc",
) -> Component:
    """An mmi1x2.

    An mmi1x2 is a splitter that splits a single input to two outputs

    Args:
        width: the width of the waveguides connecting at the mmi ports
        width_taper: the width at the base of the mmi body
        length_taper: the length of the tapers going towards the mmi body
        length_mmi: the length of the mmi body
        width_mmi: the width of the mmi body
        gap_mmi: the gap between the tapers at the mmi body
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


mmi1x2_nc = partial(mmi1x2, length_mmi=64.7, cross_section="xs_nc")
mmi1x2_no = partial(mmi1x2, length_mmi=42.0, cross_section="xs_no")


@gf.cell
def mmi2x2(
    width: float | None = None,
    width_taper: float = 5.5,
    length_taper: float = 50.0,
    length_mmi: float = 5.5,
    width_mmi: float = 12.0,
    gap_mmi: float = 0.4,
    cross_section: CrossSectionSpec = "xs_nc",
) -> Component:
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


mmi2x2_nc = partial(mmi2x2, length_mmi=232.0, cross_section="xs_nc")
mmi2x2_no = partial(mmi2x2, length_mmi=126.0, cross_section="xs_no")


##############################
# Evanescent couplers
##############################


@gf.cell
def coupler_straight(
    length: float = 20.0,
    gap: float = 0.236,
    cross_section: CrossSectionSpec = "xs_nc",
) -> Component:
    """The straight part of a coupler.

    Args:
        length: the length of the straight part of the coupler
        gap: the gap between the waveguides forming the straight part of the coupler
        cross_section: a cross section or its name or a function generating a cross section.
    """
    return gf.c.coupler_straight(
        length=length,
        gap=gap,
        cross_section=cross_section,
    )


@gf.cell
def coupler(
    gap: float = 0.236,
    length: float = 20.0,
    dy: float = 4.0,
    dx: float = 20.0,
    cross_section: CrossSectionSpec = "xs_nc",
) -> Component:
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


coupler_nc = partial(coupler, cross_section="xs_nc")
coupler_no = partial(coupler, cross_section="xs_no")


##############################
# grating couplers Rectangular
##############################


@gf.cell
def grating_coupler_rectangular(
    period: float = 0.66,
    n_periods: int = 30,
    length_taper: float = 200.0,
    wavelength: float = 1.55,
    cross_section="xs_nc",
) -> Component:
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
        length_taper=length_taper,
        wavelength=wavelength,
        taper=taper,
        cross_section=cross_section,
        fill_factor=0.5,
        width_grating=11.0,
        polarization="te",
        layer_slab=LAYER.NITRIDE,
        layer_grating=LAYER.NITRIDE_ETCH,
        fiber_angle=20.0,
        slab_xmin=-1.0,
        slab_offset=0.0,
    )


grating_coupler_rectangular_nc = partial(
    grating_coupler_rectangular,
    period=0.66,
    wavelength=1.55,
    cross_section="xs_nc",
)

grating_coupler_rectangular_no = partial(
    grating_coupler_rectangular,
    period=0.964,
    wavelength=1.31,
    cross_section="xs_no",
)


##############################
# grating couplers elliptical
##############################


@gf.cell
def grating_coupler_elliptical(
    wavelength: float = 1.55,
    grating_line_width=0.343,
    cross_section="xs_nc",
) -> Component:
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
        taper_length=16.6,
        taper_angle=30.0,
        trenches_extra_angle=9.0,
        fiber_angle=20.0,
        neff=1.6,
        ncladding=1.443,
        layer_trench=LAYER.GRA,
        p_start=26,
        n_periods=30,
        end_straight_length=0.2,
        cross_section=cross_section,
    )


grating_coupler_elliptical_nc = partial(
    grating_coupler_elliptical,
    grating_line_width=0.66 / 2,
    wavelength=1.55,
    cross_section="xs_nc",
)

grating_coupler_elliptical_no = partial(
    grating_coupler_elliptical,
    grating_line_width=0.964 / 2,
    wavelength=1.31,
    cross_section="xs_no",
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
    bend="bend_euler_nc",
    straight="straight_nc",
    splitter="mmi1x2_nc",
    combiner="mmi2x2_nc",
    cross_section: CrossSectionSpec = "xs_nc",
) -> Component:
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
        length_y=2.0,
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


mzi_nc = partial(
    mzi,
    straight="straight_nc",
    bend="bend_euler_nc",
    splitter="mmi1x2_nc",
    combiner="mmi2x2_nc",
    cross_section="xs_nc",
)

mzi_no = partial(
    mzi,
    straight="straight_no",
    bend="bend_euler_no",
    splitter="mmi1x2_no",
    combiner="mmi2x2_no",
    cross_section="xs_no",
)


################
# Packaging
################


@gf.cell
def pad() -> Component:
    """An electrical pad."""
    return gf.c.pad(layer=LAYER.PAD, size=(100.0, 100.0))


@gf.cell
def rectangle(layer=LAYER.FLOORPLAN, **kwargs) -> gf.Component:
    """A rectangle."""
    return gf.c.rectangle(layer=layer, **kwargs)


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
    cross_section="xs_nc",
    centered=True,
    grating_coupler=None,
    port_name="o1",
    rotation=-90,
    straight_to_grating_spacing=10.0,
    with_loopback=False,
    radius: float | None = None,
) -> Component:
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
        radius: the radius of the loopback.
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
            "xs_nc": "grating_coupler_rectangular_nc",
            "xs_no": "grating_coupler_rectangular_no",
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


@gf.cell
def die(cross_section="xs_nc") -> Component:
    """A die template.

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
        "xs_nc": "grating_coupler_rectangular_nc",
        "xs_no": "grating_coupler_rectangular_no",
    }
    grating_coupler = gcs.get(xs, "grating_coupler_rectangular")
    return gf.c.die_with_pads(
        cross_section=cross_section,
        edge_to_grating_distance=150.0,
        edge_to_pad_distance=150.0,
        grating_coupler=grating_coupler,
        grating_pitch=250.0,
        layer_floorplan=LAYER.FLOORPLAN,
        ngratings=14,
        npads=31,
        pad="pad",
        pad_pitch=300.0,
        size=(11470.0, 4900.0),
    )


die_nc = partial(die, cross_section="xs_nc")
die_no = partial(die, cross_section="xs_no")


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
