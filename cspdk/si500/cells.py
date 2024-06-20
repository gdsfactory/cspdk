from functools import partial

import gdsfactory as gf
from gdsfactory.components.bend_euler import _bend_euler
from gdsfactory.port import Port
from gdsfactory.typings import Component, CrossSectionSpec

from cspdk.si500.tech import LAYER

################
# Straights
################


@gf.cell
def straight(
    length: float = 10.0,
    npoints: int = 2,
    cross_section: CrossSectionSpec = "xs_rc",
    **kwargs,
) -> Component:
    """Returns a Straight waveguide.

    Args:
        length: straight length (um).
        npoints: number of points.
        cross_section: specification (CrossSection, string or dict).
        kwargs: cross_section args.

    .. code::

        o1 -------------- o2
                length
    """
    x = gf.get_cross_section(cross_section, **kwargs)
    p = gf.path.straight(length=length, npoints=npoints)
    c = p.extrude(x)
    x.add_bbox(c)

    c.info["length"] = length
    c.info["width"] = x.width if len(x.sections) == 0 else x.sections[0].width
    c.add_route_info(cross_section=x, length=length)
    return c


################
# Bends
################


@gf.cell
def wire_corner(
    cross_section: CrossSectionSpec = "metal_routing", **kwargs
) -> Component:
    """Returns 45 degrees electrical corner wire.

    Args:
        cross_section: spec.
        kwargs: cross_section parameters.
    """
    x = gf.get_cross_section(cross_section, **kwargs)
    layer = x.layer
    width = x.width

    c = Component()
    a = width / 2
    xpts = [-a, a, a, -a]
    ypts = [-a, -a, a, a]

    c.add_polygon(list(zip(xpts, ypts)), layer=layer)

    c.add_port(
        name="e1",
        center=(0, 0),
        width=width,
        orientation=180,
        layer=layer,
        port_type="electrical",
    )
    c.add_port(
        name="e2",
        center=(0, 0),
        width=width,
        orientation=90,
        layer=layer,
        port_type="electrical",
    )
    c.info["length"] = width
    c.info["dy"] = width
    x.add_bbox(c)
    return c


@gf.cell
def bend_s(
    size: tuple[float, float] = (20.0, 1.8),
    npoints: int = 99,
    cross_section: CrossSectionSpec = "xs_rc",
    allow_min_radius_violation: bool = False,
    **kwargs,
) -> Component:
    """Return S bend with bezier curve.

    stores min_bend_radius property in self.info['min_bend_radius']
    min_bend_radius depends on height and length

    Args:
        size: in x and y direction.
        npoints: number of points.
        cross_section: spec.
        allow_min_radius_violation: bool.
        kwargs: cross_section settings.

    """
    dx, dy = size

    if dy == 0:
        return gf.components.straight(length=dx, cross_section=cross_section, **kwargs)

    return gf.c.bezier(
        control_points=((0, 0), (dx / 2, 0), (dx / 2, dy), (dx, dy)),
        npoints=npoints,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
        **kwargs,
    )


@gf.cell
def bend_euler(
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.5,
    with_arc_floorplan: bool = True,
    npoints: int | None = None,
    layer: gf.typings.LayerSpec | None = None,
    width: float | None = None,
    cross_section: CrossSectionSpec = "xs_rc",
    allow_min_radius_violation: bool = False,
) -> Component:
    """Regular degree euler bend.

    Args:
        radius: in um. Defaults to cross_section_radius.
        angle: total angle of the curve.
        p: Proportion of the curve that is an Euler curve.
        with_arc_floorplan: If False: `radius` is the minimum radius of curvature.
        npoints: Number of points used per 360 degrees.
        layer: layer to use. Defaults to cross_section.layer.
        width: width to use. Defaults to cross_section.width.
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
        allow_min_radius_violation: if True allows radius to be smaller than cross_section radius.
    """
    if angle not in {90, 180}:
        gf.logger.warning(
            f"bend_euler angle should be 90 or 180. Got {angle}. Use bend_euler_all_angle instead."
        )

    return _bend_euler(
        radius=radius,
        angle=angle,
        p=p,
        with_arc_floorplan=with_arc_floorplan,
        npoints=npoints,
        layer=layer,
        width=width,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
        all_angle=False,
    )


################
# Transitions
################


@gf.cell
def taper(
    length: float = 10.0,
    width1: float = 0.5,
    width2: float | None = None,
    port: Port | None = None,
    with_two_ports: bool = True,
    cross_section: CrossSectionSpec = "strip",
    port_names: tuple[str, str] = ("o1", "o2"),
    port_types: tuple[str, str] = ("optical", "optical"),
    with_bbox: bool = True,
    **kwargs,
) -> Component:
    """Linear taper, which tapers only the main cross section section.

    Args:
        length: taper length.
        width1: width of the west/left port.
        width2: width of the east/right port. Defaults to width1.
        port: can taper from a port instead of defining width1.
        with_two_ports: includes a second port.
            False for terminator and edge coupler fiber interface.
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
        port_names: Ordered tuple of port names. First port is default \
                taper port, second name only if with_two_ports flags used.
        port_types: Ordered tuple of port types. First port is default \
                taper port, second name only if with_two_ports flags used.
        with_bbox: box in bbox_layers and bbox_offsets to avoid DRC sharp edges.
        kwargs: cross_section settings.
    """
    x1 = gf.get_cross_section(cross_section, width=width1, **kwargs)
    if width2:
        width2 = gf.snap.snap_to_grid2x(width2)
        x2 = gf.get_cross_section(cross_section, width=width2, **kwargs)
    else:
        x2 = x1

    width1 = x1.width
    width2 = x2.width
    width_max = max([width1, width2])
    x = gf.get_cross_section(cross_section, width=width_max, **kwargs)
    layer = x.layer

    if isinstance(port, gf.Port) and width1 is None:
        width1 = port.width

    width2 = width2 or width1
    c = gf.Component()
    y1 = width1 / 2
    y2 = width2 / 2

    if length:
        p1 = gf.kdb.DPolygon([(0, y1), (length, y2), (length, -y2), (0, -y1)])
        c.add_polygon(p1, layer=layer)

        s0_width = x.sections[0].width

        for section in x.sections[1:]:
            delta_width = section.width - s0_width
            p2 = p1.sized(delta_width / 2)
            c.add_polygon(p2, layer=section.layer)

    if with_bbox:
        x.add_bbox(c)
    c.add_port(
        name=port_names[0],
        center=(0, 0),
        width=width1,
        orientation=180,
        layer=x.layer,
        cross_section=x1,
        port_type=port_types[0],
    )
    if with_two_ports:
        c.add_port(
            name=port_names[1],
            center=(length, 0),
            width=width2,
            orientation=0,
            layer=x.layer,
            cross_section=x2,
            port_type=port_types[1],
        )

    x.add_bbox(c)
    c.info["length"] = length
    c.info["width1"] = float(width1)
    c.info["width2"] = float(width2)
    return c


################
# MMIs
################
@gf.cell
def mmi1x2() -> Component:
    """1x2 MultiMode Interferometer (MMI)."""
    return gf.c.mmi1x2(
        width_taper=1.5,
        length_taper=20.0,
        length_mmi=37.5,
        width_mmi=6.0,
        gap_mmi=1.47,
        taper=taper,
        straight=straight,
        cross_section="xs_rc",
    )


@gf.cell
def mmi2x2() -> Component:
    r"""Mmi 2x2.

    Args:
        width: input and output straight width.
        width_taper: interface between input straights and mmi region.
        length_taper: into the mmi region.
        length_mmi: in x direction.
        width_mmi: in y direction.
        gap_mmi: (width_taper + gap between tapered wg)/2.
        taper: taper function.
        straight: straight function.
        cross_section: spec.


    .. code::

                   length_mmi
                    <------>
                    ________
                   |        |
                __/          \__
            o2  __            __  o3
                  \          /_ _ _ _
                  |         | _ _ _ _| gap_mmi
                __/          \__
            o1  __            __  o4
                  \          /
                   |________|

                 <->
            length_taper

    """
    return gf.c.mmi2x2(
        width=None,
        width_taper=1.5,
        length_taper=50.2,
        length_mmi=5.5,
        width_mmi=6.0,
        gap_mmi=0.4,
        taper=taper,
        straight=straight,
        cross_section="xs_rc",
    )


##############################
# Evanescent couplers
##############################


@gf.cell
def coupler(
    gap: float = 0.236,
    length: float = 20.0,
    dy: float = 4.0,
    dx: float = 15.0,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    r"""Symmetric coupler.

    Args:
        gap: between straights in um.
        length: of coupling region in um.
        dy: port to port vertical spacing in um.
        dx: length of bend in x direction in um.
        cross_section: spec (CrossSection, string or dict).

    .. code::

               dx                                 dx
            |------|                           |------|
         o2 ________                           ______o3
                    \                         /           |
                     \        length         /            |
                      ======================= gap         | dy
                     /                       \            |
            ________/                         \_______    |
         o1                                          o4

                        coupler_straight  coupler_symmetric


    """
    # length = gf.snap.snap_to_grid(length)
    # gap = gf.snap.snap_to_grid2x(gap)
    c = Component()

    sbend = gf.c.coupler_symmetric(gap=gap, dy=dy, dx=dx, cross_section=cross_section)

    sr = c << sbend
    sl = c << sbend
    cs = c << gf.c.coupler_straight(length=length, gap=gap, cross_section=cross_section)
    sl.connect("o2", other=cs.ports["o1"])
    sr.connect("o1", other=cs.ports["o4"])

    c.add_port("o1", port=sl.ports["o3"])
    c.add_port("o2", port=sl.ports["o4"])
    c.add_port("o3", port=sr.ports["o3"])
    c.add_port("o4", port=sr.ports["o4"])

    c.info["length"] = sbend.info["length"]
    c.info["min_bend_radius"] = sbend.info["min_bend_radius"]
    c.auto_rename_ports()

    x = gf.get_cross_section(cross_section)
    x.add_bbox(c)
    c.flatten()
    return c


##############################
# grating couplers Rectangular
##############################
@gf.cell
def grating_coupler_rectangular() -> Component:
    """Grating coupler with rectangular shapes (not elliptical) for 10 deg angle and TE."""
    return gf.c.grating_coupler_rectangular(
        n_periods=60,
        period=0.57,
        fill_factor=0.5,
        width_grating=11.0,
        length_taper=350.0,
        polarization="te",
        wavelength=1.55,
        taper=taper,
        layer_slab=LAYER.WG,
        layer_grating=LAYER.GRA,
        fiber_angle=10,
        slab_xmin=-1.0,
        slab_offset=0.0,
        cross_section="xs_rc",
    )


##############################
# grating couplers elliptical
##############################
@gf.cell
def grating_coupler_elliptical() -> Component:
    """Returns Grating coupler with defined trenches."""
    return gf.c.grating_coupler_elliptical_trenches(
        polarization="te",
        taper_length=16.6,
        taper_angle=30.0,
        trenches_extra_angle=9.0,
        wavelength=1.53,
        fiber_angle=15.0,
        grating_line_width=0.315,
        neff=2.638,
        ncladding=1.443,
        layer_trench=LAYER.GRA,
        p_start=26,
        n_periods=30,
        end_straight_length=0.2,
        cross_section="strip",
    )


################
# MZI
################

_mzi = partial(
    gf.components.mzi,
    delta_length=10.0,
    length_y=2.0,
    length_x=0.1,
    port_e1_splitter="o2",
    port_e0_splitter="o3",
    port_e1_combiner="o3",
    port_e0_combiner="o4",
    bend="bend_euler",
    straight="straight",
    splitter="mmi1x2",
    combiner="mmi2x2",
    cross_section="xs_rc",
)


@gf.cell
def mzi_rc(
    delta_length: float = 10.0,
    length_y: float = 2.0,
    length_x: float = 0.1,
) -> gf.Component:
    return _mzi(
        bend="bend_euler",
        straight="straight",
        splitter="mmi1x2",
        combiner="mmi2x2",
        cross_section="xs_rc",
        delta_length=delta_length,
        length_y=length_y,
        length_x=length_x,
    )


################
# Packaging
################
@gf.cell
def pad(size=(100.0, 100.0)) -> gf.Component:
    return gf.c.pad(size=size, layer="PAD")


rectangle = partial(gf.components.rectangle, layer=LAYER.FLOORPLAN)

_grating_coupler_array = partial(
    gf.components.grating_coupler_array,
    port_name="o1",
    rotation=-90,
    with_loopback=False,
    centered=True,
)


@gf.cell
def grating_coupler_array(
    pitch=127,
    n=6,
    grating_coupler="grating_coupler_rectangular",
    cross_section="xs_rc",
    **kwargs,
) -> gf.Component:
    return _grating_coupler_array(
        pitch=pitch,
        n=n,
        grating_coupler=grating_coupler,
        cross_section=cross_section,
        **kwargs,
    )


_die = partial(
    gf.c.die_with_pads,
    layer_floorplan=LAYER.FLOORPLAN,
    size=(11470.0, 4900.0),
    ngratings=14,
    npads=31,
    grating_pitch=250.0,
    grating_coupler="grating_coupler_rectangular",
    pad_pitch=300.0,
    cross_section="xs_rc",
)


@gf.cell
def die() -> gf.Component:
    return _die(grating_coupler="grating_coupler_rectangular", cross_section="xs_rc")


array = partial(gf.components.array)

if __name__ == "__main__":
    c = grating_coupler_array()
    c.show()
