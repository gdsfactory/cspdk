from functools import partial

import gdsfactory as gf
import numpy as np
from gdsfactory.components.bend_euler import _bend_euler
from gdsfactory.components.grating_coupler_elliptical import grating_tooth_points
from gdsfactory.port import Port
from gdsfactory.typings import Component, ComponentSpec, CrossSectionSpec, LayerSpec

from cspdk.base_cell import base_cell
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
    size: tuple[float, float] = (11.0, 1.8),
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
def mmi1x2(
    width: float | None = None,
    width_taper: float = 1.5,
    length_taper: float = 20.0,
    length_mmi: float = 37.5,
    width_mmi: float = 6.0,
    gap_mmi: float = 1.47,
    taper: ComponentSpec = taper,
    straight: ComponentSpec = straight,
    cross_section: CrossSectionSpec = "xs_rc",
) -> Component:
    r"""1x2 MultiMode Interferometer (MMI).

    Args:
        width: input and output straight width. Defaults to cross_section width.
        width_taper: interface between input straights and mmi region.
        length_taper: into the mmi region.
        length_mmi: in x direction.
        width_mmi: in y direction.
        gap_mmi:  gap between tapered wg.
        taper: taper function.
        straight: straight function.
        cross_section: specification (CrossSection, string or dict).


    .. code::

               length_mmi
                <------>
                ________
               |        |
               |         \__
               |          __  o2
            __/          /_ _ _ _
         o1 __          | _ _ _ _| gap_mmi
              \          \__
               |          __  o3
               |         /
               |________|

             <->
        length_taper

    """
    c = Component()
    gap_mmi = gf.snap.snap_to_grid(gap_mmi, grid_factor=2)
    x = gf.get_cross_section(cross_section)
    xs_mmi = gf.get_cross_section(cross_section, width=width_mmi)
    width = width or x.width

    _taper = gf.get_component(
        taper,
        length=length_taper,
        width1=width,
        width2=width_taper,
        cross_section=cross_section,
    )

    a = gap_mmi / 2 + width_taper / 2
    _ = c << gf.get_component(straight, length=length_mmi, cross_section=xs_mmi)

    ports = [
        gf.Port(
            "o1",
            orientation=180,
            center=(0, 0),
            width=width_taper,
            layer=x.layer,
            cross_section=x,
        ),
        gf.Port(
            "o2",
            orientation=0,
            center=(+length_mmi, +a),
            width=width_taper,
            layer=x.layer,
            cross_section=x,
        ),
        gf.Port(
            "o3",
            orientation=0,
            center=(+length_mmi, -a),
            width=width_taper,
            layer=x.layer,
            cross_section=x,
        ),
    ]

    for port in ports:
        taper_ref = c << _taper
        taper_ref.connect(port="o2", other=port, allow_width_mismatch=True)
        c.add_port(name=port.name, port=taper_ref.ports["o1"])

    c.flatten()
    return c


@gf.cell
def mmi2x2(
    width: float | None = None,
    width_taper: float = 1.5,
    length_taper: float = 50.2,
    length_mmi: float = 5.5,
    width_mmi: float = 6.0,
    gap_mmi: float = 0.4,
    taper: ComponentSpec = taper,
    straight: ComponentSpec = straight,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
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
    c = gf.Component()
    gap_mmi = gf.snap.snap_to_grid(gap_mmi, grid_factor=2)
    w_taper = width_taper
    x = gf.get_cross_section(cross_section)
    width = width or x.width

    _taper = gf.get_component(
        taper,
        length=length_taper,
        width1=width,
        width2=w_taper,
        cross_section=cross_section,
    )

    a = gap_mmi / 2 + width_taper / 2
    _ = c << gf.get_component(
        straight, length=length_mmi, width=width_mmi, cross_section=cross_section
    )

    ports = [
        gf.Port("o1", orientation=180, center=(0, -a), width=w_taper, cross_section=x),
        gf.Port("o2", orientation=180, center=(0, +a), width=w_taper, cross_section=x),
        gf.Port(
            "o3",
            orientation=0,
            center=(length_mmi, +a),
            width=w_taper,
            cross_section=x,
        ),
        gf.Port(
            "o4",
            orientation=0,
            center=(length_mmi, -a),
            width=w_taper,
            cross_section=x,
        ),
    ]

    for port in ports:
        taper_ref = c << _taper
        taper_ref.connect(port="o2", other=port, allow_width_mismatch=True)
        c.add_port(name=port.name, port=taper_ref.ports["o1"])

    c.flatten()
    return c


##############################
# Evanescent couplers
##############################


@gf.cell
def coupler(
    gap: float = 0.236,
    length: float = 20.0,
    dy: float = 4.0,
    dx: float = 10.0,
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
def grating_coupler_rectangular(
    n_periods: int = 60,
    period: float = 0.57,
    fill_factor: float = 0.5,
    width_grating: float = 11.0,
    length_taper: float = 350.0,
    polarization: str = "te",
    wavelength: float = 1.55,
    taper: ComponentSpec = taper,
    layer_slab: LayerSpec | None = LAYER.WG,
    layer_grating: LayerSpec | None = LAYER.GRA,
    fiber_angle: float = 10,
    slab_xmin: float = -1.0,
    slab_offset: float = 0.0,
    cross_section: CrossSectionSpec = "xs_rc",
    **kwargs,
) -> Component:
    r"""Grating coupler with rectangular shapes (not elliptical).

    Needs longer taper than elliptical.
    Grating teeth are straight.
    For a focusing grating take a look at grating_coupler_elliptical.

    Args:
        n_periods: number of grating teeth.
        period: grating pitch.
        fill_factor: ratio of grating width vs gap.
        width_grating: 11.
        length_taper: 150.
        polarization: 'te' or 'tm'.
        wavelength: in um.
        taper: function.
        layer_slab: layer that protects the slab under the grating.
        layer_grating: layer for the grating.
        fiber_angle: in degrees.
        slab_xmin: where 0 is at the start of the taper.
        slab_offset: from edge of grating to edge of the slab.
        cross_section: for input waveguide port.
        kwargs: cross_section settings.

    .. code::

        side view
                      fiber

                   /  /  /  /
                  /  /  /  /

                _|-|_|-|_|-|___ layer
                   layer_slab |
            o1  ______________|


        top view     _________
                    /| | | | |
                   / | | | | |
                  /taper_angle
                 /_ _| | | | |
        wg_width |   | | | | |
                 \   | | | | |
                  \  | | | | |
                   \ | | | | |
                    \|_|_|_|_|
                 <-->
                taper_length
    """
    xs = gf.get_cross_section(cross_section, **kwargs)
    wg_width = xs.width
    layer = layer_grating or xs.layer

    c = Component()
    taper_ref = c << gf.get_component(
        taper,
        length=length_taper,
        width2=width_grating,
        width1=wg_width,
        cross_section=cross_section,
    )

    c.add_port(port=taper_ref.ports["o1"], name="o1")
    x0 = taper_ref.dxmax

    for i in range(n_periods):
        xsize = gf.snap.snap_to_grid(period * fill_factor)
        cgrating = c.add_ref(
            rectangle(size=(xsize, width_grating), layer=layer, port_type=None)
        )
        cgrating.dxmin = gf.snap.snap_to_grid(x0 + i * period)
        cgrating.dy = 0

    c.info["polarization"] = polarization
    c.info["wavelength"] = wavelength
    c.info["fiber_angle"] = fiber_angle

    if layer_slab:
        slab_xmin += length_taper
        slab_xsize = cgrating.dxmax + slab_offset
        slab_ysize = c.dysize + 2 * slab_offset
        yslab = slab_ysize / 2
        c.add_polygon(
            [
                (slab_xmin, yslab),
                (slab_xsize, yslab),
                (slab_xsize, -yslab),
                (slab_xmin, -yslab),
            ],
            layer_slab,
        )
    xs.add_bbox(c)
    xport = np.round((x0 + cgrating.dx) / 2, 3)
    c.add_port(
        name="o2",
        port_type=f"vertical_{polarization}",
        center=(xport, 0),
        orientation=0,
        width=width_grating,
        layer=layer,
    )
    c.flatten()
    return c


##############################
# grating couplers elliptical
##############################


@gf.cell
def grating_coupler_elliptical(
    polarization: str = "te",
    taper_length: float = 16.6,
    taper_angle: float = 30.0,
    trenches_extra_angle: float = 9.0,
    wavelength: float = 1.53,
    fiber_angle: float = 15.0,
    grating_line_width: float = 0.315,
    neff: float = 2.638,  # tooth effective index
    ncladding: float = 1.443,  # cladding index
    layer_trench: LayerSpec | None = LAYER.GRA,
    p_start: int = 26,
    n_periods: int = 30,
    end_straight_length: float = 0.2,
    cross_section: CrossSectionSpec = "strip",
    **kwargs,
) -> Component:
    r"""Returns Grating coupler with defined trenches.

    Some foundries define the grating coupler by a shallow etch step (trenches)
    Others define the slab that they keep (see grating_coupler_elliptical)

    Args:
        polarization: 'te' or 'tm'.
        taper_length: taper length from straight I/O.
        taper_angle: grating flare angle.
        trenches_extra_angle: extra angle for the trenches.
        wavelength: grating transmission central wavelength.
        fiber_angle: fibre polish angle in degrees.
        grating_line_width: of the 220 ridge.
        neff: tooth effective index.
        ncladding: cladding index.
        layer_trench: for the trench.
        p_start: first tooth.
        n_periods: number of grating teeth.
        end_straight_length: at the end of straight.
        cross_section: cross_section spec.
        kwargs: cross_section settings.


    .. code::

                      fiber

                   /  /  /  /
                  /  /  /  /
                _|-|_|-|_|-|___
        WG  o1  ______________|

    """
    xs = gf.get_cross_section(cross_section, **kwargs)
    wg_width = xs.width
    layer = xs.layer

    # Compute some ellipse parameters
    sthc = np.sin(fiber_angle * np.pi / 180)
    d = neff**2 - ncladding**2 * sthc**2
    a1 = wavelength * neff / d
    b1 = wavelength / np.sqrt(d)
    x1 = wavelength * ncladding * sthc / d

    a1 = round(a1, 3)
    b1 = round(b1, 3)
    x1 = round(x1, 3)

    period = float(a1 + x1)
    trench_line_width = period - grating_line_width

    c = gf.Component()

    # Make each grating line
    for p in range(p_start, p_start + n_periods + 1):
        pts = grating_tooth_points(
            p * a1,
            p * b1,
            p * x1,
            width=trench_line_width,
            taper_angle=taper_angle + trenches_extra_angle,
        )
        c.add_polygon(pts, layer_trench)

    # Make the taper
    p_taper = p_start - 1
    p_taper_eff = p_taper
    a_taper = a1 * p_taper_eff
    # b_taper = b1 * p_taper_eff
    x_taper = x1 * p_taper_eff
    x_output = a_taper + x_taper - taper_length + grating_line_width / 2

    xmax = x_output + taper_length + n_periods * period + 3
    y = wg_width / 2 + np.tan(taper_angle / 2 * np.pi / 180) * xmax
    pts = [
        (x_output, -wg_width / 2),
        (x_output, +wg_width / 2),
        (xmax, +y),
        (xmax + end_straight_length, +y),
        (xmax + end_straight_length, -y),
        (xmax, -y),
    ]
    c.add_polygon(pts, layer)

    c.add_port(
        name="o1",
        center=(x_output, 0),
        width=wg_width,
        orientation=180,
        layer=layer,
        cross_section=xs,
    )
    c.info["period"] = float(np.round(period, 3))
    c.info["polarization"] = polarization
    c.info["wavelength"] = wavelength
    xs.add_bbox(c)

    x = np.round(taper_length + period * n_periods / 2, 3)
    c.add_port(
        name="o2",
        center=(x, 0),
        width=10,
        orientation=0,
        layer=layer,
        port_type=f"vertical_{polarization}",
    )
    return c


################
# MZI
################

# TODO: (needs gdsfactory fix) currently function arguments need to be
# supplied as ComponentSpec strings, because when supplied as function they get
# serialized weirdly in the netlist

mzi = base_cell(
    "mzi",
    partial(
        gf.components.mzi,
        delta_length=10.0,
        length_y=2.0,
        length_x=0.1,
        port_e1_splitter="o2",
        port_e0_splitter="o3",
        port_e1_combiner="o3",
        port_e0_combiner="o4",
        bend="bend_rc",
        straight="straight_rc",
        splitter="mmi1x2_rc",
        combiner="mmi2x2_rc",
        cross_section="xs_rc",
    ),
)

mzi_rc = partial(
    mzi,
    bend="bend_rc",
    straight="straight_rc",
    splitter="mmi1x2_rc",
    combiner="mmi2x2_rc",
    cross_section="xs_rc",
)


################
# Packaging
################

pad = partial(gf.c.pad, layer="PAD", size=(100.0, 100.0))
rectangle = partial(gf.components.rectangle, layer=LAYER.FLOORPLAN)
grating_coupler_array = partial(
    gf.components.grating_coupler_array,
    pitch=127,
    n=6,
    port_name="o1",
    rotation=-90,
    with_loopback=False,
    grating_coupler="grating_coupler_rectangular_rc",
    cross_section="xs_rc",
)


die = partial(
    gf.c.die_with_pads,
    layer_floorplan=LAYER.FLOORPLAN,
    size=(11470.0, 4900.0),
    ngratings=14,
    npads=31,
    grating_pitch=250.0,
    grating_coupler="grating_coupler_rectangular_rc",
    pad_pitch=300.0,
    cross_section="xs_rc",
)
die_rc = partial(
    die,
    grating_coupler="grating_coupler_rectangular_rc",
    cross_section="xs_rc",
)
array = partial(gf.components.array)

if __name__ == "__main__":
    c = grating_coupler_elliptical()
    c.show()
