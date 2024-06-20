from functools import partial

import gdsfactory as gf
import numpy as np
from gdsfactory.components.bend_euler import _bend_euler
from gdsfactory.components.grating_coupler_elliptical import grating_tooth_points
from gdsfactory.typings import Component, ComponentSpec, CrossSectionSpec, LayerSpec

from cspdk.si220.config import PATH
from cspdk.si220.tech import LAYER

################
# Straights
################


@gf.cell
def straight(
    length: float = 10.0,
    npoints: int = 2,
    cross_section: CrossSectionSpec = "xs_sc",
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


straight_sc = partial(straight, cross_section="xs_sc")
straight_so = partial(straight, cross_section="xs_so")
straight_rc = partial(straight, cross_section="xs_rc")
straight_ro = partial(straight, cross_section="xs_ro")

################
# Bends
################


@gf.cell
def wire_corner(width: float = 10) -> Component:
    """Returns 45 degrees electrical corner wire."""
    return gf.c.wire_corner(cross_section="metal_routing", width=width)


@gf.cell
def bend_s(
    size: tuple[float, float] = (11.0, 1.8),
    npoints: int = 99,
    cross_section: CrossSectionSpec = "xs_sc",
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
    cross_section: CrossSectionSpec = "xs_sc",
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


bend_sc = partial(bend_euler, cross_section="xs_sc")
bend_so = partial(bend_euler, cross_section="xs_so")
bend_rc = partial(bend_euler, cross_section="xs_rc")
bend_ro = partial(bend_euler, cross_section="xs_ro")

################
# Transitions
################


@gf.cell
def taper(
    length: float = 10.0,
    width1: float = 0.5,
    width2: float | None = None,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Linear taper, which tapers only the main cross section section.

    Args:
        length: taper length.
        width1: width of the west/left port.
        width2: width of the east/right port. Defaults to width1.
    """
    return gf.c.taper(
        length=length, width1=width1, width2=width2, cross_section=cross_section
    )


taper_sc = partial(taper, cross_section="xs_so", length=10.0, width1=0.5, width2=None)
taper_so = partial(taper, cross_section="xs_so", length=10.0, width1=0.5, width2=None)
taper_rc = partial(taper, cross_section="xs_rc", length=10.0, width1=0.5, width2=None)
taper_ro = partial(taper, cross_section="xs_ro", length=10.0, width1=0.5, width2=None)


@gf.cell
def taper_strip_to_ridge(
    length: float = 10.0,
    width1: float = 0.5,
    width2: float = 0.5,
    w_slab1: float = 0.2,
    w_slab2: float = 10.45,
) -> Component:
    r"""Linear taper from strip to rib.

    Args:
        length: taper length (um).
        width1: in um.
        width2: in um.
        w_slab1: slab width in um.
        w_slab2: slab width in um.
        layer_wg: for input waveguide.
        layer_slab: for output waveguide with slab.
        cross_section: for input waveguide.
        use_slab_port: if True adds a second port for the slab.
        kwargs: cross_section settings.

    .. code::

                      __________________________
                     /           |
             _______/____________|______________
                   /             |
       width1     |w_slab1       | w_slab2  width2
             ______\_____________|______________
                    \            |
                     \__________________________

    """
    return gf.c.taper_strip_to_ridge(
        length=length,
        width1=width1,
        width2=width2,
        w_slab1=w_slab1,
        w_slab2=w_slab2,
        layer_wg="WG",
        layer_slab=LAYER.SLAB,
        cross_section="strip",
        use_slab_port=False,
    )


trans_sc_rc10 = partial(taper_strip_to_ridge, length=10)
trans_sc_rc20 = partial(taper_strip_to_ridge, length=20)
trans_sc_rc50 = partial(taper_strip_to_ridge, length=50)


################
# MMIs
################

_mmi1x2 = partial(
    gf.components.mmi1x2,
    cross_section="xs_sc",
    width_mmi=6.0,
    width_taper=1.5,
    length_taper=20.0,
)
_mmi2x2 = partial(
    gf.components.mmi2x2,
    cross_section="xs_sc",
    width_mmi=6.0,
    width_taper=1.5,
    length_taper=20.0,
)


@gf.cell
def mmi1x2_sc() -> gf.Component:
    return _mmi1x2(length_mmi=31.8, gap_mmi=1.64, cross_section="xs_sc")


@gf.cell
def mmi1x2_so() -> gf.Component:
    return _mmi1x2(length_mmi=40.1, gap_mmi=1.55, cross_section="xs_so")


@gf.cell
def mmi1x2_rc() -> gf.Component:
    return _mmi1x2(length_mmi=32.7, gap_mmi=1.64, cross_section="xs_rc")


@gf.cell
def mmi1x2_ro() -> gf.Component:
    return _mmi1x2(length_mmi=40.8, gap_mmi=1.55, cross_section="xs_ro")


@gf.cell
def mmi2x2_sc() -> gf.Component:
    return _mmi2x2(length_mmi=42.5, gap_mmi=0.5, cross_section="xs_sc")


@gf.cell
def mmi2x2_so() -> gf.Component:
    return _mmi2x2(length_mmi=53.5, gap_mmi=0.53, cross_section="xs_so")


@gf.cell
def mmi2x2_rc() -> gf.Component:
    return _mmi2x2(length_mmi=44.8, gap_mmi=0.53, cross_section="xs_rc")


@gf.cell
def mmi2x2_ro() -> gf.Component:
    return _mmi2x2(length_mmi=55.0, gap_mmi=0.53, cross_section="xs_ro")


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
    n_periods: int = 30,
    period: float = 0.315 * 2,
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
    cross_section: CrossSectionSpec = "xs_sc",
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


grating_coupler_elliptical_sc = partial(
    grating_coupler_elliptical,
    grating_line_width=0.315,
    wavelength=1.55,
    cross_section="xs_sc",
)

grating_coupler_elliptical_trenches_so = partial(
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

mzi = partial(
    gf.components.mzi,
    delta_length=10.0,
    length_y=2.0,
    length_x=0.1,
    port_e1_splitter="o2",
    port_e0_splitter="o3",
    port_e1_combiner="o3",
    port_e0_combiner="o4",
    bend="bend_sc",
    straight="straight_sc",
    splitter="mmi1x2_sc",
    combiner="mmi2x2_sc",
    cross_section="xs_sc",
)

mzi_sc = partial(
    mzi,
    straight="straight_sc",
    bend="bend_sc",
    splitter="mmi1x2_sc",
    combiner="mmi2x2_sc",
    cross_section="xs_sc",
)

mzi_so = partial(
    mzi,
    straight="straight_so",
    bend="bend_so",
    splitter="mmi1x2_so",
    combiner="mmi2x2_so",
    cross_section="xs_so",
)

mzi_rc = partial(
    mzi,
    straight="straight_rc",
    bend="bend_rc",
    splitter="mmi1x2_rc",
    combiner="mmi2x2_rc",
    cross_section="xs_rc",
)

mzi_ro = partial(
    mzi,
    straight="straight_ro",
    bend="bend_ro",
    splitter="mmi1x2_ro",
    combiner="mmi2x2_ro",
    cross_section="xs_ro",
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
    grating_coupler="grating_coupler_rectangular_sc",
    cross_section="xs_sc",
)

_die = partial(
    gf.c.die_with_pads,
    layer_floorplan=LAYER.FLOORPLAN,
    size=(11470.0, 4900.0),
    ngratings=14,
    npads=31,
    grating_pitch=250.0,
    pad_pitch=300.0,
    grating_coupler="grating_coupler_rectangular_sc",
    cross_section="xs_sc",
)


@gf.cell
def die_sc():
    return _die(grating_coupler="grating_coupler_rectangular_sc", cross_section="xs_sc")


@gf.cell
def die_so():
    return _die(grating_coupler="grating_coupler_rectangular_so", cross_section="xs_so")


@gf.cell
def die_rc():
    return _die(grating_coupler="grating_coupler_rectangular_rc", cross_section="xs_rc")


@gf.cell
def die_ro():
    return _die(grating_coupler="grating_coupler_rectangular_ro", cross_section="xs_ro")


################
# Imported from Cornerstone MPW SOI 220nm GDSII Template
################


@gf.cell
def heater() -> gf.Component:
    """Heater fixed cell."""
    heater = gf.import_gds(PATH.gds / "Heater.gds")
    heater.name = "heater"
    return heater


@gf.cell
def crossing_so() -> gf.Component:
    """SOI220nm_1310nm_TE_STRIP_Waveguide_Crossing fixed cell."""
    c = gf.import_gds(PATH.gds / "SOI220nm_1310nm_TE_STRIP_Waveguide_Crossing.gds")
    c.flatten()

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
    c.flatten()
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
    c.flatten()
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


array = gf.components.array


if __name__ == "__main__":
    c = taper_strip_to_ridge()
    print(c.function_name)
    # c.get_netlist()
    c.show()
