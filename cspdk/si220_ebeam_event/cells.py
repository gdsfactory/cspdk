"""Cells of the EVENT-ONLY CORNERSTONE e-beam flavour (cspdk.si220_ebeam_event).

Waveguide cores are drawn on WG (3/0) only; the foundry post-processing
(:mod:`cspdk.si220_ebeam_event.mask`) etches a 5 um trench around them. Grating
teeth are drawn on GRA (6/0). Cells reject geometry below the event rules
(25 um bend radius, 200 nm gap, 200 nm feature).
"""

import gdsfactory as gf
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from cspdk._cells import _grating_coupler_rectangular
from cspdk._checks import check_min
from cspdk.si220_ebeam_event._schematic import (
    bend_circular_schematic,
    bend_euler_schematic,
    bend_s_schematic,
    coupler_ring_schematic,
    coupler_schematic,
    grating_coupler_schematic,
    ring_single_schematic,
    straight_schematic,
)
from cspdk.si220_ebeam_event.tech import LAYER, TECH


def _check_radius(radius: float | None, name: str = "radius") -> None:
    check_min(name, radius, TECH.radius_min, TECH.design_rules)


def _check_gap(gap: float) -> None:
    check_min("gap", gap, TECH.min_space, TECH.design_rules)


def _check_width(width: float | None) -> None:
    check_min("width", width, TECH.min_width, TECH.design_rules)


################
# Waveguides
################


@gf.cell(tags=["cells"], schematic_function=straight_schematic)
def straight(
    length: float = 10.0,
    width: float | None = None,
    cross_section: CrossSectionSpec = "rib",
) -> gf.Component:
    """A straight rib waveguide.

    Args:
        length: the length of the waveguide in um.
        width: the width of the waveguide (defaults to the cross-section width).
        cross_section: a cross section or its name or a function generating a cross section.
    """
    _check_width(width)
    return gf.c.straight(length=length, width=width, cross_section=cross_section)


@gf.cell(tags=["cells"], schematic_function=bend_circular_schematic)
def bend_circular(
    radius: float | None = None,
    angle: float = 90.0,
    width: float | None = None,
    cross_section: CrossSectionSpec = "rib",
) -> gf.Component:
    """A circular bend; the routing bend of the PDK.

    Args:
        radius: centre-line radius in um (defaults to the 25 um minimum).
        angle: the angle of the bend in degrees.
        width: the width of the waveguide forming the bend (defaults to the
            cross-section width).
        cross_section: a cross section or its name or a function generating a cross section.
    """
    _check_radius(radius)
    _check_width(width)
    return gf.components.bend_circular(
        radius=radius,
        angle=angle,
        width=width,
        cross_section=cross_section,
        allow_min_radius_violation=False,
    )


@gf.cell(tags=["cells"], schematic_function=bend_euler_schematic)
def bend_euler(
    radius: float = TECH.radius_euler,
    angle: float = 90.0,
    p: float = 0.5,
    width: float | None = None,
    cross_section: CrossSectionSpec = "rib",
) -> gf.Component:
    """An Euler bend whose tightest curvature respects the 25 um minimum radius.

    An Euler bend is tighter in its middle than its effective radius (about
    0.71 x radius at p=0.5), so both the effective radius and the minimum
    radius of curvature must be at least 25 um. The default 36 um radius gives
    a 25.4 um minimum at p=0.5.

    Args:
        radius: effective radius in um.
        angle: the angle of the bend in degrees.
        p: the fraction of the bend that is an Euler curve.
        width: the width of the waveguide forming the bend (defaults to the
            cross-section width).
        cross_section: a cross section or its name or a function generating a cross section.
    """
    _check_radius(radius)
    path = gf.path.euler(radius=radius, angle=angle, p=p, use_eff=True)
    _check_radius(float(path.info["Rmin"]), "minimum radius of curvature")
    _check_width(width)
    return gf.components.bend_euler(
        radius=radius,
        angle=angle,
        p=p,
        with_arc_floorplan=True,
        width=width,
        cross_section=cross_section,
        allow_min_radius_violation=False,
    )


@gf.cell(tags=["cells"], schematic_function=bend_s_schematic)
def bend_s(
    size: tuple[float, float] = (50.0, 5.0),
    cross_section: CrossSectionSpec = "rib",
    allow_min_radius_violation: bool = False,
) -> gf.Component:
    """A Bezier S-bend whose tightest curvature respects the 25 um minimum radius.

    The default 50 x 5 um S-bend has a 93 um minimum radius of curvature.

    Args:
        size: length and offset (dx, dy) of the S-bend in um.
        cross_section: a cross section or its name or a function generating a cross section.
        allow_min_radius_violation: accepted for gdsfactory compatibility
            (gdsfactory's coupler passes it); the 25 um rule is always enforced.
    """
    del allow_min_radius_violation
    c = gf.components.bend_s(
        size=size, cross_section=cross_section, allow_min_radius_violation=True
    )
    _check_radius(float(c.info["min_bend_radius"]), "minimum radius of curvature")
    return c


################
# Couplers and rings
################


@gf.cell(tags=["cells"], schematic_function=coupler_schematic)
def coupler(
    gap: float = TECH.gap,
    length: float = 20.0,
    dx: float = 16.0,
    dy: float = 3.5,
    cross_section: CrossSectionSpec = "rib",
) -> gf.Component:
    """A symmetric directional coupler.

    The S-bend size (16 x 3.5 um) is the cspdk si220 rib coupler's, which keeps
    the S-bends above the 25 um minimum radius. The coupling ratio of this
    geometry on the e-beam process is not characterised.

    Args:
        gap: gap between the coupled waveguides in um (>= 0.2).
        length: length of the coupling region in um.
        dx: length of each S-bend in um.
        dy: height of each S-bend in um.
        cross_section: a cross section or its name or a function generating a cross section.
    """
    _check_gap(gap)
    return gf.c.coupler(
        gap=gap,
        length=length,
        dx=dx,
        dy=dy,
        cross_section=cross_section,
        allow_min_radius_violation=False,
    )


@gf.cell(tags=["cells"], schematic_function=coupler_ring_schematic)
def coupler_ring(
    gap: float = TECH.gap,
    radius: float = TECH.radius_rib,
    length_x: float = 4.0,
    bend: ComponentSpec = "bend_circular",
    straight: ComponentSpec = "straight",
    cross_section: CrossSectionSpec = "rib",
    length_extension: float | None = None,
) -> gf.Component:
    """The coupling section of a ring: a bus and two quarter-circle ring bends.

    Args:
        gap: gap between the bus and the ring in um (>= 0.2).
        radius: ring bend radius in um (>= 25).
        length_x: length of the straight coupling section in um.
        bend: 90 degree bend spec.
        straight: straight spec.
        cross_section: a cross section or its name or a function generating a cross section.
        length_extension: bus extension beyond the coupling straight at each
            end in um; defaults to radius + 5 so the bus ports clear the ring.
    """
    _check_gap(gap)
    _check_radius(radius)
    if length_extension is None:
        length_extension = radius + 5.0
    return gf.c.coupler_ring(
        gap=gap,
        radius=radius,
        length_x=length_x,
        bend=bend,
        straight=straight,
        cross_section=cross_section,
        cross_section_bend=None,
        length_extension=length_extension,
    )


@gf.cell(tags=["cells"], schematic_function=ring_single_schematic)
def ring_single(
    gap: float = TECH.gap,
    radius: float = TECH.radius_rib,
    length_x: float = 4.0,
    length_y: float = 0.6,
    cross_section: CrossSectionSpec = "rib",
    length_extension: float | None = None,
) -> gf.Component:
    """An all-pass ring with circular bends.

    Args:
        gap: gap between the bus and the ring in um (>= 0.2).
        radius: ring bend radius in um (>= 25).
        length_x: length of the straight coupling section in um.
        length_y: length of the vertical straights in um.
        cross_section: a cross section or its name or a function generating a cross section.
        length_extension: bus extension beyond the coupling straight at each
            end in um; defaults to radius + 5 so the bus ports clear the ring.
    """
    _check_gap(gap)
    _check_radius(radius)
    if length_extension is None:
        length_extension = radius + 5.0
    return gf.c.ring_single(
        gap=gap,
        radius=radius,
        length_x=length_x,
        length_y=length_y,
        bend="bend_circular",
        straight="straight",
        coupler_ring="coupler_ring",
        cross_section=cross_section,
        length_extension=length_extension,
    )


################
# Grating couplers
################


@gf.cell(tags=["cells"], schematic_function=grating_coupler_schematic)
def grating_coupler(
    period: float = 0.67,
    fill_factor: float = 0.5,
    n_periods: int = 60,
    length_taper: float = 350.0,
    width_grating: float = 10.0,
    length_end: float = 4.163,
    cross_section: CrossSectionSpec = "rib",
) -> gf.Component:
    """Rectangular 1550 nm TE grating coupler with the teeth on GRA (6/0).

    Geometry of the CORNERSTONE student-batch rib grating
    (SOI220nm_1550nm_TE_RIB_Grating_Coupler, cell
    StudentSOI_1550nm_TE_rib_GC of the student MPW9 template): a 350 um taper
    from the 450 nm rib to a 10 um wide grating with 60 etched teeth of
    335 nm at a 670 nm period, starting at the end of the taper, and 4.163 um
    of unetched Si after the last tooth; 10 degree fiber angle. The student
    cell draws the teeth as gaps in 3/0; for the e-beam event they are drawn on
    6/0 over a solid 3/0 grating (same etch). Not yet characterised on the
    e-beam process.

    Args:
        period: grating period in um.
        fill_factor: etched tooth width as a fraction of the period.
        n_periods: number of teeth.
        length_taper: taper length in um.
        width_grating: grating width in um.
        length_end: unetched Si after the last tooth in um.
        cross_section: cross-section of the waveguide port.
    """
    tooth = gf.snap.snap_to_grid(period * fill_factor)
    check_min("tooth width", tooth, TECH.min_width, TECH.design_rules)
    check_min("gap between teeth", period - tooth, TECH.min_space, TECH.design_rules)
    return _grating_coupler_rectangular(
        period=period,
        n_periods=n_periods,
        fill_factor=fill_factor,
        length_taper=length_taper,
        width_grating=width_grating,
        length_grating=(n_periods - 1) * period + tooth + length_end,
        grating_offset=0.0,
        teeth_overhang=0.0,
        wavelength=1.55,
        cross_section=cross_section,
        layer_grating=LAYER.GRA,
        fiber_angle=10.0,
    )


@gf.cell(tags=["cells"])
def grating_coupler_sweep(
    periods: tuple[float, ...] = (0.65, 0.67, 0.69),
    fill_factors: tuple[float, ...] = (0.45, 0.5, 0.55),
    length: float = 100.0,
    pitch: float = TECH.io_pitch,
) -> gf.Component:
    """Test array of grating-coupler loopbacks over period and fill factor.

    One row per (period, fill factor), periods outermost, stacked at the
    250 um placeholder I/O pitch. Each row is an input grating (fiber side
    west) and an output grating (fiber side east) facing each other east-west,
    joined by a ``length`` um straight. With the default 100 um straight the
    two fiber positions are about 830 um apart, above the 500 um minimum.

    Args:
        periods: grating periods in um.
        fill_factors: etched fill factors.
        length: straight between the two gratings of a row in um.
        pitch: vertical pitch between rows in um.
    """
    c = gf.Component()
    rows = [(period, ff) for period in periods for ff in fill_factors]
    wg = straight(length=length)
    for i, (period, ff) in enumerate(rows):
        gc = grating_coupler(period=period, fill_factor=ff)
        s = c << wg
        s.movey(i * pitch)
        gc_in = c << gc
        gc_in.connect("o1", s.ports["o1"])
        gc_out = c << gc
        gc_out.connect("o1", s.ports["o2"])
    c.info["rows"] = [list(row) for row in rows]
    return c


################
# Die
################


@gf.cell(tags=["cells"])
def die(
    size: tuple[float, float] = TECH.die_size,
    n_io: int = 8,
    io_pitch: float = TECH.io_pitch,
    edge_margin: float = 100.0,
    grating_coupler: ComponentSpec = "grating_coupler",
) -> gf.Component:
    """Design area of one submission with PROVISIONAL placeholder grating I/O.

    A 6 mm x 3 mm FLOORPLAN (99/0) centred on the origin. The I/O follows
    the placeholder of the event spec, which may change (an automated I/O tool
    is planned): ``n_io`` input gratings on the west side and ``n_io`` output
    gratings on the east side, facing each other east-west, at a ``io_pitch``
    vertical pitch, with the inputs at least 500 um horizontally from the
    outputs. Each grating's outer end sits ``edge_margin`` um inside the
    floorplan edge.

    Ports are the gratings' waveguide ends: o1..o{n_io} are the inputs
    (bottom to top, facing east), o{n_io+1}..o{2 n_io} the outputs (top to
    bottom, facing west).

    Args:
        size: design area (width, height) in um.
        n_io: number of inputs (and of outputs).
        io_pitch: vertical pitch between inputs (and between outputs) in um.
        edge_margin: distance from the floorplan edge to each grating's outer end.
        grating_coupler: grating coupler spec.
    """
    check_min("io_pitch", io_pitch, TECH.io_pitch, "placeholder I/O, provisional")
    w, h = size
    c = gf.Component()
    c.add_polygon(
        [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)],
        layer=LAYER.FLOORPLAN,
    )
    gc = gf.get_component(grating_coupler)
    span = (n_io - 1) * io_pitch + gc.ysize
    if span > h - 2 * edge_margin:
        raise ValueError(
            f"{n_io} gratings at a {io_pitch} um pitch do not fit in the "
            f"{h} um design height with a {edge_margin} um margin"
        )
    fiber_offset = gc.xmax - gc.ports["o2"].x  # fiber point to the grating end
    separation = w - 2 * (edge_margin + fiber_offset)
    check_min(
        "input-output separation",
        separation,
        TECH.io_min_separation,
        "placeholder I/O, provisional",
    )
    inputs, outputs = [], []
    for i in range(n_io):
        y = (i - (n_io - 1) / 2) * io_pitch
        gc_in = c << gc
        gc_in.rotate(180)
        gc_in.move((-w / 2 + edge_margin - gc_in.xmin, y))
        inputs.append(gc_in.ports["o1"])
        gc_out = c << gc
        gc_out.move((w / 2 - edge_margin - gc_out.xmax, y))
        outputs.append(gc_out.ports["o1"])
    for i, port in enumerate(inputs + outputs[::-1], start=1):
        c.add_port(name=f"o{i}", port=port)
    c.info["io_separation"] = float(separation)
    return c
