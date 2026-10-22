"""Cells of cspdk.si220_ebeam_event. Cores are drawn on WG (3/0) only."""

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

PLACEHOLDER_IO = "placeholder I/O, provisional"


def _check_gap(gap: float) -> None:
    check_min("gap", gap, TECH.min_space, TECH.design_rules)


def _check_curvature(c: gf.Component) -> gf.Component:
    """Raise if the tightest curvature of bend ``c`` is below the minimum radius."""
    check_min(
        "minimum radius of curvature",
        float(c.info["min_bend_radius"]),
        TECH.radius_min,
        TECH.design_rules,
    )
    return c


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
        length: length in um.
        width: width in um (defaults to the cross-section's).
        cross_section: cross-section spec.
    """
    return gf.c.straight(length=length, width=width, cross_section=cross_section)


@gf.cell(tags=["cells"], schematic_function=bend_circular_schematic)
def bend_circular(
    radius: float | None = None,
    angle: float = 90.0,
    width: float | None = None,
    cross_section: CrossSectionSpec = "rib",
) -> gf.Component:
    """A circular bend; the routing bend.

    Args:
        radius: radius in um (defaults to 25).
        angle: angle in degrees.
        width: width in um (defaults to the cross-section's).
        cross_section: cross-section spec.
    """
    check_min("radius", radius, TECH.radius_min, TECH.design_rules)
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
    """An Euler bend whose tightest curvature is at least 25 um.

    Args:
        radius: effective radius in um.
        angle: angle in degrees.
        p: fraction of the bend that is an Euler curve.
        width: width in um (defaults to the cross-section's).
        cross_section: cross-section spec.
    """
    return _check_curvature(
        gf.components.bend_euler(
            radius=radius,
            angle=angle,
            p=p,
            with_arc_floorplan=True,
            width=width,
            cross_section=cross_section,
            allow_min_radius_violation=True,
        )
    )


@gf.cell(tags=["cells"], schematic_function=bend_s_schematic)
def bend_s(
    size: tuple[float, float] = (50.0, 5.0),
    cross_section: CrossSectionSpec = "rib",
    allow_min_radius_violation: bool = False,
) -> gf.Component:
    """A Bezier S-bend whose tightest curvature is at least 25 um.

    Args:
        size: length and offset (dx, dy) in um.
        cross_section: cross-section spec.
        allow_min_radius_violation: accepted because gdsfactory's coupler passes
            it; the minimum radius is always enforced.
    """
    del allow_min_radius_violation
    return _check_curvature(
        gf.components.bend_s(
            size=size, cross_section=cross_section, allow_min_radius_violation=True
        )
    )


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

    Args:
        gap: gap in um (>= 0.2).
        length: coupling length in um.
        dx: length of each S-bend in um.
        dy: port pitch in um.
        cross_section: cross-section spec.
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
    """The coupling section of a ring: a bus and two quarter-circle bends.

    Args:
        gap: gap in um (>= 0.2).
        radius: bend radius in um (>= 25).
        length_x: coupling length in um.
        bend: 90 degree bend spec.
        straight: straight spec.
        cross_section: cross-section spec.
        length_extension: bus extension at each end in um (default radius + 5).
    """
    _check_gap(gap)
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
    """An all-pass ring.

    Args:
        gap: gap in um (>= 0.2).
        radius: bend radius in um (>= 25).
        length_x: coupling length in um.
        length_y: length of the vertical straights in um.
        cross_section: cross-section spec.
        length_extension: bus extension at each end in um (default radius + 5).
    """
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
    """The CORNERSTONE student-batch rib grating, with the teeth on GRA (6/0).

    Args:
        period: period in um.
        fill_factor: etched fraction of the period.
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
    """Grating loopbacks over period and fill factor, one row per pair.

    ``info["rows"]`` lists the (period, fill factor) of each row, bottom up.

    Args:
        periods: periods in um.
        fill_factors: etched fill factors.
        length: straight between the two gratings of a row in um.
        pitch: row pitch in um.
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
    """The 6 x 3 mm design area with placeholder grating I/O.

    Inputs on the west face outputs on the east, at least 500 um apart. Ports
    o1..o{n_io} are the inputs (bottom to top); o{n_io+1}..o{2 n_io} the
    outputs (top to bottom), so o{i} and o{2 n_io + 1 - i} share a row.

    Args:
        size: design area (width, height) in um.
        n_io: number of inputs (and of outputs).
        io_pitch: vertical pitch in um.
        edge_margin: distance from the floorplan edge to each grating's end in um.
        grating_coupler: grating coupler spec.
    """
    check_min("io_pitch", io_pitch, TECH.io_pitch, PLACEHOLDER_IO)
    w, h = size
    c = gf.Component()
    c.add_polygon(
        [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)],
        layer=LAYER.FLOORPLAN,
    )
    gc = gf.get_component(grating_coupler)
    if (n_io - 1) * io_pitch + gc.ysize > h - 2 * edge_margin:
        raise ValueError(
            f"{n_io} gratings at a {io_pitch} um pitch do not fit in the "
            f"{h} um design height with a {edge_margin} um margin"
        )
    fiber_offset = gc.xmax - gc.ports["o2"].x  # fiber point to the grating end
    separation = w - 2 * (edge_margin + fiber_offset)
    check_min(
        "input-output separation", separation, TECH.io_min_separation, PLACEHOLDER_IO
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
    return c
