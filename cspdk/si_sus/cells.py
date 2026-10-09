"""Building blocks for the cspdk.si_sus library.

Layer (404, 0) is a dark-field etch: the suspended core is the un-drawn Si
between two 3.5um etch windows, each drawn as 0.3um slots at 0.55um pitch
(0.25um Si tethers). Straights get their slots from the xs_sus along-path
slot pairs; bends, S-bends and tapers redraw them here so they follow the
curve like the foundry library GDS, keep the MPW #7 'CORNERSTONE to bias'
rules (>= 270nm features, >= 180nm gaps on 404) and stay inside the cell, so
abutting cells keep >= 0.25um tethers at their junctions.
"""

from functools import partial

import gdsfactory as gf
import numpy as np
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from cspdk.si_sus._schematic import (
    bend_circular_schematic,
    bend_euler_schematic,
    bend_s_schematic,
    grating_coupler_rectangular_schematic,
    straight_schematic,
    taper_schematic,
)
from cspdk.si_sus.config import PATH
from cspdk.si_sus.tech import LAYER, TECH, Tech

_SAMPLE_STEP = 0.01  # um, center-line sampling used to place slot vertices
_GRID_MARGIN = 0.002  # um, keeps DRC minima after snapping to the 1nm grid
# um, minimum distance from a slot edge to the cell end (0.125um): half the
# 0.25um tether, so abutting cells keep full tethers at their junction
_SLOT_INSET = TECH.etch_slot_padding - TECH.etch_slot_length / 2


################
# Etch slots
################


def _slot_centers(length: float, slot: float, pitch: float) -> np.ndarray:
    """Return slot centers spread symmetrically along `length`.

    Same rule as gdsfactory's along_path with the xs_sus padding: every slot
    edge is at least _SLOT_INSET from either end.
    """
    padding = slot / 2 + _SLOT_INSET
    if length < 2 * padding:
        return np.zeros(0)
    n = int(np.floor((length - 2 * padding) / pitch + 1e-9)) + 1
    return (length - (n - 1) * pitch) / 2 + pitch * np.arange(n)


def _curved_slots(points: np.ndarray, width: float) -> list[gf.kdb.DPolygon]:
    """Return etch slots perpendicular to a one-way bend (circular or euler).

    Each slot is bounded by two normals of the center line, so in a circular
    bend the slots are polar wedges pointing at the bend center, like the
    Suspendedsilicon500nm_3800nm_TE_90_DegreeBend GDS. Slot length and pitch
    are measured along the inner core edge (0.3um and 0.55um, the foundry
    convention: 0.0075 rad and 0.01375 rad at its 40um inner edge). Where the
    curvature would shrink the slots or tethers at the inner window's inner
    edge below the 404 minima, both are widened for the whole bend.

    Args:
        points: center-line points, sampled finely.
        width: core width.
    """
    pts = np.asarray(points, dtype=float)
    pts = pts[np.r_[True, np.hypot(*np.diff(pts, axis=0).T) > 1e-9]]
    tangent = np.gradient(pts, axis=0)
    theta = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    side = 1.0 if theta[-1] >= theta[0] else -1.0  # inner side of the bend
    normal = np.c_[-np.sin(theta), np.cos(theta)]
    edge = pts + side * width / 2 * normal
    s_edge = np.r_[0, np.cumsum(np.hypot(*np.diff(edge, axis=0).T))]

    # inner window's inner edge vs inner core edge length ratio
    w1, w2 = width / 2, width / 2 + TECH.width_etch_window
    kappa = np.abs(np.gradient(theta, s))
    if np.any(kappa * w2 >= 1):
        raise ValueError(
            f"Bend radius {1 / kappa.max():.3f}um is too small for the "
            f"{w2:.2f}um half-width of the slotted cross-section."
        )
    scale = float(np.min((1 - kappa * w2) / (1 - kappa * w1)))
    slot = max(TECH.etch_slot_length, (TECH.min_feature_404 + _GRID_MARGIN) / scale)
    pitch = max(TECH.tether_period, slot + (TECH.min_gap_404 + _GRID_MARGIN) / scale)

    def frame(at: float) -> tuple[np.ndarray, np.ndarray]:
        x = np.interp(at, s_edge, pts[:, 0])
        y = np.interp(at, s_edge, pts[:, 1])
        t = np.interp(at, s_edge, theta)
        return np.array([x, y]), np.array([-np.sin(t), np.cos(t)])

    polygons = []
    for center in _slot_centers(s_edge[-1], slot, pitch):
        (p1, n1), (p2, n2) = frame(center - slot / 2), frame(center + slot / 2)
        for d1, d2 in ((w1, w2), (-w1, -w2)):
            quad = [p1 + d1 * n1, p1 + d2 * n1, p2 + d2 * n2, p2 + d1 * n2]
            polygons.append(gf.kdb.DPolygon([gf.kdb.DPoint(*q) for q in quad]))
    return polygons


def _vertical_slots(
    length: float, upper: np.ndarray, lower: np.ndarray, xs: np.ndarray
) -> list[gf.kdb.DPolygon]:
    """Return vertical etch slots hanging off the core edges.

    Slots are 0.3um wide in x at 0.55um x-pitch; their inner ends follow the
    core edges and they extend 3.5um vertically, like the slots of the
    foundry S-bend and grating-coupler taper GDS.

    Args:
        length: x extent of the cell, starting at x=0.
        upper: y of the upper core edge at `xs`.
        lower: y of the lower core edge at `xs`.
        xs: increasing x samples for `upper` and `lower`.
    """
    window = TECH.width_etch_window
    half = TECH.etch_slot_length / 2
    polygons = []
    for center in _slot_centers(length, TECH.etch_slot_length, TECH.tether_period):
        xa, xb = center - half, center + half
        for edge, sign in ((upper, 1), (lower, -1)):
            ya, yb = np.interp(xa, xs, edge), np.interp(xb, xs, edge)
            quad = [
                (xa, ya),
                (xa, ya + sign * window),
                (xb, yb + sign * window),
                (xb, yb),
            ]
            polygons.append(gf.kdb.DPolygon([gf.kdb.DPoint(*q) for q in quad]))
    return polygons


def _replace_slots(
    component: gf.Component, slots: list[gf.kdb.DPolygon]
) -> gf.Component:
    """Return a flat copy of `component` with its (404, 0) shapes replaced."""
    c = gf.Component()
    c.add_ref(component)
    c.flatten()
    c.remove_layers([LAYER.WG])
    for slot in slots:
        c.add_polygon(slot, layer=LAYER.WG)
    c.add_ports(component.ports)
    c.info.update(component.info.model_dump())
    return c


def _get_cross_section(
    cross_section: CrossSectionSpec, width: float | None
) -> gf.CrossSection:
    """Return the cross-section, with its width overridden when given."""
    return gf.get_cross_section(cross_section, **({"width": width} if width else {}))


################
# Waveguides
################


@gf.cell(tags=["cells"], schematic_function=straight_schematic)
def straight(
    length: float = 10.0,
    cross_section: CrossSectionSpec = "xs_sus",
    **kwargs,
) -> gf.Component:
    """A straight waveguide.

    The slots are the xs_sus slot pairs, drawn flat at exact positions:
    gdsfactory's along_path adds the pitch up in floating point and can drop
    the last slot (e.g. at length 500). straight(500) reproduces the 909 slot
    pairs of the Suspendedsilicon500nm_3800nm_TE_Waveguide GDS, centered in
    the cell.

    Args:
        length: the length of the waveguide.
        cross_section: a cross section or its name or a function generating a cross section.
        kwargs: additional arguments to pass to the straight function.
    """
    base = gf.c.straight(length=length, cross_section=cross_section, **kwargs)
    edge = np.full(2, base.info["width"] / 2)
    xs = np.array([0.0, length])
    return _replace_slots(base, _vertical_slots(length, edge, -edge, xs))


@gf.cell(tags=["cells"], schematic_function=bend_s_schematic)
def bend_s(
    size: tuple[float, float] = (40.0, 8.0),
    cross_section: CrossSectionSpec = "xs_sus",
    allow_min_radius_violation: bool = True,
    width: float | None = None,
) -> gf.Component:
    """A raised-cosine S-bend with vertical etch slots.

    The default size and the slot drawing follow the foundry
    Suspendedsilicon500nm_3800nm_TE_SBend GDS: a cosine center line
    y = dy (1 - cos(pi x / dx)) / 2 and vertical 0.3um slots at 0.55um
    x-pitch whose inner ends follow the core edges (the foundry cell matches
    this to within its ~10nm drawing tolerance).

    Args:
        size: the length and height of the s-bend.
        cross_section: a cross section or its name or a function generating a cross section.
        allow_min_radius_violation: if True, allows the s-bend to have a smaller radius than the minimum radius.
        width: waveguide width; defaults to the cross-section width.
    """
    x = _get_cross_section(cross_section, width)
    dx, dy = size

    def center_line(npoints: int) -> tuple[np.ndarray, np.ndarray]:
        xs = np.linspace(0, dx, npoints)
        return xs, dy / 2 * (1 - np.cos(np.pi * xs / dx))

    path = gf.Path(np.column_stack(center_line(201)))
    path.start_angle = path.end_angle = 0
    # extrude the core only; the slots are drawn below
    c = gf.path.extrude(path, x.model_copy(update={"components_along_path": ()}))

    # core edges: offset the center line along its normal, then sample at x
    xs, ys = center_line(max(int(np.ceil(dx / _SAMPLE_STEP)), 2) + 1)
    theta = np.arctan(dy / 2 * np.pi / dx * np.sin(np.pi * xs / dx))
    w = x.width / 2
    upper = np.interp(xs, xs - w * np.sin(theta), ys + w * np.cos(theta))
    lower = np.interp(xs, xs + w * np.sin(theta), ys - w * np.cos(theta))
    for slot in _vertical_slots(dx, upper, lower, xs):
        c.add_polygon(slot, layer=LAYER.WG)

    kappa = np.pi**2 * abs(dy) / (2 * dx**2)
    min_bend_radius = float(gf.snap.snap_to_grid(1 / kappa)) if dy else np.inf
    c.info["length"] = float(np.round(path.length(), 3))
    c.info["min_bend_radius"] = min_bend_radius
    c.info["start_angle"] = 0.0
    c.info["end_angle"] = 0.0
    c.add_route_info(
        cross_section=x,
        length=c.info["length"],
        n_bend_s=1,
        min_bend_radius=min_bend_radius,
    )
    if not allow_min_radius_violation:
        x.validate_radius(min_bend_radius)
    return c


@gf.cell(tags=["cells"], schematic_function=bend_euler_schematic)
def bend_euler(
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.5,
    width: float | None = None,
    cross_section: CrossSectionSpec = "xs_sus",
) -> gf.Component:
    """An euler bend (not a foundry library cell; routing uses bend_circular).

    Etch slots are drawn perpendicular to the curve with the pitch measured
    along the inner core edge, widened where the tightest curvature would
    break the layer-404 minima. The minimum local radius must respect the
    cross-section's radius_min.

    Args:
        radius: the effective radius of the bend.
        angle: the angle of the bend (usually 90 degrees).
        p: the fraction of the bend that's represented by a polar bend.
        width: the width of the waveguide forming the bend.
        cross_section: a cross section or its name or a function generating a cross section.
    """
    base = gf.components.bend_euler(
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
    x = _get_cross_section(cross_section, width)
    x.validate_radius(base.info["min_bend_radius"])
    radius = radius or x.radius
    npoints = int(np.ceil(base.info["length"] / _SAMPLE_STEP)) + 1
    path = gf.path.euler(radius=radius, angle=angle, p=p, use_eff=True, npoints=npoints)
    return _replace_slots(base, _curved_slots(path.points, x.width))


@gf.cell(tags=["cells"], schematic_function=bend_circular_schematic)
def bend_circular(
    radius: float | None = None,
    angle: float = 90.0,
    width: float | None = None,
    cross_section: CrossSectionSpec = "xs_sus",
) -> gf.Component:
    """A circular bend with polar-wedge etch slots, like the foundry bend.

    The Suspendedsilicon500nm_3800nm_TE_90_DegreeBend GDS is a 40.75um
    center-line radius arc (the xs_sus default) whose slots are polar wedges
    of 0.0075 rad at 0.01375 rad pitch (0.3um at 0.55um on the 40um inner
    core edge); this cell draws the same wedges. At the default radius it has
    114 slot pairs per 90 degrees: the foundry cell fits a 115th by letting
    its last wedge overhang the port plane by 0.24 degrees, which would
    leave < 180nm to the slots of an abutting cell.

    Args:
        radius: the radius of the bend (defaults to the cross-section radius).
        angle: the angle of the bend (usually 90 degrees).
        width: the width of the waveguide forming the bend.
        cross_section: a cross section or its name or a function generating a cross section.
    """
    base = gf.components.bend_circular(
        radius=radius,
        angle=angle,
        width=width,
        cross_section=cross_section,
        allow_min_radius_violation=False,
    )
    radius = base.info["radius"]
    npoints = int(np.ceil(abs(np.radians(angle)) * radius / _SAMPLE_STEP)) + 1
    path = gf.path.arc(radius=radius, angle=angle, npoints=npoints)
    return _replace_slots(base, _curved_slots(path.points, base.info["width"]))


@gf.cell(tags=["cells"], schematic_function=taper_schematic)
def taper(
    length: float = 10.0,
    width1: float = Tech.width_sus,
    width2: float | None = None,
    port: gf.Port | None = None,
    cross_section: CrossSectionSpec = "xs_sus",
) -> gf.Component:
    """A linear taper with vertical etch slots along both core edges.

    The slots follow the tapered core edges like those of the foundry
    grating-coupler taper. Suspended cores wider than 16um are rejected
    (MPW #7 maximum suspended waveguide width).

    Args:
        length: the length of the taper.
        width1: the input width of the taper.
        width2: the output width of the taper (if not given, use port).
        port: the port (with certain width) to taper towards (if not given, use width2).
        cross_section: a cross section or its name or a function generating a cross section.
    """
    base = gf.c.taper(
        length=length,
        width1=width1,
        width2=width2,
        port=port,
        cross_section=cross_section,
    )
    w1, w2 = base.info["width1"], base.info["width2"]
    if max(w1, w2) > TECH.max_suspended_width:
        raise ValueError(
            f"taper width {max(w1, w2)}um exceeds the "
            f"{TECH.max_suspended_width}um maximum suspended waveguide width."
        )
    xs = np.array([0.0, length])
    upper = np.array([w1, w2]) / 2
    return _replace_slots(base, _vertical_slots(length, upper, -upper, xs))


################
# Grating couplers
################

_GC_TAPER = (4.17, 304.17)  # x span of the 1.5um -> 15um taper in the GDS
_GC_END = 355.05
_GC_CENTER = 331.169  # center of the 13 x 20 hole array


@gf.cell(tags=["cells"], schematic_function=grating_coupler_rectangular_schematic)
def grating_coupler_rectangular(
    cross_section: CrossSectionSpec = "xs_sus",
) -> gf.Component:
    """The foundry suspended-Si 3.8um TE grating coupler.

    Imports Suspendedsilicon500nm_3800nm_TE_GratingCoupler.gds unchanged: a
    slotted 300um taper from the 1.5um core to a 15um wide region with a
    13 x 20 array of 1.15um x 0.54um holes (1.1um x 2.3um periods) for a 19
    degree fiber. The GDS waveguide end is at x=0; o1 sits 0.125um further
    out (on un-etched Si) so that, like every other cell, the first slot is
    0.125um inside the port and abutting cells keep >= 0.25um tethers. o2 is
    the fiber port at the center of the hole array.

    Args:
        cross_section: cross-section of the waveguide port.
    """
    x = gf.get_cross_section(cross_section)
    c = gf.import_gds(PATH.gds / "Suspendedsilicon500nm_3800nm_TE_GratingCoupler.gds")
    half1, half2 = Tech.width_sus / 2, 7.5
    x0, x1 = _GC_TAPER
    core = [(-_SLOT_INSET, half1), (x0, half1), (x1, half2), (_GC_END, half2)]
    core += [(px, -py) for px, py in reversed(core)]
    c.add_polygon(core, layer=LAYER.WG_MARK)
    c.add_port(
        name="o1",
        center=(-_SLOT_INSET, 0),
        width=Tech.width_sus,
        orientation=180,
        layer=LAYER.WG_MARK,
        cross_section=x,
    )
    c.add_port(
        name="o2",
        center=(_GC_CENTER, 0),
        width=2 * half2,
        orientation=0,
        layer=LAYER.WG_MARK,
        port_type="vertical_te",
    )
    c.info["fiber_angle"] = 19.0
    c.info["polarization"] = "te"
    c.info["wavelength"] = 3.8
    return c


@gf.cell(tags=["cells"])
def rectangle(layer=LAYER.FLOORPLAN, **kwargs) -> gf.Component:
    """A rectangle.

    Args:
        layer: LAYER.FLOORPLAN.
        **kwargs: additional arguments.
    """
    return gf.c.rectangle(layer=layer, **kwargs)


@gf.cell(tags=["cells"])
def array(
    component: ComponentSpec = partial(straight, cross_section="xs_sus"),
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
        component: the component of which to create an array.
        columns: the number of components to place in the x-direction.
        rows: the number of components to place in the y-direction.
        add_ports: add ports to the component.
        size: Optional x, y size. Overrides columns and rows.
        centered: center the array around the origin.
        column_pitch: the pitch between columns.
        row_pitch: the pitch between rows.
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
