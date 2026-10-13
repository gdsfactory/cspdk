"""MMIs drawn like the Cornerstone Run-50 foundry MMIs."""

import math
from functools import partial

import gdsfactory as gf
import kfactory as kf
from gdsfactory.typings import CrossSectionSpec

from cspdk.si220._schematic import mmi1x2_schematic, mmi2x2_schematic
from cspdk.si220.tech import LAYER, get_band, is_rib

# Foundry defaults, keyed by (ports, band, rib): (MMI length, gap between the
# access tapers at the MMI body). From the SOI220_{band}_TE_{kind}_{1x2,2x2}_MMI
# cells of references/october-2026/CORNERSTONE-MPW-Run-50-GDSII-Template.gds and
# the 50th-call Standard Components PDF (pp. 8, 9, 15, 16, 39, 40, 43, 44).
_MMI_DEFAULTS: dict[tuple[str, str, bool], tuple[float, float]] = {
    ("1x2", "cband", False): (31.8, 1.64),
    ("1x2", "cband", True): (32.7, 1.64),
    ("1x2", "oband", False): (40.1, 1.55),
    ("1x2", "oband", True): (40.8, 1.55),
    ("2x2", "cband", False): (42.5, 0.53),
    ("2x2", "cband", True): (44.8, 0.53),
    ("2x2", "oband", False): (53.5, 0.53),
    ("2x2", "oband", True): (55.0, 0.53),
}
# Rib MMIs end in a straight so the ports sit in a full rib cross-section.
_LENGTH_STRAIGHT_RIB = 5.0


def _side_slab(
    x_body: float,
    x_port: float,
    direction: int,
    y_outer: float,
    width: float,
    width_taper: float,
    length_taper: float,
    offset: float,
    dbu: float,
) -> list[tuple[float, float]]:
    """Slab outline next to the access tapers on one side of the MMI body.

    The slab follows the outermost taper edge at ``offset``: the normal offset of
    the edge is taken at the corner of the body slab and continued with the taper
    slope until it meets the slab of the port straight, as the foundry draws it.
    Everything between the outermost tapers is slab.
    """
    slope = (width_taper - width) / 2 / length_taper
    x_corner = x_body + direction * offset
    edge = y_outer + width_taper / 2 - slope * offset
    y_corner = round((edge + offset * math.hypot(1, slope)) / dbu) * dbu
    y_port = y_outer + width / 2 + offset
    if slope:
        x_vertex = (
            round((x_corner + direction * (y_corner - y_port) / slope) / dbu) * dbu
        )
    else:
        x_vertex = x_corner
    if direction * (x_vertex - x_port) > 0:
        x_vertex = x_port
    return [
        (x_port, -y_port),
        (x_port, y_port),
        (x_vertex, y_port),
        (x_corner, y_corner),
        (x_corner, -y_corner),
        (x_vertex, -y_port),
    ]


def _mmi(
    inputs: int,
    width: float | None,
    width_taper: float,
    length_taper: float,
    length_mmi: float,
    width_mmi: float,
    gap_mmi: float,
    length_straight: float,
    cross_section: CrossSectionSpec,
) -> gf.Component:
    """An MMI with ``inputs`` (1 or 2) input and two output access tapers."""
    xs = gf.get_cross_section(cross_section)
    width = xs.width if width is None else width
    y_port = (gap_mmi + width_taper) / 2
    ys_in = (0.0,) if inputs == 1 else (-y_port, y_port)
    ys_out = (y_port, -y_port)
    x_in = -length_taper - length_straight
    x_out = length_mmi + length_taper + length_straight
    # (x of the body edge, x of the ports, direction away from the body, port ys)
    sides = ((0.0, x_in, -1, ys_in), (length_mmi, x_out, 1, ys_out))

    c = gf.Component()
    c.add_polygon(
        [
            (0, -width_mmi / 2),
            (0, width_mmi / 2),
            (length_mmi, width_mmi / 2),
            (length_mmi, -width_mmi / 2),
        ],
        layer=LAYER.WG,
    )
    for x_body, x_port, direction, ys in sides:
        x_taper = x_body + direction * length_taper
        for y in ys:
            c.add_polygon(
                [
                    (x_body, y - width_taper / 2),
                    (x_body, y + width_taper / 2),
                    (x_taper, y + width / 2),
                    (x_taper, y - width / 2),
                ],
                layer=LAYER.WG,
            )
            if length_straight:
                c.add_polygon(
                    [
                        (x_taper, y - width / 2),
                        (x_taper, y + width / 2),
                        (x_port, y + width / 2),
                        (x_port, y - width / 2),
                    ],
                    layer=LAYER.WG,
                )

    dbu = c.kcl.dbu
    for layer, offset in zip(xs.bbox_layers or (), xs.bbox_offsets or ()):
        slab = kf.kdb.Region()
        body = kf.kdb.DBox(
            -offset,
            -width_mmi / 2 - offset,
            length_mmi + offset,
            width_mmi / 2 + offset,
        )
        slab.insert(body.to_itype(dbu))
        for x_body, x_port, direction, ys in sides:
            outline = _side_slab(
                x_body,
                x_port,
                direction,
                max(abs(y) for y in ys),
                width,
                width_taper,
                length_taper,
                offset,
                dbu,
            )
            slab.insert(
                kf.kdb.DPolygon([kf.kdb.DPoint(*p) for p in outline]).to_itype(dbu)
            )
        clip = kf.kdb.DBox(x_in, body.bottom, x_out, body.top).to_itype(dbu)
        c.add_polygon((slab & kf.kdb.Region(clip)).merged(), layer=layer)

    ports = [("o1", x_in, ys_in[0], 180)]
    if inputs == 2:
        ports.append(("o2", x_in, ys_in[1], 180))
    ports += [
        (f"o{len(ports) + 1}", x_out, ys_out[0], 0),
        (f"o{len(ports) + 2}", x_out, ys_out[1], 0),
    ]
    for name, x, y, orientation in ports:
        c.add_port(
            name,
            center=(x, y),
            orientation=orientation,
            width=width,
            layer=LAYER.WG,
            port_type="optical",
        )
    return c


def _defaults(
    ports: str,
    cross_section: CrossSectionSpec,
    length_mmi: float | None,
    gap_mmi: float | None,
    length_straight: float | None,
) -> tuple[float, float, float]:
    rib = is_rib(cross_section)
    default_length, default_gap = _MMI_DEFAULTS[ports, get_band(cross_section), rib]
    if length_straight is None:
        length_straight = _LENGTH_STRAIGHT_RIB if rib else 0.0
    return (
        default_length if length_mmi is None else length_mmi,
        default_gap if gap_mmi is None else gap_mmi,
        length_straight,
    )


@gf.cell(tags=["mmis"], schematic_function=mmi1x2_schematic)
def mmi1x2(
    width: float | None = None,
    width_taper: float = 1.5,
    length_taper: float = 20.0,
    length_mmi: float | None = None,
    width_mmi: float = 6.0,
    gap_mmi: float | None = None,
    length_straight: float | None = None,
    cross_section: CrossSectionSpec = "strip_cband",
) -> gf.Component:
    """An mmi1x2.

    An mmi1x2 is a splitter that splits a single input to two outputs. The
    defaults reproduce the Cornerstone Run-50 1x2 MMI of the band and type.

    Args:
        width: the width of the waveguides connecting at the mmi ports.
        width_taper: the width at the base of the mmi body.
        length_taper: the length of the tapers going towards the mmi body.
        length_mmi: the length of the mmi body. Defaults to the foundry length.
        width_mmi: the width of the mmi body.
        gap_mmi: the gap between the tapers at the mmi body. Defaults to the
            foundry gap.
        length_straight: straight between each taper and its port. Defaults to
            5 um for rib and none for strip.
        cross_section: a cross section or its name or a function generating a cross section.
    """
    length_mmi, gap_mmi, length_straight = _defaults(
        "1x2", cross_section, length_mmi, gap_mmi, length_straight
    )
    return _mmi(
        1,
        width,
        width_taper,
        length_taper,
        length_mmi,
        width_mmi,
        gap_mmi,
        length_straight,
        cross_section,
    )


mmi1x2_rib = partial(mmi1x2, cross_section="rib_cband")


@gf.cell(tags=["mmis"], schematic_function=mmi2x2_schematic)
def mmi2x2(
    width: float | None = None,
    width_taper: float = 1.5,
    length_taper: float = 20.0,
    length_mmi: float | None = None,
    width_mmi: float = 6.0,
    gap_mmi: float | None = None,
    length_straight: float | None = None,
    cross_section: CrossSectionSpec = "strip_cband",
) -> gf.Component:
    """An mmi2x2.

    An mmi2x2 is a 2x2 splitter. The defaults reproduce the Cornerstone Run-50
    2x2 MMI of the band and type.

    Args:
        width: the width of the waveguides connecting at the mmi ports
        width_taper: the width at the base of the mmi body
        length_taper: the length of the tapers going towards the mmi body
        length_mmi: the length of the mmi body. Defaults to the foundry length.
        width_mmi: the width of the mmi body
        gap_mmi: the gap between the tapers at the mmi body. Defaults to the
            foundry gap.
        length_straight: straight between each taper and its port. Defaults to
            5 um for rib and none for strip.
        cross_section: a cross section or its name or a function generating a cross section.
    """
    length_mmi, gap_mmi, length_straight = _defaults(
        "2x2", cross_section, length_mmi, gap_mmi, length_straight
    )
    return _mmi(
        2,
        width,
        width_taper,
        length_taper,
        length_mmi,
        width_mmi,
        gap_mmi,
        length_straight,
        cross_section,
    )


mmi2x2_rib = partial(mmi2x2, cross_section="rib_cband")
