"""Tapers."""

from collections.abc import Callable
from functools import partial

import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec

from cspdk.si220._schematic import (
    taper_metal_schematic,
    taper_mm2sm_schematic,
    taper_rib_to_strip_schematic,
    taper_schematic,
    transition_schematic,
)
from cspdk.si220.tech import LAYER, TECH


@gf.cell(tags=["tapers"], schematic_function=taper_schematic)
def taper(
    length: float = 10.0,
    width1: float = TECH.width,
    width2: float | None = None,
    cross_section: CrossSectionSpec = "strip_cband",
) -> gf.Component:
    """Linear taper, which tapers only the main cross section section.

    Args:
        length: taper length.
        width1: width of the west/left port.
        width2: width of the east/right port. Defaults to width1.
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
    """
    return gf.c.taper(
        length=length,
        width1=width1,
        width2=width2,
        cross_section=cross_section,
        layer=None,
        port=None,
        with_two_ports=True,
        port_names=("o1", "o2"),
        port_types=("optical", "optical"),
        with_bbox=True,
    )


@gf.cell(tags=["tapers"], schematic_function=taper_metal_schematic)
def taper_metal(
    length: float = 10.0,
    width1: float = TECH.width_metal,
    width2: float | None = None,
    cross_section: CrossSectionSpec = "metal_routing",
) -> gf.Component:
    """Linear taper, which tapers only the main cross section section.

    Args:
        length: taper length.
        width1: width of the west/left port.
        width2: width of the east/right port. Defaults to width1.
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
    """
    return gf.c.taper(
        length=length,
        width1=width1,
        width2=width2,
        cross_section=cross_section,
        layer=None,
        port=None,
        with_two_ports=True,
        port_names=("e1", "e2"),
        port_types=("electrical", "electrical"),
        with_bbox=True,
    )


@gf.cell(tags=["tapers"], schematic_function=transition_schematic)
def taper_strip_to_ridge(
    length: float = 10.0,
    width1: float | None = None,
    width2: float | None = None,
    w_slab1: float | None = None,
    w_slab2: float | None = None,
    cross_section: CrossSectionSpec = "strip_cband",
) -> gf.Component:
    """A taper between strip and ridge.

    This is a transition between two distinct cross sections. The slab starts as
    wide as the core, like the Cornerstone rib-to-strip taper, so it never draws a
    slab narrower than the 250 nm minimum feature size.

    Args:
        length: the length of the taper.
        width1: the input width of the taper. Defaults to the cross-section width.
        width2: the output width of the taper. Defaults to the cross-section width.
        w_slab1: the input slab width of the taper. Defaults to ``width1``.
        w_slab2: the output slab width of the taper. Defaults to ``width2`` plus
            the rib slab on each side.
        cross_section: a cross section or its name or a function generating a cross section.
    """
    width = gf.get_cross_section(cross_section).width
    width1 = width if width1 is None else width1
    width2 = width if width2 is None else width2
    w_slab1 = width1 if w_slab1 is None else w_slab1
    w_slab2 = width2 + 2 * TECH.width_slab if w_slab2 is None else w_slab2
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


trans_rib10 = partial(taper_strip_to_ridge, length=10)
trans_rib20 = partial(taper_strip_to_ridge, length=20)
trans_rib50 = partial(taper_strip_to_ridge, length=50)


def _add_slab(
    c: gf.Component,
    cross_section: CrossSectionSpec,
    outline: Callable[[float], list[tuple[float, float]]],
) -> None:
    """Draw ``outline(offset)`` on each slab layer of a rib ``cross_section``."""
    xs = gf.get_cross_section(cross_section)
    for layer, offset in zip(xs.bbox_layers or (), xs.bbox_offsets or ()):
        c.add_polygon(outline(offset), layer=layer)


def _add_ports(c: gf.Component, length: float, width1: float, width2: float) -> None:
    c.add_port(
        "o1",
        center=(0, 0),
        orientation=180,
        width=width1,
        layer=LAYER.WG,
        port_type="optical",
    )
    c.add_port(
        "o2",
        center=(length, 0),
        orientation=0,
        width=width2,
        layer=LAYER.WG,
        port_type="optical",
    )


@gf.cell(tags=["tapers"], schematic_function=taper_mm2sm_schematic)
def taper_mm2sm(
    length: float = 450.0,
    width_mm: float = 10.0,
    width: float | None = None,
    cross_section: CrossSectionSpec = "strip_cband",
) -> gf.Component:
    """Linear taper from a multimode to a single-mode waveguide.

    Drawn like the Cornerstone MM2SM tapers, which the C-band defaults reproduce.
    A rib cross-section adds its slab on each side of the core along the taper.

    Args:
        length: taper length (um).
        width_mm: multimode width at port o1 (um).
        width: single-mode width at port o2 (um). Defaults to the cross-section width.
        cross_section: strip or rib cross-section.
    """
    width = gf.get_cross_section(cross_section).width if width is None else width

    def outline(offset: float) -> list[tuple[float, float]]:
        return [
            (0, -width_mm / 2 - offset),
            (0, width_mm / 2 + offset),
            (length, width / 2 + offset),
            (length, -width / 2 - offset),
        ]

    c = gf.Component()
    c.add_polygon(outline(0), layer=LAYER.WG)
    _add_slab(c, cross_section, outline)
    _add_ports(c, length, width_mm, width)
    c.info["length"] = length
    return c


@gf.cell(tags=["tapers"], schematic_function=taper_rib_to_strip_schematic)
def taper_rib_to_strip(
    length: float = 50.0,
    length_rib: float = 5.0,
    length_strip: float = 5.0,
    width: float | None = None,
    cross_section: CrossSectionSpec = "rib_cband",
) -> gf.Component:
    """Rib-to-strip transition: the slab narrows to the core width.

    Drawn like the Cornerstone rib-to-strip taper, which the C-band defaults
    reproduce: ``length_rib`` of rib, a slab taper of ``length`` ending as wide as
    the core, then ``length_strip`` of strip. Port o1 is rib, o2 strip.

    Args:
        length: slab taper length (um).
        length_rib: rib straight before the taper (um).
        length_strip: strip straight after the taper (um).
        width: core width (um). Defaults to the cross-section width.
        cross_section: rib cross-section giving the slab.
    """
    width = gf.get_cross_section(cross_section).width if width is None else width
    total = length_rib + length + length_strip
    x_tip = length_rib + length

    def outline(offset: float) -> list[tuple[float, float]]:
        return [
            (0, -width / 2 - offset),
            (0, width / 2 + offset),
            (length_rib, width / 2 + offset),
            (x_tip, width / 2),
            (x_tip, -width / 2),
            (length_rib, -width / 2 - offset),
        ]

    c = gf.Component()
    c.add_polygon(
        [(0, -width / 2), (0, width / 2), (total, width / 2), (total, -width / 2)],
        layer=LAYER.WG,
    )
    _add_slab(c, cross_section, outline)
    _add_ports(c, total, width, width)
    c.info["length"] = length
    return c
