"""Private cell bodies shared by several cspdk flavours.

The flavours keep their own decorated public cells (names, defaults and
docstrings) as thin wrappers around these helpers.
"""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec, LayerSpec


def _grating_coupler_rectangular(
    *,
    period: float,
    n_periods: int,
    fill_factor: float,
    length_taper: float,
    width_grating: float,
    length_grating: float,
    grating_offset: float,
    teeth_overhang: float,
    wavelength: float,
    cross_section: CrossSectionSpec,
    layer_grating: LayerSpec,
    fiber_angle: float = 10.0,
) -> gf.Component:
    """Hand-drawn rectangular grating coupler of the Cornerstone SOI platforms.

    A linear taper from the waveguide (o1) to a ``width_grating`` wide
    section of total length ``length_taper + length_grating``, with
    ``n_periods`` etched teeth on ``layer_grating`` that overhang the
    waveguide by ``teeth_overhang`` on each side. The first tooth starts
    ``grating_offset`` after the taper; o2 is the fiber port at the centre of
    the teeth.

    Args:
        period: grating period in um.
        n_periods: number of teeth.
        fill_factor: etched tooth width as a fraction of the period.
        length_taper: length of the taper up to the grating width in um.
        width_grating: waveguide width under the grating in um.
        length_grating: length of the full-width waveguide after the taper.
        grating_offset: distance from the end of the taper to the first tooth.
        teeth_overhang: how far the teeth extend beyond each side of the waveguide.
        wavelength: design centre wavelength in um (stored in info).
        cross_section: cross-section of the waveguide port.
        layer_grating: layer of the etched teeth.
        fiber_angle: fiber angle in degrees (stored in info).
    """
    xs = gf.get_cross_section(cross_section)
    w0, w1 = xs.width / 2, width_grating / 2
    x1 = length_taper
    x2 = length_taper + length_grating
    c = gf.Component()
    c.add_polygon(
        [(0, -w0), (x1, -w1), (x2, -w1), (x2, w1), (x1, w1), (0, w0)], layer=xs.layer
    )
    tooth = gf.snap.snap_to_grid(period * fill_factor)
    y = width_grating / 2 + teeth_overhang
    x0 = length_taper + grating_offset
    for i in range(n_periods):
        xmin = gf.snap.snap_to_grid(x0 + i * period)
        c.add_polygon(
            [(xmin, -y), (xmin + tooth, -y), (xmin + tooth, y), (xmin, y)],
            layer=layer_grating,
        )
    xs.add_bbox(c)
    c.add_port(
        name="o1",
        center=(0, 0),
        width=xs.width,
        orientation=180,
        layer=xs.layer,
        cross_section=xs,
    )
    c.add_port(
        name="o2",
        port_type="vertical_te",
        center=(gf.snap.snap_to_grid(x0 + ((n_periods - 1) * period + tooth) / 2), 0),
        orientation=0,
        width=width_grating,
        layer=layer_grating,
    )
    c.info["polarization"] = "te"
    c.info["wavelength"] = wavelength
    c.info["fiber_angle"] = fiber_angle
    return c
