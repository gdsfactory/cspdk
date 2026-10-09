"""Grating couplers of the CORNERSTONE 220 nm SOI platform.

The rectangular gratings reproduce the MPW Run 50 template cells
(``references/october-2026/CORNERSTONE-MPW-Run-50-GDSII-Template.gds``):
``SOI220_cband_TE_strip_GC``, ``SOI220_cband_TE_rib_GC``,
``SOI220_oband_TE_strip_GC`` and ``SOI220_oband_TE_rib_GC``. The template cells
have their waveguide port at the origin facing east with the grating to the west;
these cells use the gdsfactory convention instead (``o1`` at the origin facing west,
grating to the east), i.e. they equal the template cells rotated by 180 degrees.
"""

import math
from functools import partial

import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec

from cspdk._schematic import _GRATING, sax_model, schematic
from cspdk.si220._schematic import (
    grating_coupler_elliptical_schematic,
    grating_coupler_rectangular_schematic,
)
from cspdk.si220.tech import LAYER, get_band, is_rib

##############################
# grating couplers Rectangular
##############################

# Run-50 template values that depend on the band (components PDF p6 and p37).
_RECTANGULAR_DEFAULTS = {
    "cband": {
        "period": 0.63,
        "n_periods": 60,
        "grating_offset": 0.342,
        "slab_taper_shift": 0.0,
        "wavelength": 1.55,
    },
    "oband": {
        "period": 0.5,
        "n_periods": 80,
        "grating_offset": 1.875,
        "slab_taper_shift": -0.042,
        "wavelength": 1.31,
    },
}


@gf.cell(
    tags=["grating_couplers"], schematic_function=grating_coupler_rectangular_schematic
)
def grating_coupler_rectangular(
    period: float | None = None,
    n_periods: int | None = None,
    length_taper: float = 350.0,
    wavelength: float | None = None,
    cross_section: CrossSectionSpec = "strip_cband",
    fill_factor: float = 0.5,
    width_grating: float = 10.0,
    length_grating: float = 42.0,
    grating_offset: float | None = None,
    grating_overhang: float = 0.5,
    length_straight: float | None = None,
    slab_taper_start: float | None = None,
) -> gf.Component:
    r"""A grating coupler with straight and parallel teeth (Run-50 standard GC).

    With the defaults it is the Run-50 template cell of the band and type of
    ``cross_section`` rotated by 180 degrees: a ``length_taper`` linear taper from the
    waveguide to a ``width_grating`` wide waveguide (GDS 3) of length
    ``length_grating``, with ``n_periods`` etched teeth (GDS 6) that overhang the
    waveguide by ``grating_overhang`` on each side. Rib gratings start with a
    ``length_straight`` input waveguide and are surrounded by the slab (GDS 5) of the
    rib cross-section.

    ```text
              length_taper    length_grating
           |<------------>|<---------------->|
                         _____________________
                     ___/| | | | | | | | |    |
           o1 ______/    | | | | | | | | |    |  width_grating
                    \___ | | | | | | | | |    |
                        \|_|_|_|_|_|_|_|_|____|
                         <> grating_offset
    ```

    Args:
        period: grating period in um. None: 0.63 (C-band) or 0.5 (O-band).
        n_periods: number of etched teeth. None: 60 (C-band) or 80 (O-band).
        length_taper: length of the taper from the waveguide to the grating.
        wavelength: centre wavelength in um, stored in ``info``. None: band centre.
        cross_section: a cross section or its name or a function generating a cross section.
        fill_factor: etched fraction of each period (tooth width / period).
        width_grating: width of the waveguide (GDS 3) under the grating.
        length_grating: length of the wide waveguide section under the grating.
        grating_offset: distance from the end of the taper to the first tooth.
            None: 0.342 (C-band) or 1.875 (O-band).
        grating_overhang: how far the teeth (GDS 6) extend past the waveguide on each
            side. The Cornerstone deck needs more than 0.2 um.
        length_straight: input waveguide before the taper. None: 5 um for rib, 0 for
            strip.
        slab_taper_start: x where the slab starts to widen (rib only). None: the
            template value, ``length_straight`` (C-band) or ``length_straight - 0.042``
            (O-band). The slab taper ends at the miter point of the slab offset.
    """
    band = get_band(cross_section)
    defaults = _RECTANGULAR_DEFAULTS[band]
    period = defaults["period"] if period is None else period
    n_periods = defaults["n_periods"] if n_periods is None else n_periods
    wavelength = defaults["wavelength"] if wavelength is None else wavelength
    if grating_offset is None:
        grating_offset = defaults["grating_offset"]
    if length_straight is None:
        length_straight = 5.0 if is_rib(cross_section) else 0.0
    if slab_taper_start is None:
        slab_taper_start = length_straight + defaults["slab_taper_shift"]

    xs = gf.get_cross_section(cross_section)
    w1 = xs.width / 2
    w2 = width_grating / 2
    x_taper = length_straight + length_taper
    x_end = x_taper + length_grating

    c = gf.Component()
    c.add_polygon(
        [
            (0, -w1),
            (length_straight, -w1),
            (x_taper, -w2),
            (x_end, -w2),
            (x_end, w2),
            (x_taper, w2),
            (length_straight, w1),
            (0, w1),
        ],
        layer=xs.layer,
    )

    # The slab keeps a constant offset from the waveguide; the wide end of its taper
    # is the miter point of the offset taper and waveguide edges.
    half_angle = math.atan((w2 - w1) / length_taper) / 2
    for layer, offset in zip(xs.bbox_layers or (), xs.bbox_offsets or ()):
        x_miter = x_taper - offset * math.tan(half_angle)
        s1 = w1 + offset
        s2 = w2 + offset
        c.add_polygon(
            [
                (0, -s1),
                (slab_taper_start, -s1),
                (x_miter, -s2),
                (x_end + offset, -s2),
                (x_end + offset, s2),
                (x_miter, s2),
                (slab_taper_start, s1),
                (0, s1),
            ],
            layer=layer,
        )

    tooth = period * fill_factor
    y_tooth = w2 + grating_overhang
    x_first = x_taper + grating_offset
    for i in range(n_periods):
        x0 = round(x_first + i * period, 3)
        c.add_polygon(
            [
                (x0, -y_tooth),
                (x0 + tooth, -y_tooth),
                (x0 + tooth, y_tooth),
                (x0, y_tooth),
            ],
            layer=LAYER.GRA,
        )

    c.add_port(
        name="o1", center=(0, 0), width=xs.width, orientation=180, cross_section=xs
    )
    x_last = x_first + (n_periods - 1) * period + tooth
    c.add_port(
        name="o2",
        port_type="vertical_te",
        center=(round((x_first + x_last) / 2, 3), 0),
        width=2 * y_tooth,
        orientation=0,
        layer=LAYER.GRA,
    )
    c.info["polarization"] = "te"
    c.info["wavelength"] = wavelength
    c.info["fiber_angle"] = 10.0
    c.info["period"] = period
    c.info["n_periods"] = n_periods
    return c


grating_coupler_rectangular_rib = partial(
    grating_coupler_rectangular,
    cross_section="rib_cband",
)


##############################
# Run-50 foundry cells
##############################


def _foundry_gc_schematic(name: str):
    return schematic(
        "grating-coupler",
        ["grating-coupler", "rectangular"],
        _GRATING,
        models=[sax_model(name, "cspdk.si220.models", ["o1", "o2"])],
    )


@gf.cell(
    tags=["grating_couplers"],
    schematic_function=_foundry_gc_schematic("SOI220_cband_TE_strip_GC"),
)
def SOI220_cband_TE_strip_GC() -> gf.Component:
    """Run-50 C-band strip grating coupler, rotated so ``o1`` faces west."""
    return grating_coupler_rectangular(cross_section="strip_cband")


@gf.cell(
    tags=["grating_couplers"],
    schematic_function=_foundry_gc_schematic("SOI220_cband_TE_rib_GC"),
)
def SOI220_cband_TE_rib_GC() -> gf.Component:
    """Run-50 C-band rib grating coupler, rotated so ``o1`` faces west."""
    return grating_coupler_rectangular(cross_section="rib_cband")


@gf.cell(
    tags=["grating_couplers"],
    schematic_function=_foundry_gc_schematic("SOI220_oband_TE_strip_GC"),
)
def SOI220_oband_TE_strip_GC() -> gf.Component:
    """Run-50 O-band strip grating coupler, rotated so ``o1`` faces west."""
    return grating_coupler_rectangular(cross_section="strip_oband")


@gf.cell(
    tags=["grating_couplers"],
    schematic_function=_foundry_gc_schematic("SOI220_oband_TE_rib_GC"),
)
def SOI220_oband_TE_rib_GC() -> gf.Component:
    """Run-50 O-band rib grating coupler, rotated so ``o1`` faces west."""
    return grating_coupler_rectangular(cross_section="rib_oband")


##############################
# grating couplers elliptical
##############################


@gf.cell(
    tags=["grating_couplers"], schematic_function=grating_coupler_elliptical_schematic
)
def grating_coupler_elliptical(
    wavelength: float = 1.55,
    grating_line_width: float = 0.35,
    cross_section: CrossSectionSpec = "strip_cband",
) -> gf.Component:
    """A grating coupler with curved but parallel teeth.

    This is not a Cornerstone standard component. ``grating_line_width`` is the width
    of the unetched ridge between two trenches (GDS 6). The Cornerstone deck limits
    GDS 6 gaps narrower than 350 nm to 20 um length, and the ridges here are arcs
    longer than that, so they are 350 nm wide (fill factor 0.51 at the 0.685 um
    period) rather than 315 nm. The wider ridge slightly raises the grating's mean
    effective index (a few nm red shift of the centre wavelength) and weakens the
    scattering strength; the period still follows ``wavelength``.

    Args:
        wavelength: the center wavelength for which the grating is designed
        grating_line_width: width of the unetched ridge between trenches, in um.
            Keep it at 0.35 or more to pass the Cornerstone deck.
        cross_section: a cross section or its name or a function generating a cross section.
    """
    return gf.c.grating_coupler_elliptical_trenches(
        polarization="te",
        wavelength=wavelength,
        grating_line_width=grating_line_width,
        taper_length=16.0,
        taper_angle=30.0,
        trenches_extra_angle=9.0,
        fiber_angle=15.0,
        neff=2.638,
        ncladding=1.443,
        layer_trench=LAYER.GRA,
        p_start=26,
        n_periods=30,
        end_straight_length=0.2,
        cross_section=cross_section,
    )
