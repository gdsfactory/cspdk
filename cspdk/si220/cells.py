from functools import partial
from typing import Any

import gdsfactory as gf
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, LayerSpec

from cspdk.si220.config import PATH
from cspdk.si220.tech import LAYER, TECH

################
# Straights
################


@gf.cell
def _straight(
    length: float = 10.0, cross_section: CrossSectionSpec = "xs_sc", **kwargs
) -> gf.Component:
    return gf.components.straight(
        length=length,
        cross_section=cross_section,
        **kwargs,
    )


@gf.cell
def straight_sc(length: float = 10.0, **kwargs) -> gf.Component:
    """Straight waveguide in strip, c-band.

    Args:
        length (float, optional): The length of the waveguide. Defaults to 10.0.

    Returns:
        gf.Component: the component
    """
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_sc"
    return _straight(
        length=length,
        **kwargs,
    )


@gf.cell
def straight_so(length: float = 10.0, **kwargs) -> gf.Component:
    """Straight waveguide in strip, o-band.

    Args:
        length (float, optional): The length of the waveguide. Defaults to 10.0.

    Returns:
        gf.Component: the component
    """
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_so"
    return _straight(
        length=length,
        **kwargs,
    )


@gf.cell
def straight_rc(length: float = 10.0, **kwargs) -> gf.Component:
    """Straight waveguide in rib, c-band.

    Args:
        length (float, optional): The length of the waveguide. Defaults to 10.0.

    Returns:
        gf.Component: the component
    """
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_rc"
    return _straight(
        length=length,
        **kwargs,
    )


@gf.cell
def straight_ro(length: float = 10.0, **kwargs) -> gf.Component:
    """Straight waveguide in rib, o-band.

    Args:
        length (float, optional): The length of the waveguide. Defaults to 10.0.

    Returns:
        gf.Component: the component
    """
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_ro"
    return _straight(
        length=length,
        **kwargs,
    )


################
# Bends
################


@gf.cell
def bend_s(
    size: tuple[float, float] = (11.0, 1.8),
    cross_section: CrossSectionSpec = "xs_sc",
    **kwargs,
) -> gf.Component:
    """An S-bend.

    Args:
        size (tuple[float, float], optional): The size of the s-bend, in x and y. Defaults to (11.0, 1.8).
        cross_section (CrossSectionSpec, optional): the bend cross-section. Defaults to "xs_sc" (strip, c-band).

    Returns:
        gf.Component: the component
    """
    return gf.components.bend_s(
        size=size,
        cross_section=cross_section,
        **kwargs,
    )


@gf.cell
def _bend(
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.5,
    with_arc_floorplan: bool = True,
    npoints: int | None = None,
    cross_section: CrossSectionSpec = "xs_sc",
) -> gf.Component:
    return gf.components.bend_euler(
        radius=radius,
        angle=angle,
        p=p,
        with_arc_floorplan=with_arc_floorplan,
        npoints=npoints,
        cross_section=cross_section,
    )


@gf.cell
def wire_corner(
    cross_section: CrossSectionSpec = "xs_metal_routing",
) -> gf.Component:
    """The bend equivalent for electrical wires, which is a simple corner.

    Args:
        cross_section (CrossSectionSpec, optional): the bend cross-section. Defaults to "xs_metal_routing".

    Returns:
        gf.Component: the component
    """
    return gf.components.wire_corner(cross_section=cross_section)


def _float(x: Any) -> float:
    return float(x)


@gf.cell
def bend_sc(radius: float = 5, angle: float = 90.0, **kwargs) -> gf.Component:
    """An euler bend in strip, c-band.

    Args:
        radius (float, optional): the radius of the bend. Defaults to the PDK's default value for that cross-section.
        angle (float, optional): the angle of the bend. Defaults to 90.0.

    Returns:
        gf.Component: the component
    """
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_sc"
    return _bend(
        radius=radius,
        angle=angle,
        **kwargs,
    )


@gf.cell
def bend_so(
    radius: float = TECH.radius_so, angle: float = 90.0, **kwargs
) -> gf.Component:
    """An euler bend in strip, o-band.

    Args:
        radius (float, optional): the radius of the bend. Defaults to the PDK's default value for that cross-section.
        angle (float, optional): the angle of the bend. Defaults to 90.0.

    Returns:
        gf.Component: the component
    """
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_so"
    return _bend(
        radius=radius,
        angle=angle,
        **kwargs,
    )


@gf.cell
def bend_rc(
    radius: float = TECH.radius_rc, angle: float = 90.0, **kwargs
) -> gf.Component:
    """An euler bend in rib, c-band.

    Args:
        radius (float, optional): the radius of the bend. Defaults to the PDK's default value for that cross-section.
        angle (float, optional): the angle of the bend. Defaults to 90.0.

    Returns:
        gf.Component: the component
    """
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_rc"
    return _bend(
        radius=radius,
        angle=angle,
        **kwargs,
    )


@gf.cell
def bend_ro(
    radius: float = TECH.radius_ro, angle: float = 90.0, **kwargs
) -> gf.Component:
    """An euler bend in rib, o-band.

    Args:
        radius (float, optional): the radius of the bend. Defaults to the PDK's default value for that cross-section.
        angle (float, optional): the angle of the bend. Defaults to 90.0.

    Returns:
        gf.Component: the component
    """
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_ro"
    return _bend(
        radius=radius,
        angle=angle,
        **kwargs,
    )


################
# Transitions
################


@gf.cell
def _taper(
    length: float = 10.0,
    width1: float = 0.5,
    width2: float | None = None,
    port: gf.Port | None = None,
    cross_section: CrossSectionSpec = "xs_sc",
    **kwargs,
) -> gf.Component:
    return gf.components.taper(
        length=length,
        width1=width1,
        width2=width2,
        port=port,
        cross_section=cross_section,
        **kwargs,
    )


@gf.cell
def taper_sc(
    length: float = 10.0,
    width1: float = 0.5,
    width2: float | None = None,
    port: gf.Port | None = None,
    **kwargs,
) -> gf.Component:
    """A width taper in strip, c-band.

    Args:
        length (float, optional): the length of the taper, in um. Defaults to 10.0.
        width1 (float, optional): the width of the taper input, in um. Defaults to 0.5.
        width2 (float | None, optional): the width of the taper output, in um. Defaults to None.
        port (gf.Port | None, optional): if given, starts from the port's width and transitions to width1. Defaults to None.

    Returns:
        gf.Component: the component
    """
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_sc"
    return _taper(length=length, width1=width1, width2=width2, port=port, **kwargs)


@gf.cell
def taper_so(
    length: float = 10.0,
    width1: float = 0.5,
    width2: float | None = None,
    port: gf.Port | None = None,
    **kwargs,
) -> gf.Component:
    """A width taper in strip, o-band.

    Args:
        length (float, optional): the length of the taper, in um. Defaults to 10.0.
        width1 (float, optional): the width of the taper input, in um. Defaults to 0.5.
        width2 (float | None, optional): the width of the taper output, in um. Defaults to None.
        port (gf.Port | None, optional): if given, starts from the port's width and transitions to width1. Defaults to None.

    Returns:
        gf.Component: the component
    """
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_so"
    return _taper(length=length, width1=width1, width2=width2, port=port, **kwargs)


@gf.cell
def taper_rc(
    length: float = 10.0,
    width1: float = 0.5,
    width2: float | None = None,
    port: gf.Port | None = None,
    **kwargs,
) -> gf.Component:
    """A width taper in rib, c-band.

    Args:
        length (float, optional): the length of the taper, in um. Defaults to 10.0.
        width1 (float, optional): the width of the taper input, in um. Defaults to 0.5.
        width2 (float | None, optional): the width of the taper output, in um. Defaults to None.
        port (gf.Port | None, optional): if given, starts from the port's width and transitions to width1. Defaults to None.

    Returns:
        gf.Component: the component
    """
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_rc"
    return _taper(length=length, width1=width1, width2=width2, port=port, **kwargs)


@gf.cell
def taper_ro(
    length: float = 10.0,
    width1: float = 0.5,
    width2: float | None = None,
    port: gf.Port | None = None,
    **kwargs,
) -> gf.Component:
    """A width taper in rib, o-band.

    Args:
        length (float, optional): the length of the taper, in um. Defaults to 10.0.
        width1 (float, optional): the width of the taper input, in um. Defaults to 0.5.
        width2 (float | None, optional): the width of the taper output, in um. Defaults to None.
        port (gf.Port | None, optional): if given, starts from the port's width and transitions to width1. Defaults to None.

    Returns:
        gf.Component: the component
    """
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_ro"
    return _taper(length=length, width1=width1, width2=width2, port=port, **kwargs)


@gf.cell
def trans_sc_rc10() -> gf.Component:
    """A 10um-long strip-rib transition in c-band."""
    return gf.c.taper_strip_to_ridge(
        length=10,
        w_slab1=0.2,
        w_slab2=10.45,
        cross_section="xs_sc",
        layer_wg=LAYER.WG,
        layer_slab=LAYER.SLAB,
    )


@gf.cell
def trans_sc_rc20() -> gf.Component:
    """A 20um-long strip-rib transition in c-band.

    Returns:
        gf.Component: the component
    """
    return gf.c.taper_strip_to_ridge(
        length=20,
        w_slab1=0.2,
        w_slab2=10.45,
        cross_section="xs_sc",
        layer_wg=LAYER.WG,
        layer_slab=LAYER.SLAB,
    )


@gf.cell
def trans_sc_rc50() -> gf.Component:
    """A 50um-long strip-rib transition in c-band.

    Returns:
        gf.Component: the component
    """
    return gf.c.taper_strip_to_ridge(
        length=50,
        w_slab1=0.2,
        w_slab2=10.45,
        cross_section="xs_sc",
        layer_wg=LAYER.WG,
        layer_slab=LAYER.SLAB,
    )


################
# MMIs
################


@gf.cell
def _mmi1x2(
    width_mmi: float = 6.0,
    width_taper: float = 1.5,
    length_taper: float = 20.0,
    cross_section: CrossSectionSpec = "xs_sc",
    **kwargs,
) -> gf.Component:
    return gf.components.mmi1x2(
        width_mmi=width_mmi,
        length_taper=length_taper,
        width_taper=width_taper,
        cross_section=cross_section,
        **kwargs,
    )


@gf.cell
def _mmi2x2(
    width_mmi: float = 6.0,
    width_taper: float = 1.5,
    length_taper: float = 20.0,
    cross_section: CrossSectionSpec = "xs_sc",
    **kwargs,
) -> gf.Component:
    return gf.components.mmi2x2(
        width_mmi=width_mmi,
        length_taper=length_taper,
        width_taper=width_taper,
        cross_section=cross_section,
        **kwargs,
    )


################
# MMIs rib cband
################


@gf.cell
def mmi1x2_rc(**kwargs) -> gf.Component:
    """A 1x2 MMI in rib, c-band.

    Returns:
        gf.Component: the component
    """
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_rc"
    return _mmi1x2(
        length_mmi=32.7,
        gap_mmi=1.64,
        **kwargs,
    )


@gf.cell
def mmi2x2_rc(**kwargs) -> gf.Component:
    """A 2x2 MMI in rib, c-band.

    Returns:
        gf.Component: the component
    """
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_rc"
    return _mmi2x2(
        length_mmi=44.8,
        gap_mmi=0.53,
        **kwargs,
    )


################
# MMIs strip cband
################


@gf.cell
def mmi1x2_sc(**kwargs) -> gf.Component:
    """A 1x2 MMI in strip, c-band.

    Returns:
        gf.Component: the component
    """
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_sc"
    return _mmi1x2(
        length_mmi=31.8,
        gap_mmi=1.64,
        **kwargs,
    )


@gf.cell
def mmi2x2_sc(**kwargs) -> gf.Component:
    """A 2x2 MMI in rib, c-band.

    Returns:
        gf.Component: the component
    """
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_sc"
    return _mmi2x2(
        length_mmi=42.5,
        gap_mmi=0.5,
        **kwargs,
    )


################
# MMIs rib oband
################


@gf.cell
def mmi1x2_ro(**kwargs) -> gf.Component:
    """A 1x2 MMI in rib, o-band.

    Returns:
        gf.Component: the component
    """
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_ro"
    return _mmi1x2(
        length_mmi=40.8,
        gap_mmi=1.55,
        **kwargs,
    )


@gf.cell
def mmi2x2_ro(**kwargs) -> gf.Component:
    """A 2x2 MMI in rib, o-band.

    Returns:
        gf.Component: the component
    """
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_ro"
    return _mmi2x2(
        length_mmi=55.0,
        gap_mmi=0.53,
        **kwargs,
    )


################
# MMIs strip oband
################


@gf.cell
def mmi1x2_so(**kwargs) -> gf.Component:
    """A 1x2 MMI in strip, o-band.

    Returns:
        gf.Component: the component
    """
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_so"
    return _mmi1x2(
        length_mmi=40.1,
        gap_mmi=1.55,
        **kwargs,
    )


@gf.cell
def mmi2x2_so(**kwargs) -> gf.Component:
    """A 2x2 MMI in strip, o-band.

    Returns:
        gf.Component: the component
    """
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_so"
    return _mmi2x2(
        length_mmi=53.5,
        gap_mmi=0.53,
        **kwargs,
    )


##############################
# Evanescent couplers
##############################


@gf.cell
def _coupler_symmetric(
    bend: ComponentSpec = bend_s,
    gap: float = 0.234,
    dx: float = 10.0,
    dy: float = 4.0,
    cross_section: CrossSectionSpec = "xs_sc",
) -> gf.Component:
    return gf.components.coupler_symmetric(
        bend=bend,
        gap=gap,
        dx=dx,
        dy=dy,
        cross_section=cross_section,
    )


@gf.cell
def _coupler_straight(
    length: float = 10.0,
    gap: float = 0.27,
    straight: ComponentSpec = _straight,
    cross_section: CrossSectionSpec = "xs_sc",
) -> gf.Component:
    return gf.components.coupler_straight(
        length=length,
        gap=gap,
        straight=straight,
        cross_section=cross_section,
    )


@gf.cell
def _coupler(
    gap: float = 0.236,
    length: float = 20.0,
    coupler_symmetric: ComponentSpec = _coupler_symmetric,
    coupler_straight: ComponentSpec = _coupler_straight,
    dx: float = 10.0,
    dy: float = 4.0,
    cross_section: CrossSectionSpec = "xs_sc",
) -> gf.Component:
    return gf.components.coupler(
        gap=gap,
        length=length,
        coupler_symmetric=coupler_symmetric,
        coupler_straight=coupler_straight,
        dx=dx,
        dy=dy,
        cross_section=cross_section,
    )


@gf.cell
def coupler_sc(
    gap: float = 0.236,
    length: float = 20.0,
    dx: float = 10.0,
    dy: float = 4.0,
    **kwargs,
) -> gf.Component:
    """A symmetric coupler in strip, c-band.

    Args:
        gap (float, optional): the coupling gap, in um. Defaults to 0.236.
        length (float, optional): the length of the coupling section, in um. Defaults to 20.0.
        dx (float, optional): the port-to-port horizontal spacing. Defaults to 10.0.
        dy (float, optional): the port-to-port vertical spacing. Defaults to 4.0.

    Returns:
        gf.Component: the component
    """
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_sc"
    return _coupler(
        gap=gap,
        length=length,
        dx=dx,
        dy=dy,
        **kwargs,
    )


@gf.cell
def coupler_so(
    gap: float = 0.236,
    length: float = 20.0,
    dx: float = 10.0,
    dy: float = 4.0,
    **kwargs,
) -> gf.Component:
    """A symmetric coupler in strip, o-band.

    Args:
        gap (float, optional): the coupling gap, in um. Defaults to 0.236.
        length (float, optional): the length of the coupling section, in um. Defaults to 20.0.
        dx (float, optional): the port-to-port horizontal spacing. Defaults to 10.0.
        dy (float, optional): the port-to-port vertical spacing. Defaults to 4.0.

    Returns:
        gf.Component: the component
    """
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_so"
    return _coupler(
        gap=gap,
        length=length,
        dx=dx,
        dy=dy,
        **kwargs,
    )


@gf.cell
def coupler_rc(
    gap: float = 0.236,
    length: float = 20.0,
    dx: float = 10.0,
    dy: float = 4.0,
    **kwargs,
) -> gf.Component:
    """A symmetric coupler in rib, c-band.

    Args:
        gap (float, optional): the coupling gap, in um. Defaults to 0.236.
        length (float, optional): the length of the coupling section, in um. Defaults to 20.0.
        dx (float, optional): the port-to-port horizontal spacing. Defaults to 10.0.
        dy (float, optional): the port-to-port vertical spacing. Defaults to 4.0.

    Returns:
        gf.Component: the component
    """
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_rc"
    return _coupler(
        gap=gap,
        length=length,
        dx=dx,
        dy=dy,
        **kwargs,
    )


@gf.cell
def coupler_ro(
    gap: float = 0.236,
    length: float = 20.0,
    dx: float = 10.0,
    dy: float = 4.0,
    **kwargs,
) -> gf.Component:
    """A symmetric coupler in rib, o-band.

    Args:
        gap (float, optional): the coupling gap, in um. Defaults to 0.236.
        length (float, optional): the length of the coupling section, in um. Defaults to 20.0.
        dx (float, optional): the port-to-port horizontal spacing. Defaults to 10.0.
        dy (float, optional): the port-to-port vertical spacing. Defaults to 4.0.

    Returns:
        gf.Component: the component
    """
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_ro"
    return _coupler(
        gap=gap,
        length=length,
        dx=dx,
        dy=dy,
        **kwargs,
    )


##############################
# grating couplers Rectangular
##############################


_gc_rectangular = partial(
    gf.components.grating_coupler_rectangular,
    n_periods=30,
    fill_factor=0.5,
    length_taper=350.0,
    fiber_angle=10.0,
    layer_grating=LAYER.GRA,
    layer_slab=LAYER.WG,
    slab_offset=0.0,
    period=0.75,
    width_grating=11.0,
    polarization="te",
    wavelength=1.55,
    taper=_taper,
    slab_xmin=-1.0,
    cross_section="xs_sc",
)


@gf.cell
def gc_rectangular_so() -> gf.Component:
    """A rectangular grating coupler in strip, o-band.

    Returns:
        gf.Component: the component
    """
    return _gc_rectangular(
        period=0.5,
        cross_section="xs_so",
        n_periods=80,
    )


@gf.cell
def gc_rectangular_ro() -> gf.Component:
    """A rectangular grating coupler in rib, o-band.

    Returns:
        gf.Component: the component
    """
    return _gc_rectangular(
        period=0.5,
        cross_section="xs_ro",
        n_periods=80,
    )


@gf.cell
def gc_rectangular_sc() -> gf.Component:
    """A rectangular grating coupler in strip, c-band.

    Returns:
        gf.Component: the component
    """
    return _gc_rectangular(
        period=0.63,
        cross_section="xs_sc",
        fiber_angle=10,
        n_periods=60,
    )


@gf.cell
def gc_rectangular_rc() -> gf.Component:
    """A rectangular grating coupler in rib, c-band.

    Returns:
        gf.Component: the component
    """
    return _gc_rectangular(
        period=0.5,
        cross_section="xs_rc",
        n_periods=60,
    )


##############################
# grating couplers elliptical
##############################


@gf.cell
def _gc_elliptical(
    polarization: str = "te",
    taper_length: float = 16.6,
    taper_angle: float = 30.0,
    trenches_extra_angle: float = 9.0,
    wavelength: float = 1.53,
    fiber_angle: float = 15.0,
    grating_line_width: float = 0.343,
    neff: float = 2.638,
    ncladding: float = 1.443,
    layer_trench: LayerSpec = LAYER.GRA,
    p_start: int = 26,
    n_periods: int = 30,
    end_straight_length: float = 0.2,
    cross_section: CrossSectionSpec = "xs_sc",
) -> gf.Component:
    return gf.components.grating_coupler_elliptical_trenches(
        polarization=polarization,
        taper_length=taper_length,
        taper_angle=taper_angle,
        trenches_extra_angle=trenches_extra_angle,
        wavelength=wavelength,
        fiber_angle=fiber_angle,
        grating_line_width=grating_line_width,
        neff=neff,
        ncladding=ncladding,
        layer_trench=layer_trench,
        p_start=p_start,
        n_periods=n_periods,
        end_straight_length=end_straight_length,
        cross_section=cross_section,
    )


@gf.cell
def gc_elliptical_sc(
    grating_line_width: float = 0.343,
    fiber_angle: float = 15,
    wavelength: float = 1.53,
    **kwargs,
) -> gf.Component:
    """An elliptical grating coupler in strip, c-band.

    Args:
        grating_line_width (float, optional): the grating line width, in um. Defaults to 0.343.
        fiber_angle (float, optional): the fiber angle, in degrees. Defaults to 15.
        wavelength (float, optional): the center wavelength, in um. Defaults to 1.53.

    Returns:
        gf.Component: the component
    """
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_sc"
    return _gc_elliptical(
        grating_line_width=grating_line_width,
        fiber_angle=fiber_angle,
        wavelength=wavelength,
        **kwargs,
    )


@gf.cell
def gc_elliptical_so(
    grating_line_width: float = 0.343,
    fiber_angle: float = 15,
    wavelength: float = 1.31,
    **kwargs,
) -> gf.Component:
    """An elliptical grating coupler in strip, o-band.

    Args:
        grating_line_width (float, optional): the grating line width, in um. Defaults to 0.343.
        fiber_angle (float, optional): the fiber angle, in degrees. Defaults to 15.
        wavelength (float, optional): the center wavelength, in um. Defaults to 1.31.

    Returns:
        gf.Component: the component
    """
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_so"
    return _gc_elliptical(
        grating_line_width=grating_line_width,
        fiber_angle=fiber_angle,
        wavelength=wavelength,
        **kwargs,
    )


################
# MZI
################


@gf.cell
def _mzi(
    delta_length: float = 10.0,
    length_y: float = 2.0,
    length_x: float = 0.1,
    cross_section: CrossSectionSpec = "xs_sc",
    bend: ComponentSpec = _bend,
    straight: ComponentSpec = _straight,
    splitter: ComponentSpec = _mmi1x2,
    combiner: ComponentSpec = _mmi2x2,
) -> gf.Component:
    return gf.components.mzi(
        delta_length=delta_length,
        length_y=length_y,
        length_x=length_x,
        bend=bend,
        straight=straight,
        straight_y=straight,
        straight_x_top=straight,
        straight_x_bot=straight,
        splitter=splitter,
        combiner=combiner,
        with_splitter=True,
        port_e1_splitter="o2",
        port_e0_splitter="o3",
        port_e1_combiner="o2",
        port_e0_combiner="o3",
        nbends=2,
        cross_section=cross_section,
        cross_section_x_top=cross_section,
        cross_section_x_bot=cross_section,
        mirror_bot=False,
        add_optical_ports_arms=False,
        min_length=0.01,
    )


@gf.cell
def mzi_sc(
    delta_length: float = 10.0,
    length_y: float = 2.0,
    length_x: float = 0.1,
    add_electrical_ports_bot: bool = True,
    cross_section: CrossSectionSpec = "xs_sc",
    **kwargs,
) -> gf.Component:
    """A Mach-Zehnder Interferometer (MZI) in strip, c-band.

    Args:
        delta_length (float, optional): the length differential between the two arms. Defaults to 10.0.
        length_y (float, optional): the common vertical length, in um. Defaults to 2.0.
        length_x (float, optional): the common horizontal length, in um. Defaults to 0.1.
        add_electrical_ports_bot (bool, optional): if true, adds electrical ports to the bottom. Defaults to True.

    Returns:
        gf.Component: the component
    """
    return _mzi(
        delta_length=delta_length,
        length_y=length_y,
        length_x=length_x,
        straight=straight_sc,
        bend=bend_sc,
        combiner=mmi1x2_sc,
        splitter=mmi1x2_sc,
        cross_section=cross_section,
        **kwargs,
    )


@gf.cell
def mzi_so(
    delta_length: float = 10.0,
    length_y: float = 2.0,
    length_x: float = 0.1,
    **kwargs,
) -> gf.Component:
    """A Mach-Zehnder Interferometer (MZI) in strip, o-band.

    Args:
        delta_length (float, optional): the length differential between the two arms. Defaults to 10.0.
        length_y (float, optional): the common vertical length, in um. Defaults to 2.0.
        length_x (float, optional): the common horizontal length, in um. Defaults to 0.1.

    Returns:
        gf.Component: the component
    """
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_so"
    return _mzi(
        delta_length=delta_length,
        length_y=length_y,
        length_x=length_x,
        straight=straight_so,
        bend=bend_so,
        combiner=mmi1x2_so,
        splitter=mmi1x2_so,
        **kwargs,
    )


@gf.cell
def mzi_rc(
    delta_length: float = 10.0,
    length_y: float = 2.0,
    length_x: float = 0.1,
    **kwargs,
) -> gf.Component:
    """A Mach-Zehnder Interferometer (MZI) in rib, c-band.

    Args:
        delta_length (float, optional): the length differential between the two arms. Defaults to 10.0.
        length_y (float, optional): the common vertical length, in um. Defaults to 2.0.
        length_x (float, optional): the common horizontal length, in um. Defaults to 0.1.

    Returns:
        gf.Component: the component
    """
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_rc"
    return _mzi(
        delta_length=delta_length,
        length_y=length_y,
        length_x=length_x,
        straight=straight_rc,
        bend=bend_rc,
        combiner=mmi1x2_rc,
        splitter=mmi1x2_rc,
        **kwargs,
    )


@gf.cell
def mzi_ro(
    delta_length: float = 10.0,
    length_y: float = 2.0,
    length_x: float = 0.1,
    add_electrical_ports_bot: bool = True,
    **kwargs,
) -> gf.Component:
    """A Mach-Zehnder Interferometer (MZI) in rib, o-band.

    Args:
        delta_length (float, optional): the length differential between the two arms. Defaults to 10.0.
        length_y (float, optional): the common vertical length, in um. Defaults to 2.0.
        length_x (float, optional): the common horizontal length, in um. Defaults to 0.1.
        add_electrical_ports_bot (bool, optional): if true, adds electrical ports to the bottom. Defaults to True.

    Returns:
        gf.Component: the component
    """
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_ro"
    return _mzi(
        delta_length=delta_length,
        length_y=length_y,
        length_x=length_x,
        add_electrical_ports_bot=add_electrical_ports_bot,
        straight=straight_ro,
        bend=bend_ro,
        combiner=mmi1x2_ro,
        splitter=mmi1x2_ro,
        **kwargs,
    )


################
# Packaging
################


@gf.cell
def pad(
    size: tuple[float, float] = (100.0, 100.0),
    layer: LayerSpec = LAYER.PAD,
    bbox_layers: None = None,
    bbox_offsets: None = None,
    port_inclusion: float = 0.0,
    port_orientation: float = 0,
) -> gf.Component:
    """An electrical pad.

    Args:
        size (tuple[float, float], optional): Size of the pad in (x, y). Defaults to (100.0, 100.0).
        layer (LayerSpec, optional): the layer to draw the pad on. Defaults to LAYER.PAD.
        bbox_layers (None, optional): if set, draws a box around the pad with the given layers. Defaults to None.
        bbox_offsets (None, optional): if set, applies an offset to grow the bbox's specified with `bbox_layers`. Defaults to None.
        port_inclusion (float, optional): if set, insets the port from the edge by the specified amount. Defaults to 0.0.
        port_orientation (None, optional): if set, gives the port a fixed orientation. Defaults to None.

    Returns:
        gf.Component: the component
    """
    return gf.components.pad(
        size=size,
        layer=layer,
        bbox_layers=bbox_layers,
        bbox_offsets=bbox_offsets,
        port_inclusion=port_inclusion,
        port_orientation=port_orientation,
    )


@gf.cell
def rectangle(
    size: tuple[float, float] = (4.0, 2.0),
    layer: LayerSpec = LAYER.FLOORPLAN,
    centered: bool = False,
    port_type: str = "electrical",
    port_orientations: tuple[float, float, float, float] = (180.0, 90.0, 0.0, -90.0),
) -> gf.Component:
    """A simple rectangle on the given layer.

    Args:
        size (tuple[float, float], optional): the size of the rectangle in (x, y). Defaults to (4.0, 2.0).
        layer (LayerSpec, optional): the layer to draw the rectangle on. Defaults to LAYER.FLOORPLAN.
        centered (bool, optional): if true, the rectangle's origin will be placed at the center (otherwise it will be bottom-left). Defaults to False.
        port_type (str, optional): the port type for ports automatically added to edges of the rectangle. Defaults to "electrical".
        port_orientations (tuple[float, float, float, float], optional): orientations of the ports to be automatically added. Defaults to (180.0, 90.0, 0.0, -90.0).

    Returns:
        gf.Component: the component
    """
    return gf.components.rectangle(
        size=size,
        layer=layer,
        centered=centered,
        port_type=port_type,
        port_orientations=port_orientations,
    )


@gf.cell
def grating_coupler_array(
    pitch: float = 127.0,
    n: int = 6,
    port_name: str = "o1",
    rotation: float = -90,
    with_loopback: bool = False,
    grating_coupler_spacing: float = 0.0,
    grating_coupler: ComponentSpec = gc_rectangular_sc,
    cross_section: CrossSectionSpec = "xs_sc",
) -> gf.Component:
    """An array of grating couplers.

    Args:
        pitch (float, optional): the center-center pitch between grating couplers. Defaults to 127.0.
        n (int, optional): the number of grating couplers to place. Defaults to 6.
        port_name (str, optional): the routing port of the grating coupler to be placed. Defaults to "o1".
        rotation (float, optional): rotation of the grating couplers, in degrees. Defaults to 0.0.
        with_loopback (bool, optional): if True, adds a loopback. Defaults to False.
        bend (ComponentSpec, optional): the bend to be used for the loopback. Defaults to _bend.
        grating_coupler_spacing (float, optional): the spacing to be used in the loopback. Defaults to 0.0.
        grating_coupler (ComponentSpec, optional): the grating coupler component to use.
        cross_section (CrossSectionSpec, optional): the cross section to be used for routing in the loopback.

    Returns:
        gf.Component: the component
    """
    return gf.components.grating_coupler_array(
        pitch=pitch,
        n=n,
        port_name=port_name,
        rotation=rotation,
        with_loopback=with_loopback,
        grating_coupler=grating_coupler,
        cross_section=cross_section,
    )


@gf.cell
def _die(
    size: tuple[float, float] = (11470.0, 4900.0),
    ngratings: int = 14,
    npads: int = 31,
    grating_pitch: float = 250.0,
    pad_pitch: float = 300.0,
    grating_coupler: ComponentSpec = gc_rectangular_sc,
    cross_section: CrossSectionSpec = "xs_sc",
    pad: ComponentSpec = pad,
) -> gf.Component:
    c = gf.Component()

    fp = c << rectangle(size=size, layer=LAYER.FLOORPLAN, centered=True)

    # Add optical ports
    x0 = -4925 + 2.827
    y0 = 1650

    gca = grating_coupler_array(
        n=ngratings,
        pitch=grating_pitch,
        with_loopback=True,
        grating_coupler=grating_coupler,
        cross_section=cross_section,
    )
    left = c << gca
    left.drotate(-90)
    left.dxmax = x0
    left.dy = fp.y
    c.add_ports(left.ports, prefix="W")

    right = c << gca
    right.drotate(+90)
    right.dxmax = -x0
    right.dy = fp.y
    c.add_ports(right.ports, prefix="E")

    # Add electrical ports
    x0 = -4615
    y0 = 2200
    pad = pad()

    for i in range(npads):
        pad_ref = c << pad
        pad_ref.dxmin = x0 + i * pad_pitch
        pad_ref.ymin = y0
        c.add_port(
            name=f"N{i}",
            port=pad_ref.ports["e4"],
        )

    for i in range(npads):
        pad_ref = c << pad
        pad_ref.dxmin = x0 + i * pad_pitch
        pad_ref.dymax = -y0
        c.add_port(
            name=f"S{i}",
            port=pad_ref.ports["e2"],
        )

    c.auto_rename_ports()
    return c


@gf.cell
def die_sc() -> gf.Component:
    """The standard die template for strip, c-band. This has 24 grating couplers, split evenly between the left and right sides of the chip and 62 electrical pads split between the top and bottom.

    Returns:
        gf.Component: the component
    """
    return _die(
        grating_coupler=gc_rectangular_sc,
        cross_section="xs_sc",
    )


@gf.cell
def die_so() -> gf.Component:
    """The standard die template for strip, o-band. This has 24 grating couplers, split evenly between the left and right sides of the chip and 62 electrical pads split between the top and bottom.

    Returns:
        gf.Component: the component
    """
    return _die(
        grating_coupler=gc_rectangular_so,
        cross_section="xs_so",
    )


@gf.cell
def die_rc() -> gf.Component:
    """The standard die template for rib, c-band. This has 24 grating couplers, split evenly between the left and right sides of the chip and 62 electrical pads split between the top and bottom.

    Returns:
        gf.Component: the component
    """
    return _die(
        grating_coupler=gc_rectangular_rc,
        cross_section="xs_rc",
    )


@gf.cell
def die_ro() -> gf.Component:
    """The standard die template for rib, o-band. This has 24 grating couplers, split evenly between the left and right sides of the chip and 62 electrical pads split between the top and bottom.

    Returns:
        gf.Component: the component
    """
    return _die(
        grating_coupler=gc_rectangular_ro,
        cross_section="xs_ro",
    )


################
# Imported from Cornerstone MPW SOI 220nm GDSII Template
################
_import_gds = partial(gf.import_gds, gdsdir=PATH.gds)


@gf.cell
def heater() -> gf.Component:
    """Heater fixed cell."""
    heater = _import_gds("Heater.gds")
    heater.name = "heater"
    return heater


@gf.cell
def crossing_so() -> gf.Component:
    """SOI220nm_1310nm_TE_STRIP_Waveguide_Crossing fixed cell."""
    c = _import_gds("SOI220nm_1310nm_TE_STRIP_Waveguide_Crossing.gds")

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
    c = _import_gds("SOI220nm_1550nm_TE_RIB_Waveguide_Crossing.gds")
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
    c = _import_gds("SOI220nm_1550nm_TE_STRIP_Waveguide_Crossing.gds")
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
    c = die_sc()
    # c = mzi_rc()
    # c = trans_sc_rc20()
    # c = crossing_sc()
    c.show()
    # for name, func in list(globals().items()):
    #     if not callable(func):
    #         continue
    #     if name in ["partial", "_import_gds"]:
    #         continue
    #     print(name, func())
