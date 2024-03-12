from functools import partial
from typing import Any

import gdsfactory as gf
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, LayerSpec

from cspdk.config import PATH
from cspdk.tech import LAYER, xs_nc, xs_no, xs_rc, xs_ro, xs_sc, xs_so

################
# Straights
################


@gf.cell
def _straight(
    length: float = 10.0,
    cross_section: CrossSectionSpec = "xs_sc",
) -> gf.Component:
    return gf.components.straight(
        length=length,
        cross_section=cross_section,
    )


@gf.cell
def straight_sc(length: float = 10.0, **kwargs) -> gf.Component:
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_sc"
    return _straight(
        length=length,
        **kwargs,
    )


@gf.cell
def straight_so(length: float = 10.0, **kwargs) -> gf.Component:
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_so"
    return _straight(
        length=length,
        **kwargs,
    )


@gf.cell
def straight_rc(length: float = 10.0, **kwargs) -> gf.Component:
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_rc"
    return _straight(
        length=length,
        **kwargs,
    )


@gf.cell
def straight_ro(length: float = 10.0, **kwargs) -> gf.Component:
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_ro"
    return _straight(
        length=length,
        **kwargs,
    )


@gf.cell
def straight_nc(length: float = 10.0, **kwargs) -> gf.Component:
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_nc"
    return _straight(
        length=length,
        **kwargs,
    )


@gf.cell
def straight_no(length: float = 10.0, **kwargs) -> gf.Component:
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_no"
    return _straight(
        length=length,
        **kwargs,
    )


################
# Bends
################


@gf.cell
def _bend_s(
    size: tuple[float, float] = (11.0, 1.8),
    cross_section: CrossSectionSpec = "xs_sc",
    **kwargs,
) -> gf.Component:
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
    direction: str = "ccw",
    cross_section: CrossSectionSpec = "xs_sc",
) -> gf.Component:
    return gf.components.bend_euler(
        radius=radius,
        angle=angle,
        p=p,
        with_arc_floorplan=with_arc_floorplan,
        npoints=npoints,
        direction=direction,
        cross_section=cross_section,
    )


@gf.cell
def wire_corner(
    cross_section: CrossSectionSpec = "xs_metal_routing",
) -> gf.Component:
    return gf.components.wire_corner(cross_section=cross_section)


def _float(x: Any) -> float:
    return float(x)


@gf.cell
def bend_sc(
    radius: float = _float(xs_sc.radius), angle: float = 90.0, **kwargs
) -> gf.Component:
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_sc"
    return _bend(
        radius=radius,
        angle=angle,
        **kwargs,
    )


@gf.cell
def bend_so(
    radius: float = _float(xs_so.radius), angle: float = 90.0, **kwargs
) -> gf.Component:
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_so"
    return _bend(
        radius=radius,
        angle=angle,
        **kwargs,
    )


@gf.cell
def bend_rc(
    radius: float = _float(xs_rc.radius), angle: float = 90.0, **kwargs
) -> gf.Component:
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_rc"
    return _bend(
        radius=radius,
        angle=angle,
        **kwargs,
    )


@gf.cell
def bend_ro(
    radius: float = _float(xs_ro.radius), angle: float = 90.0, **kwargs
) -> gf.Component:
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_ro"
    return _bend(
        radius=radius,
        angle=angle,
        **kwargs,
    )


@gf.cell
def bend_nc(
    radius: float = _float(xs_nc.radius), angle: float = 90.0, **kwargs
) -> gf.Component:
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_nc"
    return _bend(
        radius=radius,
        angle=angle,
        **kwargs,
    )


@gf.cell
def bend_no(
    radius: float = _float(xs_no.radius), angle: float = 90.0, **kwargs
) -> gf.Component:
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_no"
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
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_ro"
    return _taper(length=length, width1=width1, width2=width2, port=port, **kwargs)


@gf.cell
def taper_nc(
    length: float = 10.0,
    width1: float = 0.5,
    width2: float | None = None,
    port: gf.Port | None = None,
    **kwargs,
) -> gf.Component:
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_nc"
    return _taper(length=length, width1=width1, width2=width2, port=port, **kwargs)


@gf.cell
def taper_no(
    length: float = 10.0,
    width1: float = 0.5,
    width2: float | None = None,
    port: gf.Port | None = None,
    **kwargs,
) -> gf.Component:
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_no"
    return _taper(length=length, width1=width1, width2=width2, port=port, **kwargs)


@gf.cell
def _taper_cross_section(
    length: float = 10,
    cross_section1: str = "xs_rc_tip",
    cross_section2: str = "xs_rc",
    linear: bool = True,
    **kwargs,
) -> gf.Component:
    if linear:
        kwargs["npoints"] = 2
    return gf.components.taper_cross_section(
        length=length,
        cross_section1=cross_section1,
        cross_section2=cross_section2,
        linear=linear,
        **kwargs,
    ).flatten()


@gf.cell
def trans_sc_rc10() -> gf.Component:
    return _taper_cross_section(
        length=10,
        cross_section1="xs_rc_tip",
        cross_section2="xs_rc",
    )


@gf.cell
def trans_sc_rc20() -> gf.Component:
    return _taper_cross_section(
        length=20,
        cross_section1="xs_rc_tip",
        cross_section2="xs_rc",
    )


@gf.cell
def trans_sc_rc50() -> gf.Component:
    return _taper_cross_section(
        length=50,
        cross_section1="xs_rc_tip",
        cross_section2="xs_rc",
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
    return gf.components.mmi1x2(
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
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_rc"
    return _mmi1x2(
        length_mmi=32.7,
        gap_mmi=1.64,
        **kwargs,
    )


@gf.cell
def mmi2x2_rc(**kwargs) -> gf.Component:
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
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_sc"
    return _mmi1x2(
        length_mmi=31.8,
        gap_mmi=1.64,
        **kwargs,
    )


@gf.cell
def mmi2x2_sc(**kwargs) -> gf.Component:
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
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_ro"
    return _mmi1x2(
        length_mmi=40.8,
        gap_mmi=1.55,
        **kwargs,
    )


@gf.cell
def mmi2x2_ro(**kwargs) -> gf.Component:
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
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_so"
    return _mmi1x2(
        length_mmi=40.1,
        gap_mmi=1.55,
        **kwargs,
    )


@gf.cell
def mmi2x2_so(**kwargs) -> gf.Component:
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_so"
    return _mmi2x2(
        length_mmi=53.5,
        gap_mmi=0.53,
        **kwargs,
    )


################
# Nitride MMIs oband
################


@gf.cell
def mmi1x2_no(**kwargs) -> gf.Component:
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_no"
    return _mmi1x2(
        width_mmi=12.0,
        length_taper=50.0,
        width_taper=5.5,
        gap_mmi=0.4,
        length_mmi=42.0,
        **kwargs,
    )


@gf.cell
def mmi2x2_no(**kwargs) -> gf.Component:
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_no"
    return _mmi2x2(
        width_mmi=12.0,
        length_taper=50.0,
        width_taper=5.5,
        gap_mmi=0.4,
        length_mmi=126.0,
        **kwargs,
    )


################
# Nitride MMIs cband
################


@gf.cell
def mmi1x2_nc(**kwargs) -> gf.Component:
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_nc"
    return _mmi1x2(
        width_mmi=12.0,
        length_taper=50.0,
        width_taper=5.5,
        gap_mmi=0.4,
        length_mmi=64.7,
        **kwargs,
    )


@gf.cell
def mmi2x2_nc(**kwargs) -> gf.Component:
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_nc"
    return _mmi2x2(
        width_mmi=12.0,
        length_taper=50.0,
        width_taper=5.5,
        gap_mmi=0.4,
        length_mmi=232.0,
        **kwargs,
    )


##############################
# Evanescent couplers
##############################


@gf.cell
def _coupler_symmetric(
    bend: ComponentSpec = _bend_s,
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
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_ro"
    return _coupler(
        gap=gap,
        length=length,
        dx=dx,
        dy=dy,
        **kwargs,
    )


@gf.cell
def coupler_nc(
    gap: float = 0.4,
    length: float = 20.0,
    dx: float = 10.0,
    dy: float = 4.0,
    **kwargs,
) -> gf.Component:
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_nc"
    return _coupler(
        gap=gap,
        length=length,
        dx=dx,
        dy=dy,
        **kwargs,
    )


@gf.cell
def coupler_no(
    gap: float = 0.4,
    length: float = 20.0,
    dx: float = 10.0,
    dy: float = 4.0,
    **kwargs,
) -> gf.Component:
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_no"
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


@gf.cell
def _gc_rectangular(
    n_periods: int = 30,
    fill_factor: float = 0.5,
    length_taper: float = 350.0,
    fiber_angle: float = 10.0,
    layer_grating: LayerSpec = LAYER.GRA,
    layer_slab: LayerSpec = LAYER.WG,
    slab_offset: float = 0.0,
    period: float = 0.75,
    width_grating: float = 11.0,
    polarization: str = "te",
    wavelength: float = 1.55,
    taper: ComponentSpec = _taper,
    slab_xmin: float = -1.0,
    cross_section: CrossSectionSpec = "xs_sc",
) -> gf.Component:
    c = gf.components.grating_coupler_rectangular(
        n_periods=n_periods,
        fill_factor=fill_factor,
        length_taper=length_taper,
        fiber_angle=fiber_angle,
        layer_grating=layer_grating,
        layer_slab=layer_slab,
        slab_offset=slab_offset,
        period=period,
        width_grating=width_grating,
        polarization=polarization,
        wavelength=wavelength,
        taper=taper,
        slab_xmin=slab_xmin,
        cross_section=cross_section,
    ).flatten()
    return c


@gf.cell
def gc_rectangular_so() -> gf.Component:
    return _gc_rectangular(
        period=0.5,
        cross_section="xs_so",
        n_periods=80,
    )


@gf.cell
def gc_rectangular_ro() -> gf.Component:
    return _gc_rectangular(
        period=0.5,
        cross_section="xs_ro",
        n_periods=80,
    )


@gf.cell
def gc_rectangular_sc() -> gf.Component:
    return _gc_rectangular(
        period=0.63,
        cross_section="xs_sc",
        fiber_angle=10,
        n_periods=60,
    )


@gf.cell
def gc_rectangular_rc() -> gf.Component:
    return _gc_rectangular(
        period=0.5,
        cross_section="xs_rc",
        n_periods=60,
    )


@gf.cell
def gc_rectangular_nc() -> gf.Component:
    return _gc_rectangular(
        period=0.66,
        cross_section="xs_nc",
        length_taper=200,
        fiber_angle=20,
        layer_grating=LAYER.NITRIDE_ETCH,
        layer_slab=LAYER.NITRIDE,
        slab_offset=0,
    )


@gf.cell
def gc_rectangular_no() -> gf.Component:
    return _gc_rectangular(
        period=0.964,
        cross_section="xs_no",
        length_taper=200,
        fiber_angle=20,
        layer_grating=LAYER.NITRIDE_ETCH,
        layer_slab=LAYER.NITRIDE,
        slab_offset=0,
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
    p_start: float = 26.0,
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
    add_electrical_ports_bot: bool = True,
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
        add_electrical_ports_bot=add_electrical_ports_bot,
        min_length=0.01,
        extend_ports_straight_x=None,
    )


@gf.cell
def mzi_sc(
    delta_length: float = 10.0,
    length_y: float = 2.0,
    length_x: float = 0.1,
    add_electrical_ports_bot: bool = True,
    **kwargs,
) -> gf.Component:
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_sc"
    return _mzi(
        delta_length=delta_length,
        length_y=length_y,
        length_x=length_x,
        add_electrical_ports_bot=add_electrical_ports_bot,
        straight=straight_sc,
        bend=bend_sc,
        combiner=mmi1x2_sc,
        splitter=mmi1x2_sc,
        **kwargs,
    )


@gf.cell
def mzi_so(
    delta_length: float = 10.0,
    length_y: float = 2.0,
    length_x: float = 0.1,
    add_electrical_ports_bot: bool = True,
    **kwargs,
) -> gf.Component:
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_so"
    return _mzi(
        delta_length=delta_length,
        length_y=length_y,
        length_x=length_x,
        add_electrical_ports_bot=add_electrical_ports_bot,
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
    add_electrical_ports_bot: bool = True,
    **kwargs,
) -> gf.Component:
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_rc"
    return _mzi(
        delta_length=delta_length,
        length_y=length_y,
        length_x=length_x,
        add_electrical_ports_bot=add_electrical_ports_bot,
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


@gf.cell
def mzi_nc(
    delta_length: float = 10.0,
    length_y: float = 2.0,
    length_x: float = 0.1,
    add_electrical_ports_bot: bool = True,
    **kwargs,
) -> gf.Component:
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_nc"
    return _mzi(
        delta_length=delta_length,
        length_y=length_y,
        length_x=length_x,
        add_electrical_ports_bot=add_electrical_ports_bot,
        straight=straight_nc,
        bend=bend_nc,
        combiner=mmi1x2_nc,
        splitter=mmi1x2_nc,
        **kwargs,
    )


@gf.cell
def mzi_no(
    delta_length: float = 10.0,
    length_y: float = 2.0,
    length_x: float = 0.1,
    add_electrical_ports_bot: bool = True,
    **kwargs,
) -> gf.Component:
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_no"
    return _mzi(
        delta_length=delta_length,
        length_y=length_y,
        length_x=length_x,
        add_electrical_ports_bot=add_electrical_ports_bot,
        bend=bend_no,
        straight=straight_no,
        combiner=mmi1x2_no,
        splitter=mmi1x2_no,
        **kwargs,
    )


################
# Packaging
################


@gf.cell
def _pad(
    size: tuple[float, float] = (100.0, 100.0),
    layer: LayerSpec = LAYER.PAD,
    bbox_layers: None = None,
    bbox_offsets: None = None,
    port_inclusion: float = 0.0,
    port_orientation: None = None,
) -> gf.Component:
    return gf.components.pad(
        size=size,
        layer=layer,
        bbox_layers=bbox_layers,
        bbox_offsets=bbox_offsets,
        port_inclusion=port_inclusion,
        port_orientation=port_orientation,
    )


@gf.cell
def _rectangle(
    size: tuple[float, float] = (4.0, 2.0),
    layer: LayerSpec = LAYER.FLOORPLAN,
    centered: bool = False,
    port_type: str = "electrical",
    port_orientations: tuple[float, float, float, float] = (180.0, 90.0, 0.0, -90.0),
    round_corners_east_west: bool = False,
    round_corners_north_south: bool = False,
) -> gf.Component:
    return gf.components.rectangle(
        size=size,
        layer=layer,
        centered=centered,
        port_type=port_type,
        port_orientations=port_orientations,
        round_corners_east_west=round_corners_east_west,
        round_corners_north_south=round_corners_north_south,
    )


@gf.cell
def _grating_coupler_array(
    pitch: float = 127.0,
    n: int = 6,
    port_name: str = "o1",
    rotation: float = 0.0,
    with_loopback: bool = False,
    bend: ComponentSpec = _bend,
    grating_coupler_spacing: float = 0.0,
    grating_coupler: ComponentSpec = gc_rectangular_nc,
    cross_section: CrossSectionSpec = "xs_nc",
) -> gf.Component:
    return gf.components.grating_coupler_array(
        pitch=pitch,
        n=n,
        port_name=port_name,
        rotation=rotation,
        with_loopback=with_loopback,
        bend=bend,
        grating_coupler_spacing=grating_coupler_spacing,
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
    grating_coupler: ComponentSpec = gc_rectangular_nc,
    cross_section: CrossSectionSpec = "xs_nc",
) -> gf.Component:
    c = gf.Component()

    fp = c << _rectangle(size=size, layer=LAYER.FLOORPLAN, centered=True)

    # Add optical ports
    x0 = -4925 + 2.827
    y0 = 1650

    gca = _grating_coupler_array(
        n=ngratings,
        pitch=grating_pitch,
        with_loopback=True,
        rotation=90,
        grating_coupler=grating_coupler,
        cross_section=cross_section,
    )
    left = c << gca
    left.rotate(90)
    left.xmax = x0
    left.y = fp.y
    c.add_ports(left.ports, prefix="W")

    right = c << gca
    right.rotate(-90)
    right.xmax = -x0
    right.y = fp.y
    c.add_ports(right.ports, prefix="E")

    # Add electrical ports
    x0 = -4615
    y0 = 2200
    pad = _pad()

    for i in range(npads):
        pad_ref = c << pad
        pad_ref.xmin = x0 + i * pad_pitch
        pad_ref.ymin = y0
        c.add_port(
            name=f"N{i}",
            port=pad_ref.ports["e4"],
        )

    for i in range(npads):
        pad_ref = c << pad
        pad_ref.xmin = x0 + i * pad_pitch
        pad_ref.ymax = -y0
        c.add_port(
            name=f"S{i}",
            port=pad_ref.ports["e2"],
        )

    c.auto_rename_ports()
    return c


@gf.cell
def die_nc() -> gf.Component:
    return _die(
        grating_coupler=gc_rectangular_nc,
        cross_section="xs_nc",
    )


@gf.cell
def die_no() -> gf.Component:
    return _die(
        grating_coupler=gc_rectangular_no,
        cross_section="xs_no",
    )


@gf.cell
def die_sc() -> gf.Component:
    return _die(
        grating_coupler=gc_rectangular_sc,
        cross_section="xs_sc",
    )


@gf.cell
def die_so() -> gf.Component:
    return _die(
        grating_coupler=gc_rectangular_so,
        cross_section="xs_so",
    )


@gf.cell
def die_rc() -> gf.Component:
    return _die(
        grating_coupler=gc_rectangular_rc,
        cross_section="xs_rc",
    )


@gf.cell
def die_ro() -> gf.Component:
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
    """Returns Heater fixed cell."""
    heater = _import_gds("Heater.gds")
    heater.name = "heater"
    return heater


@gf.cell
def crossing_so() -> gf.Component:
    """Returns SOI220nm_1310nm_TE_STRIP_Waveguide_Crossing fixed cell."""
    c = _import_gds("SOI220nm_1310nm_TE_STRIP_Waveguide_Crossing.gds")
    c.name = "crossing_so"

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
    """Returns SOI220nm_1550nm_TE_RIB_Waveguide_Crossing fixed cell."""
    c = _import_gds("SOI220nm_1550nm_TE_RIB_Waveguide_Crossing.gds")
    c.name = "crossing_rc"
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
    """Returns SOI220nm_1550nm_TE_STRIP_Waveguide_Crossing fixed cell."""
    c = _import_gds("SOI220nm_1550nm_TE_STRIP_Waveguide_Crossing.gds")
    c.name = "crossing_sc"
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


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if not callable(func):
            continue
        if name in ["partial", "_import_gds"]:
            continue
        print(name, func())
