from functools import partial

import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec

from cspdk.si220.config import PATH
from cspdk.si220.tech import LAYER

################
# Straights
################


@gf.cell
def straight(
    length: float = 10.0,
    cross_section: CrossSectionSpec = "xs_sc",
) -> gf.Component:
    return gf.c.straight(length=length, cross_section=cross_section)  # type: ignore


straight_sc = partial(straight, cross_section="xs_sc")
straight_so = partial(straight, cross_section="xs_so")
straight_rc = partial(straight, cross_section="xs_rc")
straight_ro = partial(straight, cross_section="xs_ro")

################
# Bends
################


@gf.cell
def wire_corner() -> gf.Component:
    return gf.components.wire_corner(cross_section="metal_routing")  # type: ignore


@gf.cell
def bend_s(
    size: tuple[float, float] = (11.0, 1.8),
    cross_section: CrossSectionSpec = "xs_sc",
) -> gf.Component:
    return gf.components.bend_s(size=size, cross_section=cross_section)  # type: ignore


@gf.cell
def bend_euler(
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.5,
    width: float | None = None,
    cross_section: CrossSectionSpec = "xs_sc",
) -> gf.Component:
    return gf.components.bend_euler(  # type: ignore
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
    port: gf.Port | None = None,
    cross_section: CrossSectionSpec = "xs_sc",
) -> gf.Component:
    return gf.c.taper(  # type: ignore
        length=length,
        width1=width1,
        width2=width2,
        port=port,
        cross_section=cross_section,
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
    cross_section: CrossSectionSpec = "xs_sc",
) -> gf.Component:
    return gf.c.taper_strip_to_ridge(  # type: ignore
        length=length,
        width1=width1,
        width2=width2,
        w_slab1=w_slab1,
        w_slab2=w_slab2,
        cross_section=cross_section,
        layer_wg=LAYER.WG,
        layer_slab=LAYER.SLAB,
    )


trans_sc_rc10 = partial(taper_strip_to_ridge, length=10)
trans_sc_rc20 = partial(taper_strip_to_ridge, length=20)
trans_sc_rc50 = partial(taper_strip_to_ridge, length=50)


################
# MMIs
################


@gf.cell
def mmi1x2(
    width: float | None = None,
    width_taper=1.5,
    length_taper=20.0,
    length_mmi: float = 5.5,
    width_mmi=6.0,
    gap_mmi: float = 0.25,
    cross_section: CrossSectionSpec = "xs_sc",
) -> gf.Component:
    return gf.c.mmi1x2(
        width=width,
        width_taper=width_taper,
        length_taper=length_taper,
        length_mmi=length_mmi,
        width_mmi=width_mmi,
        gap_mmi=gap_mmi,
        taper=taper,  # type: ignore
        straight=straight,  # type: ignore
        cross_section=cross_section,
    )


mmi1x2_sc = partial(mmi1x2, length_mmi=31.8, gap_mmi=1.64, cross_section="xs_sc")
mmi1x2_so = partial(mmi1x2, length_mmi=40.1, gap_mmi=1.55, cross_section="xs_so")
mmi1x2_rc = partial(mmi1x2, length_mmi=32.7, gap_mmi=1.64, cross_section="xs_rc")
mmi1x2_ro = partial(mmi1x2, length_mmi=40.8, gap_mmi=1.55, cross_section="xs_ro")


@gf.cell
def mmi2x2(
    width: float | None = None,
    width_taper: float = 1.5,
    length_taper: float = 20.0,
    length_mmi: float = 5.5,
    width_mmi: float = 6.0,
    gap_mmi: float = 0.25,
    cross_section: CrossSectionSpec = "xs_sc",
) -> gf.Component:
    return gf.c.mmi2x2(  # type: ignore
        width=width,
        width_taper=width_taper,
        length_taper=length_taper,
        length_mmi=length_mmi,
        width_mmi=width_mmi,
        gap_mmi=gap_mmi,
        taper=taper,  # type: ignore
        straight=straight,  # type: ignore
        cross_section=cross_section,
    )


mmi2x2_sc = partial(mmi2x2, length_mmi=42.5, gap_mmi=0.5, cross_section="xs_sc")
mmi2x2_so = partial(mmi2x2, length_mmi=53.5, gap_mmi=0.53, cross_section="xs_so")
mmi2x2_rc = partial(mmi2x2, length_mmi=44.8, gap_mmi=0.53, cross_section="xs_rc")
mmi2x2_ro = partial(mmi2x2, length_mmi=55.0, gap_mmi=0.53, cross_section="xs_ro")


##############################
# Evanescent couplers
##############################


@gf.cell
def coupler_straight(
    length: float = 10.0,
    gap: float = 0.27,
    cross_section: CrossSectionSpec = "xs_sc",
) -> gf.Component:
    return gf.c.coupler_straight(  # type: ignore
        length=length,
        gap=gap,
        cross_section=cross_section,
    )


@gf.cell
def coupler_symmetric(
    gap: float = 0.234,
    dy: float = 4.0,
    dx: float = 10.0,
    cross_section: CrossSectionSpec = "xs_sc",
) -> gf.Component:
    return gf.c.coupler_symmetric(
        bend="bend_s",  # type: ignore
        gap=gap,
        dy=dy,
        dx=dx,
        cross_section=cross_section,
    )


@gf.cell
def coupler(
    gap: float = 0.234,
    length: float = 20.0,
    dy: float = 4.0,
    dx: float = 10.0,
    cross_section: CrossSectionSpec = "xs_sc",
) -> gf.Component:
    return gf.c.coupler(  # type: ignore
        gap=gap,
        length=length,
        dy=dy,
        dx=dx,
        cross_section=cross_section,
    )


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
    period=0.315 * 2,
    n_periods: int = 30,
    length_taper: float = 350.0,
    wavelength: float = 1.55,
    cross_section="xs_sc",
) -> gf.Component:
    return gf.c.grating_coupler_rectangular(  # type: ignore
        n_periods=n_periods,
        period=period,
        fill_factor=0.5,
        width_grating=11.0,
        length_taper=length_taper,
        polarization="te",
        wavelength=wavelength,
        taper=taper,  # type: ignore
        layer_slab=LAYER.WG,
        layer_grating=LAYER.GRA,
        fiber_angle=10.0,
        slab_xmin=-1.0,
        slab_offset=0.0,
        cross_section=cross_section,
    )


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
    wavelength: float = 1.53,
    grating_line_width=0.315,
    cross_section="xs_sc",
) -> gf.Component:
    return gf.c.grating_coupler_elliptical_trenches(  # type: ignore
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


@gf.cell
def mzi(
    delta_length: float = 10.0,
    straight="straight",
    splitter="mmi1x2",
    combiner="mmi2x2",
    cross_section: CrossSectionSpec = "xs_sc",
) -> gf.Component:
    return gf.c.mzi(  # type: ignore
        delta_length=delta_length,
        length_y=1.0,
        length_x=0.1,
        bend="bend_sc",
        straight_y=None,
        straight_x_top=None,
        straight_x_bot=None,
        with_splitter=True,
        port_e1_splitter="o2",
        port_e0_splitter="o3",
        port_e1_combiner="o3",
        port_e0_combiner="o4",
        nbends=2,
        cross_section=cross_section,
        cross_section_x_top=None,
        cross_section_x_bot=None,
        mirror_bot=False,
        add_optical_ports_arms=False,
        min_length=10e-3,
        auto_rename_ports=True,
        straight=straight,
        splitter=splitter,
        combiner=combiner,
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


@gf.cell
def pad() -> gf.Component:
    return gf.c.pad(layer="PAD", size=(100.0, 100.0))  # type: ignore


@gf.cell
def rectangle() -> gf.Component:
    return gf.c.rectangle(layer=LAYER.FLOORPLAN)  # type: ignore


@gf.cell
def grating_coupler_array(
    pitch: float = 127.0,
    n: int = 6,
    cross_section="xs_sc",
) -> gf.Component:
    return gf.c.grating_coupler_array(
        grating_coupler=grating_coupler_elliptical,  # type: ignore
        pitch=pitch,
        n=n,
        with_loopback=False,
        rotation=-90,
        straight_to_grating_spacing=10.0,
        port_name="o1",
        centered=True,
        cross_section=cross_section,
    )


@gf.cell
def die(
    cross_section="xs_sc",
) -> gf.Component:
    return gf.c.die_with_pads(  # type: ignore
        cross_section=cross_section,
        edge_to_grating_distance=150.0,
        edge_to_pad_distance=150.0,
        grating_coupler="grating_coupler_rectangular_sc",
        grating_pitch=250.0,
        layer_floorplan=LAYER.FLOORPLAN,
        ngratings=14,
        npads=31,
        pad="pad",
        pad_pitch=300.0,
        size=(11470.0, 4900.0),
    )


die_sc = partial(
    die,
    grating_coupler="grating_coupler_rectangular_sc",
    cross_section="xs_sc",
)
die_so = partial(
    die,
    grating_coupler="grating_coupler_rectangular_so",
    cross_section="xs_so",
)
die_rc = partial(
    die,
    grating_coupler="grating_coupler_rectangular_rc",
    cross_section="xs_rc",
)
die_ro = partial(
    die,
    grating_coupler="grating_coupler_rectangular_ro",
    cross_section="xs_ro",
)


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


@gf.cell
def array(
    component="pad",
    spacing: tuple[float, float] = (150.0, 150.0),
    columns: int = 6,
    rows: int = 1,
    add_ports: bool = True,
    size=None,
    centered: bool = False,
) -> gf.Component:
    return gf.c.array(  # type: ignore
        component=component,
        spacing=spacing,
        columns=columns,
        rows=rows,
        size=size,
        centered=centered,
        add_ports=add_ports,
    )


if __name__ == "__main__":
    c = straight()
    print(c.function_name)
    # c.get_netlist()
    c.show()
