from functools import partial

import gdsfactory as gf
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from cspdk.si220.config import PATH
from cspdk.si220.tech import LAYER

################
# Straights
################

straight = partial(gf.components.straight, cross_section="xs_sc")
straight_sc = partial(straight, cross_section="xs_sc")
straight_so = partial(straight, cross_section="xs_so")
straight_rc = partial(straight, cross_section="xs_rc")
straight_ro = partial(straight, cross_section="xs_ro")

################
# Bends
################

bend_s = partial(gf.components.bend_s, size=(11.0, 1.8), cross_section="xs_sc")
wire_corner = partial(gf.components.wire_corner, cross_section="metal_routing")
bend_euler = partial(gf.components.bend_euler, cross_section="xs_sc")
bend_sc = partial(bend_euler, cross_section="xs_sc")
bend_so = partial(bend_euler, cross_section="xs_so")
bend_rc = partial(bend_euler, cross_section="xs_rc")
bend_ro = partial(bend_euler, cross_section="xs_ro")

################
# Transitions
################

taper = partial(
    gf.components.taper, cross_section="xs_sc", length=10.0, width1=0.5, width2=None
)
taper_sc = partial(
    gf.components.taper, cross_section="xs_so", length=10.0, width1=0.5, width2=None
)
taper_so = partial(
    gf.components.taper, cross_section="xs_so", length=10.0, width1=0.5, width2=None
)
taper_rc = partial(
    gf.components.taper, cross_section="xs_rc", length=10.0, width1=0.5, width2=None
)
taper_ro = partial(
    gf.components.taper, cross_section="xs_ro", length=10.0, width1=0.5, width2=None
)

taper_strip_to_ridge = trans_sc_rc10 = partial(
    gf.c.taper_strip_to_ridge,
    length=10,
    w_slab1=0.2,
    w_slab2=10.45,
    cross_section="xs_sc",
    layer_wg=LAYER.WG,
    layer_slab=LAYER.SLAB,
)

trans_sc_rc20 = partial(trans_sc_rc10, length=20)
trans_sc_rc50 = partial(trans_sc_rc10, length=50)


################
# MMIs
################

mmi1x2 = partial(
    gf.components.mmi1x2,
    cross_section="xs_sc",
    width_mmi=6.0,
    width_taper=1.5,
    length_taper=20.0,
)
mmi2x2 = partial(
    gf.components.mmi2x2,
    cross_section="xs_sc",
    width_mmi=6.0,
    width_taper=1.5,
    length_taper=20.0,
)

mmi1x2_sc = partial(mmi1x2, length_mmi=31.8, gap_mmi=1.64, cross_section="xs_sc")
mmi2x2_sc = partial(mmi2x2, length_mmi=42.5, gap_mmi=0.5, cross_section="xs_sc")
mmi1x2_so = partial(mmi1x2, length_mmi=40.1, gap_mmi=1.55, cross_section="xs_so")
mmi2x2_so = partial(mmi2x2, length_mmi=53.5, gap_mmi=0.53, cross_section="xs_so")
mmi1x2_rc = partial(mmi1x2, length_mmi=32.7, gap_mmi=1.64, cross_section="xs_rc")
mmi2x2_rc = partial(mmi2x2, length_mmi=44.8, gap_mmi=0.53, cross_section="xs_rc")
mmi1x2_ro = partial(mmi1x2, length_mmi=40.8, gap_mmi=1.55, cross_section="xs_ro")
mmi2x2_ro = partial(mmi2x2, length_mmi=55.0, gap_mmi=0.53, cross_section="xs_ro")


##############################
# Evanescent couplers
##############################

coupler = partial(
    gf.c.coupler, gap=0.234, length=20.0, dx=10.0, dy=4.0, cross_section="xs_sc"
)
coupler_sc = partial(coupler, cross_section="xs_sc")
coupler_so = partial(coupler, cross_section="xs_so")
coupler_rc = partial(coupler, cross_section="xs_rc", dx=15, dy=4.0)
coupler_ro = partial(coupler, cross_section="xs_ro", dx=15, dy=4.0)


##############################
# grating couplers Rectangular
##############################

grating_coupler_rectangular = partial(
    gf.components.grating_coupler_rectangular,
    n_periods=30,
    fill_factor=0.5,
    length_taper=350.0,
    fiber_angle=10.0,
    layer_grating=LAYER.GRA,
    layer_slab=LAYER.WG,
    slab_offset=0.0,
    period=0.315 * 2,
    width_grating=11.0,
    polarization="te",
    wavelength=1.55,
    slab_xmin=-1.0,
    cross_section="xs_sc",
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

grating_coupler_elliptical_trenches = partial(
    gf.components.grating_coupler_elliptical_trenches,
    polarization="te",
    taper_length=16.6,
    taper_angle=30.0,
    trenches_extra_angle=9.0,
    wavelength=1.53,
    fiber_angle=15.0,
    grating_line_width=0.315,
    neff=2.638,
    ncladding=1.443,
    layer_trench=LAYER.GRA,
    p_start=26,
    n_periods=30,
    end_straight_length=0.2,
    cross_section="xs_sc",
)

grating_coupler_elliptical_trenches_sc = partial(
    grating_coupler_elliptical_trenches,
    grating_line_width=0.315,
    wavelength=1.55,
    cross_section="xs_sc",
)

grating_coupler_elliptical_trenches_so = partial(
    grating_coupler_elliptical_trenches,
    grating_line_width=0.250,
    wavelength=1.31,
    cross_section="xs_so",
)


################
# MZI
################

mzi_sc = partial(
    gf.components.mzi,
    delta_length=10.0,
    length_y=2.0,
    length_x=0.1,
    bend=bend_sc,
    straight=straight_sc,
    splitter=mmi1x2_sc,
    combiner=mmi2x2_sc,
    port_e1_splitter="o2",
    port_e0_splitter="o3",
    port_e1_combiner="o3",
    port_e0_combiner="o4",
    cross_section="xs_sc",
)

mzi_so = partial(
    gf.components.mzi,
    delta_length=10.0,
    length_y=2.0,
    length_x=0.1,
    bend=bend_sc,
    straight=straight_so,
    splitter=mmi1x2_so,
    combiner=mmi2x2_so,
    port_e1_splitter="o2",
    port_e0_splitter="o3",
    port_e1_combiner="o2",
    port_e0_combiner="o4",
    cross_section="xs_so",
)

mzi_rc = partial(
    gf.components.mzi,
    combiner=mmi1x2_rc,
    splitter=mmi1x2_rc,
    cross_section="xs_rc",
)

mzi_ro = partial(
    gf.components.mzi,
    splitter=mmi1x2_ro,
    combiner=mmi1x2_ro,
    cross_section="xs_ro",
)


################
# Packaging
################


pad = partial(gf.c.pad, layer="PAD", size=(100.0, 100.0))
rectangle = partial(gf.components.rectangle, layer=LAYER.FLOORPLAN)
grating_coupler_array = partial(
    gf.components.grating_coupler_array,
    pitch=127,
    n=6,
    port_name="o1",
    rotation=-90,
    with_loopback=False,
    grating_coupler="grating_coupler_rectangular_sc",
    cross_section="xs_sc",
)


@gf.cell
def _die(
    size: tuple[float, float] = (11470.0, 4900.0),
    ngratings: int = 14,
    npads: int = 31,
    grating_pitch: float = 250.0,
    pad_pitch: float = 300.0,
    grating_coupler: ComponentSpec = "grating_coupler_rectangular_sc",
    cross_section: CrossSectionSpec = "xs_sc",
    pad: ComponentSpec = "pad",
) -> gf.Component:
    """A die with grating couplers and pads.

    Args:
        size: the size of the die, in um.
        ngratings: the number of grating couplers.
        npads: the number of pads.
        grating_pitch: the pitch of the grating couplers, in um.
        pad_pitch: the pitch of the pads, in um.
        grating_coupler: the grating coupler component.
        cross_section: the cross section.
        pad: the pad component.
    """
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
    left.dy = fp.dy
    c.add_ports(left.ports, prefix="W")

    right = c << gca
    right.drotate(+90)
    right.dxmax = -x0
    right.dy = fp.dy
    c.add_ports(right.ports, prefix="E")

    # Add electrical ports
    x0 = -4615
    y0 = 2200
    pad = gf.get_component(pad)

    for i in range(npads):
        pad_ref = c << pad
        pad_ref.dxmin = x0 + i * pad_pitch
        pad_ref.dymin = y0
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


die_sc = partial(
    _die, grating_coupler="grating_coupler_rectangular_sc", cross_section="xs_sc"
)
die_so = partial(
    _die, grating_coupler="grating_coupler_rectangular_so", cross_section="xs_so"
)
die_rc = partial(
    _die, grating_coupler="grating_coupler_rectangular_rc", cross_section="xs_rc"
)
die_ro = partial(
    _die, grating_coupler="grating_coupler_rectangular_ro", cross_section="xs_ro"
)


################
# Imported from Cornerstone MPW SOI 220nm GDSII Template
################
_import_gds = gf.import_gds


@gf.cell
def heater() -> gf.Component:
    """Heater fixed cell."""
    heater = _import_gds(PATH.gds / "Heater.gds")
    heater.name = "heater"
    return heater


@gf.cell
def crossing_so() -> gf.Component:
    """SOI220nm_1310nm_TE_STRIP_Waveguide_Crossing fixed cell."""
    c = _import_gds(PATH.gds / "SOI220nm_1310nm_TE_STRIP_Waveguide_Crossing.gds")

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
    c = _import_gds(PATH.gds / "SOI220nm_1550nm_TE_RIB_Waveguide_Crossing.gds")
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
    c = _import_gds(PATH.gds / "SOI220nm_1550nm_TE_STRIP_Waveguide_Crossing.gds")
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
    c = coupler_ro()
    c.show()
