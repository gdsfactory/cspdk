from functools import partial

import gdsfactory as gf

from cspdk.config import PATH
from cspdk.tech import LAYER, xs_nc, xs_no, xs_rc, xs_rc_tip, xs_ro, xs_sc, xs_so

################
# Adapted from gdsfactory generic PDK
################

################
# Waveguides
################
straight = gf.components.straight
straight_sc = partial(straight, cross_section=xs_sc, info={"model": "straight_sc"})
straight_so = partial(straight, cross_section=xs_so, info={"model": "straight_so"})
straight_rc = partial(straight, cross_section=xs_rc, info={"model": "straight_rc"})
straight_ro = partial(straight, cross_section=xs_ro, info={"model": "straight_ro"})
straight_nc = partial(straight, cross_section=xs_nc, info={"model": "straight_nc"})
straight_no = partial(straight, cross_section=xs_no, info={"model": "straight_no"})


bend_euler = gf.components.bend_euler
bend_sc = partial(bend_euler, cross_section=xs_sc, info={"model": "bend_sc"})
bend_so = partial(bend_euler, cross_section=xs_so, info={"model": "bend_so"})
bend_rc = partial(bend_euler, cross_section=xs_rc, info={"model": "bend_rc"})
bend_ro = partial(bend_euler, cross_section=xs_ro, info={"model": "bend_ro"})
bend_nc = partial(bend_euler, cross_section=xs_nc, info={"model": "bend_nc"})
bend_no = partial(bend_euler, cross_section=xs_no, info={"model": "bend_no"})

################
# Transitions
################
trans_sc_rc10 = partial(
    gf.c.taper_cross_section_linear,
    cross_section1=xs_rc_tip,
    cross_section2=xs_rc,
    length=10,
)
trans_sc_rc20 = partial(
    gf.c.taper_cross_section_linear,
    cross_section1=xs_rc_tip,
    cross_section2=xs_rc,
    length=20,
)
trans_sc_rc50 = partial(
    gf.c.taper_cross_section_linear,
    cross_section1=xs_rc_tip,
    cross_section2=xs_rc,
    length=50,
)

################
# MMIs
################
_mmi1x2 = partial(gf.components.mmi1x2, width_mmi=6, length_taper=20, width_taper=1.5)
_mmi2x2 = partial(gf.components.mmi2x2, width_mmi=6, length_taper=20, width_taper=1.5)

################
# MMIs rib cband
################
mmi1x2_rc = partial(
    _mmi1x2,
    length_mmi=32.7,
    gap_mmi=1.64,
    cross_section=xs_rc,
)
mmi2x2_rc = partial(
    _mmi2x2,
    length_mmi=44.8,
    gap_mmi=0.53,
    cross_section=xs_rc,
)
################
# MMIs strip cband
################
mmi1x2_sc = partial(
    _mmi1x2,
    length_mmi=31.8,
    gap_mmi=1.64,
    cross_section=xs_sc,
)
mmi2x2_sc = partial(
    _mmi2x2,
    length_mmi=42.5,
    gap_mmi=0.5,
    cross_section=xs_sc,
)
################
# MMIs rib oband
################
mmi1x2_ro = partial(
    _mmi1x2,
    length_mmi=40.8,
    gap_mmi=1.55,
    cross_section=xs_ro,
)
mmi2x2_ro = partial(
    _mmi2x2,
    length_mmi=55,
    gap_mmi=0.53,
    cross_section=xs_ro,
)
################
# MMIs strip oband
################
mmi1x2_so = partial(
    _mmi1x2,
    length_mmi=40.1,
    gap_mmi=1.55,
    cross_section=xs_so,
)
mmi2x2_so = partial(
    _mmi2x2,
    length_mmi=53.5,
    gap_mmi=0.53,
    cross_section=xs_so,
)

################
# Nitride MMIs oband
################
_mmi1x2_nitride_oband = partial(
    gf.components.mmi1x2,
    width_mmi=12,
    length_taper=50,
    width_taper=5.5,
    gap=0.4,
    cross_section=xs_no,
)
_mmi2x2_nitride_oband = partial(
    gf.components.mmi2x2,
    width_mmi=12,
    length_taper=50,
    width_taper=5.5,
    gap=0.4,
    cross_section=xs_no,
)

_mm1x2_nitride_cband = partial(
    _mmi1x2_nitride_oband, length_taper=50, cross_section=xs_nc
)
_mm2x2_nitride_cband = partial(
    _mmi2x2_nitride_oband, length_taper=50, cross_section=xs_nc
)

mmi1x2_no = partial(
    _mmi1x2_nitride_oband,
    length_mmi=42,
    cross_section=xs_no,
)
mmi2x2_no = partial(
    _mmi2x2_nitride_oband,
    length_mmi=126,
    cross_section=xs_no,
)
mmi1x2_nc = partial(
    _mmi1x2_nitride_oband,
    length_mmi=64.7,
    cross_section=xs_nc,
)
mmi2x2_nc = partial(
    _mmi2x2_nitride_oband,
    length_mmi=232,
    cross_section=xs_nc,
)

##############################
# grating couplers Rectangular
##############################

_gc_rectangular = partial(
    gf.components.grating_coupler_rectangular,
    n_periods=30,
    fill_factor=0.5,
    length_taper=350,
    fiber_angle=10,
    layer_grating=LAYER.GRA,
    layer_slab=LAYER.WG,
    slab_offset=0,
)

gc_rectangular_so = partial(
    _gc_rectangular,
    period=0.5,
    cross_section=xs_so,
    n_periods=80,
)
gc_rectangular_ro = partial(
    _gc_rectangular,
    period=0.5,
    cross_section=xs_ro,
    n_periods=80,
)

gc_rectangular_sc = partial(
    _gc_rectangular,
    period=0.63,
    cross_section=xs_sc,
    fiber_angle=10,
    n_periods=60,
)
gc_rectangular_rc = partial(
    _gc_rectangular,
    period=0.5,
    cross_section=xs_rc,
    n_periods=60,
)

gc_rectangular_nc = partial(
    _gc_rectangular,
    period=0.66,
    cross_section=xs_nc,
    length_taper=200,
    fiber_angle=20,
    layer_grating=LAYER.NITRIDE_ETCH,
    layer_slab=LAYER.NITRIDE,
    slab_offset=0,
)
gc_rectangular_no = partial(
    _gc_rectangular,
    period=0.964,
    cross_section=xs_no,
    length_taper=200,
    fiber_angle=20,
    layer_grating=LAYER.NITRIDE_ETCH,
    layer_slab=LAYER.NITRIDE,
    slab_offset=0,
)
################
# MZI
################

mzi_sc = partial(
    gf.c.mzi,
    straight=straight_sc,
    cross_section=xs_sc,
    combiner=mmi1x2_sc,
    splitter=mmi1x2_sc,
)
mzi_so = partial(
    gf.c.mzi,
    straight=straight_so,
    cross_section=xs_so,
    combiner=mmi1x2_sc,
    splitter=mmi1x2_sc,
)
mzi_rc = partial(
    gf.c.mzi,
    straight=straight_rc,
    cross_section=xs_rc,
    combiner=mmi1x2_rc,
    splitter=mmi1x2_rc,
)
mzi_ro = partial(
    gf.c.mzi,
    straight=straight_ro,
    cross_section=xs_ro,
    combiner=mmi1x2_ro,
    splitter=mmi1x2_ro,
)

mzi_nc = partial(
    gf.c.mzi,
    straight=straight_nc,
    cross_section=xs_nc,
    combiner=mmi1x2_nc,
    splitter=mmi1x2_nc,
)
mzi_no = partial(
    gf.c.mzi,
    straight=straight_no,
    cross_section=xs_no,
    combiner=mmi1x2_no,
    splitter=mmi1x2_no,
)


################
# Packaging
################
pad = partial(gf.components.pad, layer=LAYER.PAD)


@gf.cell
def die_nc(
    size=(11470, 4900),
    grating_coupler=gc_rectangular_nc,
    pad=pad,
    ngratings=14,
    npads=31,
    grating_pitch=250,
    pad_pitch=300,
    cross_section=xs_nc,
) -> gf.Component:
    """Returns cell for die.

    Args:
        size: die size.
        grating_coupler: grating coupler function.
        pad: pad function.
        ngratings: number of grating couplers.
        npads: number of pads.
        grating_pitch: grating pitch.
        pad_pitch: pad pitch.
        cross_section: cross_section.
    """
    c = gf.Component()

    fp = c << gf.c.rectangle(size=size, layer=LAYER.FLOORPLAN, centered=True)

    # Add optical ports
    x0 = -4925 + 2.827
    y0 = 1650
    grating_coupler = grating_coupler()

    grating_coupler_array = gf.c.grating_coupler_array(
        grating_coupler=grating_coupler,
        n=ngratings,
        pitch=grating_pitch,
        with_loopback=True,
        rotation=90,
        cross_section=cross_section,
    )
    left = c << grating_coupler_array
    left.rotate(90)
    left.xmax = x0
    left.y = fp.y
    c.add_ports(left.ports, prefix="W")

    right = c << grating_coupler_array
    right.rotate(-90)
    right.xmax = -x0
    right.y = fp.y
    c.add_ports(right.ports, prefix="E")

    # Add electrical ports
    x0 = -4615
    y0 = 2200
    pad = pad()

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


die_no = partial(die_nc, grating_coupler=gc_rectangular_no, cross_section=xs_no)
die_sc = partial(die_nc, grating_coupler=gc_rectangular_sc, cross_section=xs_sc)
die_so = partial(die_nc, grating_coupler=gc_rectangular_so, cross_section=xs_so)
die_rc = partial(die_nc, grating_coupler=gc_rectangular_rc, cross_section=xs_rc)
die_ro = partial(die_nc, grating_coupler=gc_rectangular_ro, cross_section=xs_ro)


################
# Imported from Cornerstone MPW SOI 220nm GDSII Template
################
gdsdir = PATH.gds
import_gds = partial(gf.import_gds, gdsdir=gdsdir)


@gf.cell
def heater() -> gf.Component:
    """Returns Heater fixed cell."""
    return import_gds("Heater.gds")


@gf.cell
def crossing_so() -> gf.Component:
    """Returns SOI220nm_1310nm_TE_STRIP_Waveguide_Crossing fixed cell."""
    return import_gds("SOI220nm_1310nm_TE_STRIP_Waveguide_Crossing.gds")


@gf.cell
def crossing_rc() -> gf.Component:
    """Returns SOI220nm_1550nm_TE_RIB_Waveguide_Crossing fixed cell."""
    return import_gds("SOI220nm_1550nm_TE_RIB_Waveguide_Crossing.gds")


@gf.cell
def crossing_sc() -> gf.Component:
    """Returns SOI220nm_1550nm_TE_STRIP_Waveguide_Crossing fixed cell."""
    return import_gds("SOI220nm_1550nm_TE_STRIP_Waveguide_Crossing.gds")


if __name__ == "__main__":
    c = straight_sc()
    # c = gf.Component()
    # w = c << straight_rc()
    # t = c << trans_sc_rc10()
    # t.connect("o2", w.ports["o1"])
    # ref = c << SOI220nm_1550nm_TE_RIB_2x1_MMI()
    # ref.mirror()
    # ref.center = (0, 0)
    # run.center = (0, 0)
    # c = gc_rectangular_ro()
    # c = packaging_MPW_SOI220()
    # c = gc_rectangular_no()
    # c = packaging_MPW_nc()
    # c = die_nc()
    c.show(show_ports=True)
