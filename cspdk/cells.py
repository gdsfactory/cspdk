from functools import partial, wraps

import gdsfactory as gf

from cspdk.config import PATH
from cspdk.tech import LAYER

################
# Adapted from gdsfactory generic PDK
################

################
# Waveguides
################

straight = gf.components.straight
straight_sc = partial(straight, cross_section="xs_sc", info={"model": "straight_sc"})
straight_so = partial(straight, cross_section="xs_so", info={"model": "straight_so"})
straight_rc = partial(straight, cross_section="xs_rc", info={"model": "straight_rc"})
straight_ro = partial(straight, cross_section="xs_ro", info={"model": "straight_ro"})
straight_nc = partial(straight, cross_section="xs_nc", info={"model": "straight_nc"})
straight_no = partial(straight, cross_section="xs_no", info={"model": "straight_no"})


@wraps(gf.components.bend_euler)
def bend_euler(info=None, **kwargs):
    c = gf.components.bend_euler(**kwargs)
    if info is not None:
        c.info.update(info)
    return c


bend_sc = partial(bend_euler, cross_section="xs_sc", info={"model": "bend_sc"})
bend_so = partial(bend_euler, cross_section="xs_so", info={"model": "bend_so"})
bend_rc = partial(bend_euler, cross_section="xs_rc", info={"model": "bend_rc"})
bend_ro = partial(bend_euler, cross_section="xs_ro", info={"model": "bend_ro"})
bend_nc = partial(bend_euler, cross_section="xs_nc", info={"model": "bend_nc"})
bend_no = partial(bend_euler, cross_section="xs_no", info={"model": "bend_no"})

################
# Transitions
################

taper = gf.components.taper


@gf.cell
def taper_cross_section(
    cross_section1="xs_rc_tip",
    cross_section2="xs_rc",
    length: float = 10,
    npoints: int = 2,
    linear: bool = True,
    **kwargs,
) -> gf.Component:
    return gf.components.taper_cross_section(
        cross_section1=cross_section1,
        cross_section2=cross_section2,
        length=length,
        npoints=npoints,
        linear=linear,
        **kwargs,
    ).flatten()


trans_sc_rc10 = partial(
    taper_cross_section,
    cross_section1="xs_rc_tip",
    cross_section2="xs_rc",
    length=10,
    info={"model": "trans_sc_rc10"},
)
trans_sc_rc20 = partial(
    taper_cross_section,
    cross_section1="xs_rc_tip",
    cross_section2="xs_rc",
    length=20,
    info={"model": "trans_sc_rc20"},
)
trans_sc_rc50 = partial(
    taper_cross_section,
    cross_section1="xs_rc_tip",
    cross_section2="xs_rc",
    length=50,
    info={"model": "trans_sc_rc50"},
)

################
# MMIs
################

mmi1x2 = gf.components.mmi1x2
mmi2x2 = gf.components.mmi2x2

_mmi1x2 = partial(
    gf.components.mmi1x2,
    width_mmi=6,
    length_taper=20,
    width_taper=1.5,
)

_mmi2x2 = partial(
    gf.components.mmi2x2,
    width_mmi=6,
    length_taper=20,
    width_taper=1.5,
)

################
# MMIs rib cband
################
mmi1x2_rc = partial(
    _mmi1x2,
    length_mmi=32.7,
    gap_mmi=1.64,
    cross_section="xs_rc",
    info={"model": "mmi1x2_rc"},
)
mmi2x2_rc = partial(
    _mmi2x2,
    length_mmi=44.8,
    gap_mmi=0.53,
    cross_section="xs_rc",
    info={"model": "mmi2x2_rc"},
)
################
# MMIs strip cband
################
mmi1x2_sc = partial(
    _mmi1x2,
    length_mmi=31.8,
    gap_mmi=1.64,
    cross_section="xs_sc",
    info={"model": "mmi1x2_sc"},
)
mmi2x2_sc = partial(
    _mmi2x2,
    length_mmi=42.5,
    gap_mmi=0.5,
    cross_section="xs_sc",
    info={"model": "mmi2x2_sc"},
)
################
# MMIs rib oband
################
mmi1x2_ro = partial(
    _mmi1x2,
    length_mmi=40.8,
    gap_mmi=1.55,
    cross_section="xs_ro",
    info={"model": "mmi2x2_ro"},
)
mmi2x2_ro = partial(
    _mmi2x2,
    length_mmi=55,
    gap_mmi=0.53,
    cross_section="xs_ro",
    info={"model": "mmi2x2_ro"},
)
################
# MMIs strip oband
################
mmi1x2_so = partial(
    _mmi1x2,
    length_mmi=40.1,
    gap_mmi=1.55,
    cross_section="xs_so",
    info={"model": "mmi1x2_so"},
)
mmi2x2_so = partial(
    _mmi2x2,
    length_mmi=53.5,
    gap_mmi=0.53,
    cross_section="xs_so",
    info={"model": "mmi2x2_so"},
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
    cross_section="xs_no",
)
_mmi2x2_nitride_oband = partial(
    gf.components.mmi2x2,
    width_mmi=12,
    length_taper=50,
    width_taper=5.5,
    gap=0.4,
    cross_section="xs_no",
)

mmi1x2_no = partial(
    _mmi1x2_nitride_oband,
    length_mmi=42,
    cross_section="xs_no",
    info={"model": "mmi1x2_no"},
)
mmi2x2_no = partial(
    _mmi2x2_nitride_oband,
    length_mmi=126,
    cross_section="xs_no",
    info={"model": "mmi2x2_no"},
)

################
# Nitride MMIs cband
################

_mmi1x2_nitride_cband = partial(
    _mmi1x2_nitride_oband,
    length_taper=50,
    cross_section="xs_nc",
)
_mmi2x2_nitride_cband = partial(
    _mmi2x2_nitride_oband,
    length_taper=50,
    cross_section="xs_nc",
)

mmi1x2_nc = partial(
    _mmi1x2_nitride_cband,
    length_mmi=64.7,
    cross_section="xs_nc",
    info={"model": "mmi1x2_nc"},
)
mmi2x2_nc = partial(
    _mmi2x2_nitride_cband,
    length_mmi=232,
    cross_section="xs_nc",
    info={"model": "mmi2x2_nc"},
)

##############################
# grating couplers Rectangular
##############################


@gf.cell
def gc_rectangular(
    n_periods=30,
    fill_factor=0.5,
    length_taper=350,
    fiber_angle=10,
    layer_grating=LAYER.GRA,
    layer_slab=LAYER.WG,
    slab_offset=0,
    info=None,
    **kwargs,
) -> gf.Component:
    c = gf.components.grating_coupler_rectangular(
        n_periods=n_periods,
        fill_factor=fill_factor,
        length_taper=length_taper,
        fiber_angle=fiber_angle,
        layer_grating=layer_grating,
        layer_slab=layer_slab,
        slab_offset=slab_offset,
        info=info,
        **kwargs,
    ).flatten()
    return c


gc_rectangular_so = partial(
    gc_rectangular,
    period=0.5,
    cross_section="xs_so",
    n_periods=80,
    info={"model": "gc_rectangular_so"},
)
gc_rectangular_ro = partial(
    gc_rectangular,
    period=0.5,
    cross_section="xs_ro",
    n_periods=80,
    info={"model": "gc_rectangular_ro"},
)

gc_rectangular_sc = partial(
    gc_rectangular,
    period=0.63,
    cross_section="xs_sc",
    fiber_angle=10,
    n_periods=60,
    info={"model": "gc_rectangular_sc"},
)
gc_rectangular_rc = partial(
    gc_rectangular,
    period=0.5,
    cross_section="xs_rc",
    n_periods=60,
    info={"model": "gc_rectangular_rc"},
)

gc_rectangular_nc = partial(
    gc_rectangular,
    period=0.66,
    cross_section="xs_nc",
    length_taper=200,
    fiber_angle=20,
    layer_grating=LAYER.NITRIDE_ETCH,
    layer_slab=LAYER.NITRIDE,
    slab_offset=0,
    info={"model": "gc_rectangular_nc"},
)
gc_rectangular_no = partial(
    gc_rectangular,
    period=0.964,
    cross_section="xs_no",
    length_taper=200,
    fiber_angle=20,
    layer_grating=LAYER.NITRIDE_ETCH,
    layer_slab=LAYER.NITRIDE,
    slab_offset=0,
    info={"model": "gc_rectangular_no"},
)

################
# MZI
################

mzi = gf.components.mzi

mzi_sc = partial(
    mzi,
    straight=straight_sc,
    cross_section="xs_sc",
    combiner=mmi1x2_sc,
    splitter=mmi1x2_sc,
)
mzi_so = partial(
    mzi,
    straight=straight_so,
    cross_section="xs_so",
    combiner=mmi1x2_so,
    splitter=mmi1x2_so,
)
mzi_rc = partial(
    mzi,
    straight=straight_rc,
    cross_section="xs_rc",
    combiner=mmi1x2_rc,
    splitter=mmi1x2_rc,
)
mzi_ro = partial(
    mzi,
    straight=straight_ro,
    cross_section="xs_ro",
    combiner=mmi1x2_ro,
    splitter=mmi1x2_ro,
)

mzi_nc = partial(
    mzi,
    straight=straight_nc,
    cross_section="xs_nc",
    combiner=mmi1x2_nc,
    splitter=mmi1x2_nc,
)
mzi_no = partial(
    mzi,
    straight=straight_no,
    cross_section="xs_no",
    combiner=mmi1x2_no,
    splitter=mmi1x2_no,
)


################
# Packaging
################

pad = partial(
    gf.components.pad,
    layer=LAYER.PAD,
)

rectangle = partial(gf.components.rectangle, layer=LAYER.FLOORPLAN)
grating_coupler_array = partial(
    gf.components.grating_coupler_array,
    cross_section="xs_nc",
    grating_coupler=gc_rectangular_nc,
)


@gf.cell
def die_nc(
    size=(11470, 4900),
    grating_coupler=gc_rectangular_nc,
    pad=pad,
    ngratings=14,
    npads=31,
    grating_pitch=250,
    pad_pitch=300,
    cross_section="xs_nc",
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

    fp = c << rectangle(size=size, layer=LAYER.FLOORPLAN, centered=True)

    # Add optical ports
    x0 = -4925 + 2.827
    y0 = 1650

    gca = grating_coupler_array(
        grating_coupler=grating_coupler(),
        n=ngratings,
        pitch=grating_pitch,
        with_loopback=True,
        rotation=90,
        cross_section=cross_section,  # type: ignore
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


die_no = partial(
    die_nc,
    grating_coupler=gc_rectangular_no,
    cross_section="xs_no",
)
die_sc = partial(
    die_nc,
    grating_coupler=gc_rectangular_sc,
    cross_section="xs_sc",
)
die_so = partial(
    die_nc,
    grating_coupler=gc_rectangular_so,
    cross_section="xs_so",
)
die_rc = partial(
    die_nc,
    grating_coupler=gc_rectangular_rc,
    cross_section="xs_rc",
)
die_ro = partial(
    die_nc,
    grating_coupler=gc_rectangular_ro,
    cross_section="xs_ro",
)


################
# Imported from Cornerstone MPW SOI 220nm GDSII Template
################
gdsdir = PATH.gds
_import_gds = partial(gf.import_gds, gdsdir=gdsdir)


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
    # for name, func in list(globals().items()):
    #    if not callable(func):
    #        continue
    #    if name in ['partial', '_import_gds']:
    #        continue
    #    print(name, func())
    print(help(gf.components.grating_coupler_rectangular))
