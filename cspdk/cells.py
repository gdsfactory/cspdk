from functools import partial

import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec, LayerSpec

from cspdk import tech
from cspdk.config import PATH
from cspdk.tech import LAYER

cell = gf.cell

################
# Adapted from gdsfactory generic PDK
################

################
# Waveguides
################

_straight = gf.components.straight


@cell
def straight_sc(cross_section="xs_sc", **kwargs):
    return _straight(cross_section=cross_section, **kwargs)


@cell
def straight_so(cross_section="xs_so", **kwargs):
    return _straight(cross_section=cross_section, **kwargs)


@cell
def straight_rc(cross_section="xs_rc", **kwargs):
    return _straight(cross_section=cross_section, **kwargs)


@cell
def straight_ro(cross_section="xs_ro", **kwargs):
    return _straight(cross_section=cross_section, **kwargs)


@cell
def straight_nc(cross_section="xs_nc", **kwargs):
    return _straight(cross_section=cross_section, **kwargs)


@cell
def straight_no(cross_section="xs_no", **kwargs):
    return _straight(cross_section=cross_section, **kwargs)


_bend_euler = gf.components.bend_euler


@cell
def bend_sc(cross_section="xs_sc", **kwargs):
    return _bend_euler(cross_section=cross_section, **kwargs)


@cell
def bend_so(cross_section="xs_so", **kwargs):
    return _bend_euler(cross_section=cross_section, **kwargs)


@cell
def bend_rc(cross_section="xs_rc", **kwargs):
    return _bend_euler(cross_section=cross_section, **kwargs)


@cell
def bend_ro(cross_section="xs_ro", **kwargs):
    return _bend_euler(cross_section=cross_section, **kwargs)


@cell
def bend_nc(cross_section="xs_nc", **kwargs):
    return _bend_euler(cross_section=cross_section, **kwargs)


@cell
def bend_no(cross_section="xs_no", **kwargs):
    return _bend_euler(cross_section=cross_section, **kwargs)


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
)
trans_sc_rc20 = partial(
    taper_cross_section,
    cross_section1="xs_rc_tip",
    cross_section2="xs_rc",
    length=20,
)
trans_sc_rc50 = partial(
    taper_cross_section,
    cross_section1="xs_rc_tip",
    cross_section2="xs_rc",
    length=50,
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
_mmi1x2_rc = partial(
    _mmi1x2,
    length_mmi=32.7,
    gap_mmi=1.64,
    cross_section="xs_rc",
)
_mmi2x2_rc = partial(
    _mmi2x2,
    length_mmi=44.8,
    gap_mmi=0.53,
    cross_section="xs_rc",
)


@cell
def mmi1x2_rc():
    return _mmi1x2_rc()


@cell
def mmi2x2_rc():
    return _mmi2x2_rc()


################
# MMIs strip cband
################
_mmi1x2_sc = partial(
    _mmi1x2,
    length_mmi=31.8,
    gap_mmi=1.64,
    cross_section="xs_sc",
)
_mmi2x2_sc = partial(
    _mmi2x2,
    length_mmi=42.5,
    gap_mmi=0.5,
    cross_section="xs_sc",
)


@gf.cell
def mmi1x2_sc():
    return _mmi1x2_sc()


@gf.cell
def mmi2x2_sc():
    return _mmi2x2_sc()


################
# MMIs rib oband
################
_mmi1x2_ro = partial(
    _mmi1x2,
    length_mmi=40.8,
    gap_mmi=1.55,
    cross_section="xs_ro",
)
_mmi2x2_ro = partial(
    _mmi2x2,
    length_mmi=55,
    gap_mmi=0.53,
    cross_section="xs_ro",
)


@gf.cell
def mmi1x2_ro():
    return _mmi1x2_ro()


@gf.cell
def mmi2x2_ro():
    return _mmi2x2_ro()


################
# MMIs strip oband
################
_mmi1x2_so = partial(
    _mmi1x2,
    length_mmi=40.1,
    gap_mmi=1.55,
    cross_section="xs_so",
)
_mmi2x2_so = partial(
    _mmi2x2,
    length_mmi=53.5,
    gap_mmi=0.53,
    cross_section="xs_so",
)


@gf.cell
def mmi1x2_so():
    return _mmi1x2_so()


@gf.cell
def mmi2x2_so():
    return _mmi2x2_so()


################
# Nitride MMIs oband
################
_mmi1x2_nitride_oband = partial(
    gf.components.mmi1x2,
    width_mmi=12,
    length_taper=50,
    width_taper=5.5,
    gap_mmi=0.4,
    cross_section="xs_no",
    length_mmi=42,
)
_mmi2x2_nitride_oband = partial(
    gf.components.mmi2x2,
    width_mmi=12,
    length_taper=50,
    width_taper=5.5,
    gap_mmi=0.4,
    cross_section="xs_no",
    length_mmi=126,
)


@gf.cell
def mmi1x2_no():
    return _mmi1x2_nitride_oband()


@gf.cell
def mmi2x2_no():
    return _mmi2x2_nitride_oband()


################
# Nitride MMIs cband
################

_mmi1x2_nitride_cband = partial(
    _mmi1x2_nitride_oband,
    length_taper=50,
    cross_section="xs_nc",
    length_mmi=64.7,
)
_mmi2x2_nitride_cband = partial(
    _mmi2x2_nitride_oband,
    length_taper=50,
    cross_section="xs_nc",
    length_mmi=232,
)


@gf.cell
def mmi1x2_nc():
    return _mmi1x2_nitride_cband()


@gf.cell
def mmi2x2_nc():
    return _mmi2x2_nitride_cband()


##############################
# Evanescent couplers
##############################


@cell
def coupler_sc(gap: float = 0.236, length: float = 20, cross_section="xs_sc", **kwargs):
    return gf.components.coupler(
        gap=gap,
        length=length,
        cross_section=cross_section,
        **kwargs,
    )


@cell
def coupler_so(gap: float = 0.236, length: float = 20, cross_section="xs_so", **kwargs):
    return gf.components.coupler(
        gap=gap,
        length=length,
        cross_section=cross_section,
        **kwargs,
    )


@cell
def coupler_rc(gap: float = 0.236, length: float = 20, cross_section="xs_rc", **kwargs):
    return gf.components.coupler(
        gap=gap,
        length=length,
        cross_section=cross_section,
        **kwargs,
    )


@cell
def coupler_ro(gap: float = 0.236, length: float = 20, cross_section="xs_ro", **kwargs):
    return gf.components.coupler(
        gap=gap,
        length=length,
        cross_section=cross_section,
        **kwargs,
    )


@cell
def coupler_nc(gap: float = 0.4, length: float = 20, cross_section="xs_nc", **kwargs):
    return gf.components.coupler(
        gap=gap,
        length=length,
        cross_section=cross_section,
        **kwargs,
    )


@cell
def coupler_no(gap: float = 0.4, length: float = 20, cross_section="xs_no", **kwargs):
    return gf.components.coupler(
        gap=gap,
        length=length,
        cross_section=cross_section,
        **kwargs,
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
        **kwargs,
    ).flatten()
    return c


_gc_rectangular_so = partial(
    gc_rectangular,
    period=0.5,
    cross_section="xs_so",
    n_periods=80,
)
_gc_rectangular_ro = partial(
    gc_rectangular,
    period=0.5,
    cross_section="xs_ro",
    n_periods=80,
)

_gc_rectangular_sc = partial(
    gc_rectangular,
    period=0.63,
    cross_section="xs_sc",
    fiber_angle=10,
    n_periods=60,
)
_gc_rectangular_rc = partial(
    gc_rectangular,
    period=0.5,
    cross_section="xs_rc",
    n_periods=60,
)

_gc_rectangular_nc = partial(
    gc_rectangular,
    period=0.66,
    cross_section="xs_nc",
    length_taper=200,
    fiber_angle=20,
    layer_grating=LAYER.NITRIDE_ETCH,
    layer_slab=LAYER.NITRIDE,
    slab_offset=0,
)
_gc_rectangular_no = partial(
    gc_rectangular,
    period=0.964,
    cross_section="xs_no",
    length_taper=200,
    fiber_angle=20,
    layer_grating=LAYER.NITRIDE_ETCH,
    layer_slab=LAYER.NITRIDE,
    slab_offset=0,
)


@cell
def gc_rectangular_so():
    return _gc_rectangular_so()


@cell
def gc_rectangular_ro():
    return _gc_rectangular_ro()


@cell
def gc_rectangular_sc():
    return _gc_rectangular_sc()


@cell
def gc_rectangular_rc():
    return _gc_rectangular_rc()


@cell
def gc_rectangular_nc():
    return _gc_rectangular_nc()


@cell
def gc_rectangular_no():
    return _gc_rectangular_no()


##############################
# grating couplers elliptical
##############################


@gf.cell
def gc_elliptical_sc(
    grating_line_width: float = 0.343,
    cross_section: CrossSectionSpec = "xs_sc",
    fiber_angle: float = 15,
    layer_trench: LayerSpec = LAYER.GRA,
    wavelength: float = 1.53,
) -> gf.Component:
    return gf.components.grating_coupler_elliptical_trenches(
        grating_line_width=grating_line_width,
        cross_section=cross_section,
        fiber_angle=fiber_angle,
        layer_trench=layer_trench,
        wavelength=wavelength,
    )


@gf.cell
def gc_elliptical_so(
    grating_line_width: float = 0.343,
    cross_section: CrossSectionSpec = "xs_so",
    fiber_angle: float = 15,
    layer_trench: LayerSpec = LAYER.GRA,
    wavelength: float = 1.31,
) -> gf.Component:
    return gf.components.grating_coupler_elliptical_trenches(
        grating_line_width=grating_line_width,
        cross_section=cross_section,
        fiber_angle=fiber_angle,
        layer_trench=layer_trench,
        wavelength=wavelength,
    )


################
# MZI
################

mzi = gf.components.mzi

mzi_sc = partial(
    mzi,
    straight=straight_sc,
    bend=bend_sc,
    cross_section="xs_sc",
    combiner=mmi1x2_sc,
    splitter=mmi1x2_sc,
)
mzi_so = partial(
    mzi,
    straight=straight_so,
    bend=bend_so,
    cross_section="xs_so",
    combiner=mmi1x2_so,
    splitter=mmi1x2_so,
)
mzi_rc = partial(
    mzi,
    straight=straight_rc,
    bend=bend_rc,
    cross_section="xs_rc",
    combiner=mmi1x2_rc,
    splitter=mmi1x2_rc,
)
mzi_ro = partial(
    mzi,
    straight=straight_ro,
    bend=bend_ro,
    cross_section="xs_ro",
    combiner=mmi1x2_ro,
    splitter=mmi1x2_ro,
)

mzi_nc = partial(
    mzi,
    straight=straight_nc,
    bend=bend_nc,
    cross_section=tech.xs_nc,
    combiner=mmi1x2_nc,
    splitter=mmi1x2_nc,
)
mzi_no = partial(
    mzi,
    bend=bend_no,
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
    # print(help(gf.components.grating_coupler_rectangular))
    # c = straight_sc(cross_section="xs_nc")
    # c = gc_elliptical_sc()
    c = coupler_nc()
    c.show()
