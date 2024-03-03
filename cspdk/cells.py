from functools import partial

import gdsfactory as gf

from cspdk.config import PATH
from cspdk.tech import LAYER, xs_rc, xs_ro, xs_sc, xs_so

################
# Adapted from gdsfactory generic PDK
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

##############################
# grating couplers Rectangular
##############################

_gc_rectangular = partial(
    gf.components.grating_coupler_rectangular,
    fill_factor=0.5,
    layer_slab=None,
    length_taper=350,
)

gc_rectangular_so = partial(
    _gc_rectangular,
    period=0.5,
    cross_section=xs_so,
)
gc_rectangular_ro = partial(
    _gc_rectangular,
    period=0.5,
    cross_section=xs_ro,
)

gc_rectangular_sc = partial(
    _gc_rectangular,
    period=0.63,
    cross_section=xs_sc,
)
gc_rectangular_rc = partial(
    _gc_rectangular,
    period=0.5,
    cross_section=xs_rc,
)

################
# Imported from Cornerstone MPW SOI 220nm GDSII Template
################
gdsdir = PATH.gds
import_gds = partial(gf.import_gds, gdsdir=gdsdir)


@gf.cell
def packaging_MPW_SOI220() -> gf.Component:
    """Returns CORNERSTONE_MPW_SOI_220nm_GDSII_Template fixed cell."""
    c = gf.Component()
    _ = c << import_gds("Cell0_SOI220_Full_1550nm_Packaging_Template.gds")

    # Add optical ports
    x0 = -4925 + 2.827
    y0 = 1650
    pitch = 250
    ngratings = 14
    for i in range(ngratings):
        c.add_port(
            name=f"W{i}",
            center=(x0, y0 - i * pitch),
            width=0.45,
            orientation=0,
            layer=LAYER.WG,
        )

    for i in range(ngratings):
        c.add_port(
            name=f"E{i}",
            center=(-x0, y0 - i * pitch),
            width=0.45,
            orientation=180,
            layer=LAYER.WG,
        )

    # Add electrical ports
    x0 = -4615
    y0 = 2200
    pitch = 300
    npads = 31
    for i in range(npads):
        c.add_port(
            name=f"N{i}",
            center=(x0 + i * pitch, y0),
            width=150,
            orientation=270,
            layer=LAYER.PAD,
        )

    for i in range(npads):
        c.add_port(
            name=f"S{i}",
            center=(x0 + i * pitch, -y0),
            width=150,
            orientation=90,
            layer=LAYER.PAD,
        )
    return c


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
    # ref = c << SOI220nm_1550nm_TE_RIB_2x1_MMI()
    # ref.mirror()
    # ref.center = (0, 0)
    # run.center = (0, 0)
    # c = gc_rectangular_ro()
    c = packaging_MPW_SOI220()
    c.show(show_ports=True)
