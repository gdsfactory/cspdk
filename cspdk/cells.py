from functools import partial

import gdsfactory as gf

from cspdk.config import PATH
from cspdk.tech import xs_rc, xs_ro, xs_sc, xs_so

################
# Adapted from gdsfactory generic PDK
################
_mmi1x2 = partial(gf.components.mmi1x2, width_mmi=6, length_taper=20, width_taper=1.5)
_mmi2x2 = partial(gf.components.mmi2x2, width_mmi=6, length_taper=20, width_taper=1.5)
################
# rib cband
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
# strip cband
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
# rib oband
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
# strip oband
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
# Imported from PDK
################

gdsdir = PATH.gds
import_gds = partial(gf.import_gds, gdsdir=gdsdir)


@gf.cell
def CORNERSTONE_MPW_SOI_220nm_GDSII_Template() -> gf.Component:
    """Returns CORNERSTONE_MPW_SOI_220nm_GDSII_Template fixed cell."""
    return import_gds("CORNERSTONE MPW SOI 220nm GDSII Template.gds")


@gf.cell
def Cell0_SOI220_Full_1550nm_Packaging_Template() -> gf.Component:
    """Returns Cell0_SOI220_Full_1550nm_Packaging_Template fixed cell."""
    return import_gds("Cell0_SOI220_Full_1550nm_Packaging_Template.gds")


@gf.cell
def Cell0_SOI_Full_Institution_Name() -> gf.Component:
    """Returns Cell0_SOI_Full_Institution_Name fixed cell."""
    return import_gds("Cell0_SOI_Full_Institution_Name.gds")


@gf.cell
def Cell0_SOI_Half_Institution_Name() -> gf.Component:
    """Returns Cell0_SOI_Half_Institution_Name fixed cell."""
    return import_gds("Cell0_SOI_Half_Institution_Name.gds")


@gf.cell
def Flip_Chip_Bonding_Example() -> gf.Component:
    """Returns Flip_Chip_Bonding_Example fixed cell."""
    return import_gds("Flip_Chip_Bonding_Example.gds")


@gf.cell
def Heater() -> gf.Component:
    """Returns Heater fixed cell."""
    return import_gds("Heater.gds")


@gf.cell
def Layer_Designations() -> gf.Component:
    """Returns Layer_Designations fixed cell."""
    return import_gds("Layer_Designations.gds")


@gf.cell
def SOI220nm_1310nm_TE_RIB_2x1_MMI() -> gf.Component:
    """Returns SOI220nm_1310nm_TE_RIB_2x1_MMI fixed cell."""
    return import_gds("SOI220nm_1310nm_TE_RIB_2x1_MMI.gds")


@gf.cell
def SOI220nm_1310nm_TE_RIB_2x2_MMI() -> gf.Component:
    """Returns SOI220nm_1310nm_TE_RIB_2x2_MMI fixed cell."""
    return import_gds("SOI220nm_1310nm_TE_RIB_2x2_MMI.gds")


@gf.cell
def SOI220nm_1310nm_TE_RIB_90_Degree_Bend() -> gf.Component:
    """Returns SOI220nm_1310nm_TE_RIB_90_Degree_Bend fixed cell."""
    return import_gds("SOI220nm_1310nm_TE_RIB_90_Degree_Bend.gds")


@gf.cell
def SOI220nm_1310nm_TE_RIB_Grating_Coupler() -> gf.Component:
    """Returns SOI220nm_1310nm_TE_RIB_Grating_Coupler fixed cell."""
    return import_gds("SOI220nm_1310nm_TE_RIB_Grating_Coupler.gds")


@gf.cell
def SOI220nm_1310nm_TE_RIB_Waveguide() -> gf.Component:
    """Returns SOI220nm_1310nm_TE_RIB_Waveguide fixed cell."""
    return import_gds("SOI220nm_1310nm_TE_RIB_Waveguide.gds")


@gf.cell
def SOI220nm_1310nm_TE_RIB_Waveguide_Crossing() -> gf.Component:
    """Returns SOI220nm_1310nm_TE_RIB_Waveguide_Crossing fixed cell."""
    return import_gds("SOI220nm_1310nm_TE_RIB_Waveguide_Crossing.gds")


@gf.cell
def SOI220nm_1310nm_TE_STRIP_2x1_MMI() -> gf.Component:
    """Returns SOI220nm_1310nm_TE_STRIP_2x1_MMI fixed cell."""
    return import_gds("SOI220nm_1310nm_TE_STRIP_2x1_MMI.gds")


@gf.cell
def SOI220nm_1310nm_TE_STRIP_2x2_MMI() -> gf.Component:
    """Returns SOI220nm_1310nm_TE_STRIP_2x2_MMI fixed cell."""
    return import_gds("SOI220nm_1310nm_TE_STRIP_2x2_MMI.gds")


@gf.cell
def SOI220nm_1310nm_TE_STRIP_90_Degree_Bend() -> gf.Component:
    """Returns SOI220nm_1310nm_TE_STRIP_90_Degree_Bend fixed cell."""
    return import_gds("SOI220nm_1310nm_TE_STRIP_90_Degree_Bend.gds")


@gf.cell
def SOI220nm_1310nm_TE_STRIP_Grating_Coupler() -> gf.Component:
    """Returns SOI220nm_1310nm_TE_STRIP_Grating_Coupler fixed cell."""
    return import_gds("SOI220nm_1310nm_TE_STRIP_Grating_Coupler.gds")


@gf.cell
def SOI220nm_1310nm_TE_STRIP_Waveguide() -> gf.Component:
    """Returns SOI220nm_1310nm_TE_STRIP_Waveguide fixed cell."""
    return import_gds("SOI220nm_1310nm_TE_STRIP_Waveguide.gds")


@gf.cell
def SOI220nm_1310nm_TE_STRIP_Waveguide_Crossing() -> gf.Component:
    """Returns SOI220nm_1310nm_TE_STRIP_Waveguide_Crossing fixed cell."""
    return import_gds("SOI220nm_1310nm_TE_STRIP_Waveguide_Crossing.gds")


@gf.cell
def SOI220nm_1550nm_TE_RIB_2x1_MMI() -> gf.Component:
    """Returns SOI220nm_1550nm_TE_RIB_2x1_MMI fixed cell."""
    return import_gds("SOI220nm_1550nm_TE_RIB_2x1_MMI.gds")


@gf.cell
def SOI220nm_1550nm_TE_RIB_2x2_MMI() -> gf.Component:
    """Returns SOI220nm_1550nm_TE_RIB_2x2_MMI fixed cell."""
    return import_gds("SOI220nm_1550nm_TE_RIB_2x2_MMI.gds")


@gf.cell
def SOI220nm_1550nm_TE_RIB_90_Degree_Bend() -> gf.Component:
    """Returns SOI220nm_1550nm_TE_RIB_90_Degree_Bend fixed cell."""
    return import_gds("SOI220nm_1550nm_TE_RIB_90_Degree_Bend.gds")


@gf.cell
def SOI220nm_1550nm_TE_RIB_Grating_Coupler() -> gf.Component:
    """Returns SOI220nm_1550nm_TE_RIB_Grating_Coupler fixed cell."""
    return import_gds("SOI220nm_1550nm_TE_RIB_Grating_Coupler.gds")


@gf.cell
def SOI220nm_1550nm_TE_RIB_Waveguide() -> gf.Component:
    """Returns SOI220nm_1550nm_TE_RIB_Waveguide fixed cell."""
    return import_gds("SOI220nm_1550nm_TE_RIB_Waveguide.gds")


@gf.cell
def SOI220nm_1550nm_TE_RIB_Waveguide_Crossing() -> gf.Component:
    """Returns SOI220nm_1550nm_TE_RIB_Waveguide_Crossing fixed cell."""
    return import_gds("SOI220nm_1550nm_TE_RIB_Waveguide_Crossing.gds")


@gf.cell
def SOI220nm_1550nm_TE_STRIP_2x1_MMI() -> gf.Component:
    """Returns SOI220nm_1550nm_TE_STRIP_2x1_MMI fixed cell."""
    return import_gds("SOI220nm_1550nm_TE_STRIP_2x1_MMI.gds")


@gf.cell
def SOI220nm_1550nm_TE_STRIP_2x2_MMI() -> gf.Component:
    """Returns SOI220nm_1550nm_TE_STRIP_2x2_MMI fixed cell."""
    return import_gds("SOI220nm_1550nm_TE_STRIP_2x2_MMI.gds")


@gf.cell
def SOI220nm_1550nm_TE_STRIP_90_Degree_Bend() -> gf.Component:
    """Returns SOI220nm_1550nm_TE_STRIP_90_Degree_Bend fixed cell."""
    return import_gds("SOI220nm_1550nm_TE_STRIP_90_Degree_Bend.gds")


@gf.cell
def SOI220nm_1550nm_TE_STRIP_Waveguide() -> gf.Component:
    """Returns SOI220nm_1550nm_TE_STRIP_Waveguide fixed cell."""
    return import_gds("SOI220nm_1550nm_TE_STRIP_Waveguide.gds")


@gf.cell
def SOI220nm_1550nm_TE_STRIP_Waveguide_Crossing() -> gf.Component:
    """Returns SOI220nm_1550nm_TE_STRIP_Waveguide_Crossing fixed cell."""
    return import_gds("SOI220nm_1550nm_TE_STRIP_Waveguide_Crossing.gds")


if __name__ == "__main__":
    c = gf.Component()
    run = c << mmi1x2_rc()
    ref = c << SOI220nm_1550nm_TE_RIB_2x1_MMI()
    ref.mirror()
    ref.center = (0, 0)
    run.center = (0, 0)
    c.show()
