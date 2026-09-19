"""This module contains the building blocks for the CSPDK PDK."""

import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec

from cspdk.si220._schematic import crossing_rib_schematic, crossing_schematic
from cspdk.si220.config import PATH
from cspdk.si220.tech import LAYER, get_band

################
# Imported from Cornerstone MPW SOI 220nm GDSII Template
################


@gf.cell(tags=["fixed"])
def heater() -> gf.Component:
    """Heater fixed cell."""
    return gf.import_gds(PATH.gds / "Heater.gds")


@gf.cell(tags=["fixed"], schematic_function=crossing_rib_schematic)
def crossing_rib(
    cross_section: CrossSectionSpec = "rib_cband",
) -> gf.Component:
    """Return the fixed rib-waveguide crossing for the selected band.

    Args:
        cross_section: rib cross-section selecting the operating band.
    """
    band = get_band(cross_section)
    wavelength = "1310" if band == "oband" else "1550"
    c = gf.import_gds(
        PATH.gds / f"SOI220nm_{wavelength}nm_TE_RIB_Waveguide_Crossing.gds"
    )
    xc = 404.24
    dx = 9.24 / 2
    x = xc - dx
    xl = xc - 2 * dx
    xr = xc
    yb = -dx
    yt = +dx
    width = gf.get_cross_section(cross_section).width

    c.add_port("o1", orientation=180, center=(xl, 0), width=width, layer=LAYER.WG)
    c.add_port("o2", orientation=90, center=(x, yt), width=width, layer=LAYER.WG)
    c.add_port("o3", orientation=0, center=(xr, 0), width=width, layer=LAYER.WG)
    c.add_port("o4", orientation=270, center=(x, yb), width=width, layer=LAYER.WG)
    return c


@gf.cell(tags=["fixed"], schematic_function=crossing_schematic)
def crossing(
    cross_section: CrossSectionSpec = "strip_cband",
) -> gf.Component:
    """Return the fixed strip-waveguide crossing for the selected band.

    Args:
        cross_section: strip cross-section selecting the operating band.
    """
    band = get_band(cross_section)
    wavelength = "1310" if band == "oband" else "1550"
    c = gf.import_gds(
        PATH.gds / f"SOI220nm_{wavelength}nm_TE_STRIP_Waveguide_Crossing.gds"
    )
    xc = 493.47 if band == "oband" else 494.24
    yc = 0 if band == "oband" else 800
    dx = (8.47 if band == "oband" else 9.24) / 2
    x = xc - dx
    xl = xc - 2 * dx
    xr = xc
    yb = yc - dx
    yt = yc + dx
    width = gf.get_cross_section(cross_section).width

    c.add_port("o1", orientation=180, center=(xl, yc), width=width, layer=LAYER.WG)
    c.add_port("o2", orientation=90, center=(x, yt), width=width, layer=LAYER.WG)
    c.add_port("o3", orientation=0, center=(xr, yc), width=width, layer=LAYER.WG)
    c.add_port("o4", orientation=270, center=(x, yb), width=width, layer=LAYER.WG)
    return c
