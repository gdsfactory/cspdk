"""This module contains the building blocks for the CSPDK PDK."""

import gdsfactory as gf

from cspdk.si220.oband.config import PATH
from cspdk.si220.oband.tech import LAYER

################
# Imported from Cornerstone MPW SOI 220nm GDSII Template
################


@gf.cell
def heater() -> gf.Component:
    """Heater fixed cell."""
    return gf.import_gds(PATH.gds / "Heater.gds")


@gf.cell
def crossing_rib() -> gf.Component:
    """SOI220nm_1550nm_TE_RIB_Waveguide_Crossing fixed cell."""
    c = gf.import_gds(PATH.gds / "SOI220nm_1310nm_TE_RIB_Waveguide_Crossing.gds")
    xc = 404.24
    dx = 9.24 / 2
    x = xc - dx
    xl = xc - 2 * dx
    xr = xc
    yb = -dx
    yt = +dx
    width = 0.4

    c.add_port("o1", orientation=180, center=(xl, 0), width=width, layer=LAYER.WG)
    c.add_port("o2", orientation=90, center=(x, yt), width=width, layer=LAYER.WG)
    c.add_port("o3", orientation=0, center=(xr, 0), width=width, layer=LAYER.WG)
    c.add_port("o4", orientation=270, center=(x, yb), width=width, layer=LAYER.WG)
    return c


@gf.cell
def crossing() -> gf.Component:
    """SOI220nm_1310nm_TE_STRIP_Waveguide_Crossing fixed cell."""
    c = gf.import_gds(PATH.gds / "SOI220nm_1310nm_TE_STRIP_Waveguide_Crossing.gds")
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


if __name__ == "__main__":
    c = crossing_rib()
    c.pprint_ports()
    c.show()
