"""Write a GDS with all cells."""

import gdsfactory as gf

from cspdk.si220 import PDK, cells

if __name__ == "__main__":
    PDK.activate()

    c1 = cells.straight(cross_section="strip_cband", length=5)
    c2 = cells.straight(cross_section="rib_cband", length=5)

    c = gf.grid([c1, c2])
    c.show()
    s = c.to_3d()
    s.show()
