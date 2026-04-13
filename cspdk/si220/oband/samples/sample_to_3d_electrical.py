"""Write a GDS with all cells."""

import gdsfactory as gf

from cspdk.si220.oband import PDK, cells

if __name__ == "__main__":
    PDK.activate()

    c0 = cells.pad()
    c1 = cells.via_stack_heater_mtop()
    c = gf.grid([c0, c1])
    c.show()
    s = c.to_3d()
    s.show()
