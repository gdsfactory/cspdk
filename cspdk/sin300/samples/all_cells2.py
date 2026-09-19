"""Debug cells with port position issues in library."""

import gdsfactory as gf

from cspdk.sin300 import PDK

if __name__ == "__main__":
    PDK.activate()
    cell_name = "coupler"
    cell_name = "coupler_nc"
    cell_name = "coupler_no"
    c = gf.get_component(cell_name)
    c.show()
