"""This script generates a reticle with all the cells in the library."""

import gdsfactory as gf

from cspdk.si220.cband import PDK

skip = {}


@gf.cell
def all_cells() -> gf.Component:
    """Returns a sample reticle with all cells."""
    c = gf.Component()
    cell_functions = [
        cell for cell_name, cell in PDK.cells.items() if cell_name not in skip
    ]
    _ = c << gf.pack(cell_functions)[0]
    return c


if __name__ == "__main__":
    PDK.activate()
    # c = gf.get_component("grating_coupler_elliptical")
    # s = c.to_3d()
    # s.show()
    c = all_cells()
    c.show()
