"""This script generates a reticle with all the cells in the library."""

import gdsfactory as gf
from cspdk.si220.cband import PDK

skip = {"all_cells"}


@gf.cell
def all_cells() -> gf.Component:
    """Returns a sample reticle with all cells."""
    c = gf.Component()
    cell_functions = [
        cell for cell_name, cell in PDK.cells.items() if cell_name not in skip
    ]
    _ = c << gf.pack(cell_functions)[0]
    return c
