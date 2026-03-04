"""This script generates a reticle with all the cells in the library."""

import gdsfactory as gf

from cspdk.si220.cband import LAYER, PDK

skip = {"all_cells", "die"}


@gf.cell
def all_cells(raise_if_error: bool = True) -> gf.Component:
    """Returns a sample reticle with all cells."""
    c = gf.Component()
    cell_functions = [
        cell for cell_name, cell in PDK.cells.items() if cell_name not in skip
    ]

    cells = []

    for cell in cell_functions:
        try:
            cells.append(cell())
        except (ValueError, KeyError) as e:
            print(f"Error creating cell {cell}: {e}")
            if raise_if_error:
                raise e

    cell_matrix = c << gf.pack(cells)[0]
    floorplan = c << gf.c.rectangle(
        size=(cell_matrix.xsize + 20, cell_matrix.ysize + 20),
        layer=LAYER.FLOORPLAN,
    )
    floorplan.center = cell_matrix.center
    return c


if __name__ == "__main__":
    PDK.activate()
    # c = gf.get_component("mrm")
    c = all_cells()
    c.show()
