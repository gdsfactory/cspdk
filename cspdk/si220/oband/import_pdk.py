"""From a list of GDS files, generate a script to import the cells from a pdk."""

import gdsfactory as gf

if __name__ == "__main__":
    print(gf.write_cells.get_import_gdsript("gds", module="cspdk.si220.oband.cells"))
