"""Arrange Cornerstone Si220 building blocks on a grid."""

import gdsfactory as gf
from cspdk.si220 import cells


@gf.cell
def sample3_grid() -> gf.Component:
    """Show the main Si220 building blocks side by side."""
    components = [
        cells.mmi1x2(),
        cells.mmi2x2(),
        cells.coupler(),
        cells.ring_single(),
        cells.grating_coupler_elliptical(),
        cells.crossing(),
    ]
    return gf.grid(components, shape=(2, 3), spacing=(50, 50))


if __name__ == "__main__":
    sample3_grid().show()
