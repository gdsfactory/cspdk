"""Build a custom cross-section from Cornerstone Si220 layers."""

import gdsfactory as gf
from cspdk.si220 import LAYER, tech


@gf.cell
def sample6_cross_section() -> gf.Component:
    """Strip waveguide with a heater track above it and slab rails on either side."""
    p = gf.path.straight(length=50)
    core = gf.Section(
        width=tech.TECH.width_cband,
        offset=0,
        layer=LAYER.WG,
        port_names=("o1", "o2"),
    )
    heater = gf.Section(width=2.5, offset=0, layer=LAYER.HEATER)
    rail_top = gf.Section(width=2, offset=4, layer=LAYER.SLAB)
    rail_bot = gf.Section(width=2, offset=-4, layer=LAYER.SLAB)
    xs = gf.CrossSection(sections=(core, heater, rail_top, rail_bot))
    return gf.path.extrude(p, cross_section=xs)


if __name__ == "__main__":
    sample6_cross_section().show()
