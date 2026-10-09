"""Remove layers from a Cornerstone Si220 component."""

import gdsfactory as gf
from cspdk.si220 import LAYER, cells


@gf.cell
def sample2_remove_layers() -> gf.Component:
    """Keep only the waveguide core of a rib straight, dropping its slab."""
    c = gf.Component()
    c << cells.straight(length=20, cross_section="rib_cband")
    c.flatten()
    assert len(c.layers) == 2  # WG core and SLAB
    c = c.remove_layers(layers=[LAYER.SLAB])
    assert len(c.layers) == 1  # WG core only
    return c


if __name__ == "__main__":
    sample2_remove_layers().show()
