"""Write GDS with hello world on Cornerstone Si220 layers."""

import gdsfactory as gf
from cspdk.si220 import LAYER, cells


@gf.cell
def sample0_hello_world() -> gf.Component:
    """Hello world: a waveguide-layer square next to text on the pad layer."""
    c = gf.Component()
    ref1 = c.add_ref(gf.components.rectangle(size=(10, 10), layer=LAYER.WG))
    ref2 = c.add_ref(cells.text_rectangular(text="Hello", size=2, layer=LAYER.PAD))
    ref3 = c.add_ref(cells.text_rectangular(text="world", size=2, layer=LAYER.PAD))
    ref1.xmax = ref2.xmin - 5
    ref3.xmin = ref2.xmax + 2
    ref3.rotate(90)
    return c


if __name__ == "__main__":
    sample0_hello_world().show()
