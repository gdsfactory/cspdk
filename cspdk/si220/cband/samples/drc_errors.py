"""Write GDS with sample errors."""

from __future__ import annotations

import gdsfactory as gf
import numpy as np
from gdsfactory.component import Component
from gdsfactory.typings import Float2, Layer

from cspdk.si220.cband import LAYER

layer = LAYER.WG
layer1 = LAYER.WG


@gf.cell
def width_min(size: Float2 = (0.1, 0.1)) -> Component:
    """Minimum width for a waveguide."""
    return gf.components.rectangle(size=size, layer=layer)


@gf.cell
def area_min() -> Component:
    """Minimum area for a waveguide."""
    size = (0.2, 0.2)
    return gf.components.rectangle(size=size, layer=layer)


@gf.cell
def gap_min(gap: float = 0.1) -> Component:
    """Minimum gap between two waveguides."""
    c = gf.Component()
    r1 = c << gf.components.rectangle(size=(1, 1), layer=layer)
    r2 = c << gf.components.rectangle(size=(1, 1), layer=layer)
    r1.xmax = 0
    r2.xmin = gap
    return c


@gf.cell
def separation(
    gap: float = 0.1, layer1: Layer = (47, 0), layer2: Layer = (41, 0)
) -> Component:
    """Minimum separation between two layers."""
    c = gf.Component()
    r1 = c << gf.components.rectangle(size=(1, 1), layer=layer1)
    r2 = c << gf.components.rectangle(size=(1, 1), layer=layer2)
    r1.xmax = 0
    r2.xmin = gap
    return c


@gf.cell
def enclosing(
    enclosing: float = 0.1, layer1: Layer = (40, 0), layer2: Layer = (41, 0)
) -> Component:
    """Layer1 must be enclosed by layer2 by value.

    checks if layer1 encloses (is bigger than) layer2 by value
    """
    w1 = 1
    w2 = w1 + enclosing
    c = gf.Component()
    _ = c << gf.components.rectangle(size=(w1, w1), layer=layer1, centered=True)
    r2 = c << gf.components.rectangle(size=(w2, w2), layer=layer2, centered=True)
    r2.dmovex(0.5)
    return c


@gf.cell
def snapping_error(gap: float = 1e-3) -> Component:
    """Snapping error."""
    c = gf.Component()
    r1 = c << gf.components.rectangle(size=(1, 1), layer=layer)
    r2 = c << gf.components.rectangle(size=(1, 1), layer=layer)
    r1.xmax = 0
    r2.xmin = gap
    return c


@gf.cell
def errors() -> Component:
    """Write GDS with sample errors."""
    components = [width_min(), gap_min(), separation(), enclosing()]
    components += [gap_min(spacing) for spacing in np.linspace(0.1, 0.2, 5)]
    c = gf.pack(components, spacing=1.5)
    c = gf.add_padding_container(c[0], layers=(LAYER.FLOORPLAN,), default=5)
    return c


if __name__ == "__main__":
    # c = width_min()
    # c.write_gds("wmin.gds")
    # c = gap_min()
    # c.write_gds("gmin.gds")
    # c = snapping_error()
    # c.write_gds("snap.gds")

    c = errors()
    # c.write_gds("errors.gds")
    c.show()
