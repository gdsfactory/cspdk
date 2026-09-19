"""Sample cell: routed MZI using cspdk si220 C-band."""

import gdsfactory as gf

from cspdk.si220 import PDK, cells, tech


@gf.cell
def sample0_routed_mzi() -> gf.Component:
    """Create two straights connected by a route."""
    PDK.activate()
    c = gf.Component()
    s1 = c << cells.straight(length=10)
    s2 = c << cells.straight(length=10)
    s2.dmove((100, 50))
    tech.route_single(c, s1.ports["o2"], s2.ports["o1"])
    c.add_port(name="o1", port=s1.ports["o1"])
    c.add_port(name="o2", port=s2.ports["o2"])
    return c


if __name__ == "__main__":
    c = sample0_routed_mzi()
    c.show()
