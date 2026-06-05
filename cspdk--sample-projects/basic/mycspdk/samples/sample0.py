"""Sample cell: routed MZI using cspdk si220 C-band."""

import gdsfactory as gf

from cspdk.si220.cband import cells, tech


@gf.cell
def sample0_routed_mzi() -> gf.Component:
    """Create two straights connected by a route."""
    c = gf.Component()
    s1 = c << cells.straight(length=10)
    s2 = c << cells.straight(length=10)
    s2.dmove((100, 50))
    tech.route_single(c, s1.ports["o2"], s2.ports["o1"])
    return c


if __name__ == "__main__":
    from cspdk.si220.cband import PDK

    PDK.activate()
    c = sample0_routed_mzi()
    c.show()
