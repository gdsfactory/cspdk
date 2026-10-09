"""Connect Cornerstone Si220 cells port to port."""

import gdsfactory as gf
from cspdk.si220 import cells


@gf.cell
def sample1_connect() -> gf.Component:
    """Chain a straight, a bend and an MMI by connecting their ports."""
    c = gf.Component()
    wg = c << cells.straight(length=20)
    bend = c << cells.bend_euler()
    mmi = c << cells.mmi1x2()
    bend.connect("o1", wg["o2"])
    mmi.connect("o1", bend["o2"])
    c.add_port("o1", port=wg["o1"])
    c.add_ports(mmi.ports.filter(orientation=mmi["o2"].orientation), prefix="out_")
    return c


if __name__ == "__main__":
    sample1_connect().show()
