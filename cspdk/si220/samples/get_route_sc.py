"""`get_route` returns a Manhattan route between two ports."""

import gdsfactory as gf

from cspdk.si220 import cells, tech

if __name__ == "__main__":
    c = gf.Component("sample_connect")
    mmi1 = c << cells.mmi1x2_sc()
    mmi2 = c << cells.mmi1x2_sc()
    mmi2.move((500, 50))

    route = tech.get_route_sc(
        mmi1.ports["o3"],
        mmi2.ports["o1"],
    )
    c.add(route.references)
    c.show()
