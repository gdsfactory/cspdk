"""`get_route` returns a Manhattan route between two ports. """

import gdsfactory as gf

import cspdk

if __name__ == "__main__":
    c = gf.Component("sample_connect")
    mmi1 = c << cspdk.cells.mmi1x2_nc()
    mmi2 = c << cspdk.cells.mmi1x2_nc()
    mmi2.move((500, 50))

    route = cspdk.tech.get_route_nc(
        mmi1.ports["o3"],
        mmi2.ports["o1"],
    )
    c.add(route.references)
    c.show()
