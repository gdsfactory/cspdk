"""Circuit simulation with routes."""

if __name__ == "__main__":
    import gdsfactory as gf

    from cspdk.si220.cband import cells, tech

    c = gf.Component()
    ptl = c << cells.pad()
    pbl = c << cells.pad()
    ptr = c << cells.pad()
    pbr = c << cells.pad()

    dx = 500

    ptr.dmove((dx, 400 + 500))
    ptl.dmove((0, 400))
    pbr.move((dx, 200 + 500))
    route = tech.route_bundle_metal(
        c, [ptl.ports["e3"], pbl.ports["e3"]], [ptr.ports["e1"], pbr.ports["e1"]]
    )
    c.show()
