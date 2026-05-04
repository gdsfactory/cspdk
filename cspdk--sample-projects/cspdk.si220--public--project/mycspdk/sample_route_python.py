import gdsfactory as gf


@gf.cell
def sample_route_python() -> gf.Component:
    c = gf.Component()
    top = c << gf.components.nxn(north=8, south=0, east=0, west=0)
    bot = c << gf.components.nxn(north=2, south=2, east=2, west=2, xsize=10, ysize=10)
    top.movey(100)

    routes = gf.routing.route_bundle(
        c,
        ports1=bot.ports,
        ports2=top.ports,
        radius=5,
        sort_ports=True,
        cross_section="strip",
    )
    for route in routes:
        print(route.length)
    return c
