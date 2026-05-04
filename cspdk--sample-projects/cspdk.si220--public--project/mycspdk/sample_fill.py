import gdsfactory as gf
from cspdk.si220.cband import cells


@gf.cell
def sample_fill() -> gf.Component:
    """Sample fill example."""
    c = gf.Component()
    _ = c << cells.die_with_pads()
    fill = gf.c.rectangle(layer="PAD")
    c.fill(
        fill_cell=fill,
        fill_layers=[("FLOORPLAN", -100)],
        exclude_layers=[((1, 0), 100), ("PAD", 100)],
        x_space=1,
        y_space=1,
    )
    return c
