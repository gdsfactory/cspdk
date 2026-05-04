import gdsfactory as gf


@gf.cell
def gc() -> gf.Component:
    c = gf.c.grating_coupler_elliptical(layer_slab=None)
    return c
