import gdsfactory as gf


@gf.cell
def gc() -> gf.Component:
    return gf.c.grating_coupler_elliptical(layer_slab=None, cross_section="strip_cband")
