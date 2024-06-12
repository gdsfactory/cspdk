from functools import partial

import gdsfactory as gf

from cspdk.sin300.tech import LAYER

################
# Straights
################

straight = partial(gf.components.straight, cross_section="xs_nc")
straight_nc = partial(straight, cross_section="xs_nc")
straight_no = partial(straight, cross_section="xs_no")
################
# Bends
################

bend_s = partial(gf.components.bend_s, size=(11.0, 1.8), cross_section="xs_nc")
wire_corner = partial(gf.components.wire_corner, cross_section="metal_routing")
bend_euler = partial(gf.components.bend_euler, cross_section="xs_nc")
bend_nc = partial(bend_euler, cross_section="xs_nc")
bend_no = partial(bend_euler, cross_section="xs_no")


################
# Transitions
################

taper = taper_nc = partial(
    gf.components.taper, cross_section="xs_nc", length=10.0, width1=0.5, width2=None
)
taper_no = partial(
    gf.components.taper, cross_section="xs_no", length=10.0, width1=0.5, width2=None
)


################
# MMIs
################

mmi1x2 = partial(
    gf.components.mmi1x2,
    cross_section="xs_nc",
    width_mmi=12.0,
    width_taper=5.5,
    length_taper=50.0,
)
mmi2x2 = partial(
    gf.c.mmi2x2,
    cross_section="xs_nc",
    width_mmi=12.0,
    width_taper=5.5,
    length_taper=50.0,
)
mmi1x2_no = partial(
    mmi1x2,
    cross_section="xs_no",
    length_mmi=42.0,
    gap_mmi=0.4,
)
mmi2x2_no = partial(
    mmi2x2,
    cross_section="xs_no",
    length_taper=50.0,
    length_mmi=126.0,
    gap_mmi=0.4,
)

mmi1x2_nc = partial(
    mmi1x2,
    cross_section="xs_nc",
    length_mmi=64.7,
    gap_mmi=0.4,
)

mmi2x2_nc = partial(
    mmi2x2,
    cross_section="xs_nc",
    length_mmi=232.0,
    gap_mmi=0.4,
)


##############################
# Evanescent couplers
##############################


coupler = partial(
    gf.c.coupler, gap=0.234, length=20.0, dx=15.0, dy=4.0, cross_section="xs_nc"
)
coupler_nc = partial(coupler, cross_section="xs_nc")
coupler_no = partial(coupler, cross_section="xs_no")

##############################
# grating couplers Rectangular
##############################
grating_coupler_rectangular = partial(
    gf.components.grating_coupler_rectangular,
    n_periods=30,
    fill_factor=0.5,
    length_taper=200.0,
    fiber_angle=20.0,
    layer_grating=LAYER.NITRIDE_ETCH,
    layer_slab=LAYER.NITRIDE,
    slab_offset=0.0,
    period=0.75,
    width_grating=11.0,
    polarization="te",
    wavelength=1.55,
    slab_xmin=-1.0,
    cross_section="xs_nc",
)


grating_coupler_rectangular_nc = partial(
    grating_coupler_rectangular,
    period=0.66,
    cross_section="xs_nc",
)

grating_coupler_rectangular_no = partial(
    grating_coupler_rectangular,
    period=0.964,
    cross_section="xs_no",
)

##############################
# grating couplers elliptical
##############################

grating_coupler_elliptical_trenches = partial(
    gf.components.grating_coupler_elliptical_trenches,
    polarization="te",
    taper_length=16.6,
    taper_angle=30.0,
    wavelength=1.55,
    fiber_angle=20.0,
    grating_line_width=0.343,
    neff=1.6,
    ncladding=1.443,
    layer_trench=LAYER.GRA,
    p_start=26,
    n_periods=30,
    end_straight_length=0.2,
    cross_section="xs_nc",
)

grating_coupler_elliptical_trenches_nc = partial(
    grating_coupler_elliptical_trenches,
    cross_section="xs_nc",
    grating_line_width=0.66 / 2,
)

grating_coupler_elliptical_trenches_no = partial(
    grating_coupler_elliptical_trenches,
    grating_line_width=0.964 / 2,
    cross_section="xs_no",
)


################
# MZI
################


mzi = partial(gf.components.mzi, cross_section="xs_nc")
mzi_nc = partial(mzi, splitter=mmi1x2_nc, cross_section="xs_nc")
mzi_no = partial(mzi, splitter=mmi1x2_no, cross_section="xs_no")


################
# Packaging
################


pad = partial(gf.c.pad, layer="PAD", size=(100.0, 100.0))
rectangle = partial(gf.components.rectangle, layer=LAYER.FLOORPLAN)
grating_coupler_array = partial(
    gf.components.grating_coupler_array,
    pitch=127,
    n=6,
    port_name="o1",
    rotation=-90,
    with_loopback=False,
    grating_coupler="grating_coupler_rectangular_nc",
    cross_section="xs_nc",
)


_die = partial(
    gf.c.die_with_pads,
    layer_floorplan=LAYER.FLOORPLAN,
    size=(11470.0, 4900.0),
    ngratings=14,
    npads=31,
    grating_pitch=250.0,
    pad_pitch=300.0,
)

die_nc = partial(
    _die, grating_coupler="grating_coupler_rectangular_nc", cross_section="xs_nc"
)
die_no = partial(
    _die, grating_coupler="grating_coupler_rectangular_no", cross_section="xs_no"
)


array = gf.components.array


if __name__ == "__main__":
    c = mzi_no()
    c.show()
