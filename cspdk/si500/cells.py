from functools import partial

import gdsfactory as gf

from cspdk.base_cell import base_cell
from cspdk.si500.tech import LAYER

################
# Straights
################


straight = base_cell(
    "straight",
    partial(
        gf.components.straight,
        cross_section="xs_rc",
    ),
)
straight_rc = partial(straight, cross_section="xs_rc")


################
# Bends
################

wire_corner = base_cell(
    "wire_corner",
    partial(
        gf.components.wire_corner,
        cross_section="metal_routing",
    ),
)
bend_s = base_cell(
    "bend_s",
    partial(
        gf.components.bend_s,
        size=(20.0, 1.8),
        cross_section="xs_rc",
    ),
)
bend_euler = base_cell(
    "bend_euler",
    partial(
        gf.components.bend_euler,
        cross_section="xs_rc",
    ),
)

bend_rc = partial(bend_euler, cross_section="xs_rc")

################
# Transitions
################


taper = base_cell(
    "taper",
    partial(
        gf.components.taper,
        cross_section="xs_rc",
        length=10.0,
        width1=0.5,
        width2=None,
    ),
)

taper_rc = partial(
    taper,
    cross_section="xs_rc",
    length=10.0,
    width1=0.5,
    width2=None,
)

################
# MMIs
################

mmi1x2 = base_cell(
    "mmi1x2",
    partial(
        gf.components.mmi1x2,
        cross_section="xs_rc",
        width_mmi=6.0,
        width_taper=1.5,
        length_taper=20.0,
    ),
)

mmi1x2_rc = partial(mmi1x2, length_mmi=37.5, gap_mmi=1.47)

mmi2x2 = base_cell(
    "mmi2x2",
    partial(
        gf.components.mmi2x2,
        cross_section="xs_rc",
        width_mmi=6.0,
        width_taper=1.5,
        length_taper=20.0,
    ),
)
mmi2x2_rc = partial(mmi2x2, length_mmi=50.2, gap_mmi=0.4)


##############################
# Evanescent couplers
##############################

coupler_straight = base_cell(
    "coupler_straight",
    partial(
        gf.components.coupler_straight,
        cross_section="xs_rc",
    ),
)

coupler_symmetric = base_cell(
    "coupler_symmetric",
    partial(
        gf.components.coupler_symmetric,
        dx=15.0,
        dy=4.0,
        bend="bend_s",
        cross_section="xs_rc",
    ),
)

# TODO: (needs gdsfactory fix): gf.c.coupler should probably accept a bend_s
# argument that's passed into coupler_symmetric factory....

coupler = base_cell(
    "coupler",
    partial(
        gf.c.coupler,
        gap=0.234,
        length=20.0,
        dx=15.0,
        dy=4.0,
        coupler_straight="coupler_straight",
        coupler_symmetric="coupler_symmetric",
        cross_section="xs_rc",
    ),
)
coupler_rc = partial(coupler, cross_section="xs_rc")

##############################
# grating couplers Rectangular
##############################

grating_coupler_rectangular = base_cell(
    "grating_coupler_rectangular",
    partial(
        gf.components.grating_coupler_rectangular,
        n_periods=30,
        fill_factor=0.5,
        length_taper=350.0,
        fiber_angle=10.0,
        layer_grating=LAYER.GRA,
        layer_slab=LAYER.WG,
        slab_offset=0.0,
        period=0.57,
        width_grating=11.0,
        polarization="te",
        wavelength=1.55,
        slab_xmin=-1.0,
        cross_section="xs_rc",
    ),
)


grating_coupler_rectangular_rc = partial(
    grating_coupler_rectangular,
    cross_section="xs_rc",
    period=0.57,
    n_periods=60,
)


##############################
# grating couplers elliptical
##############################

grating_coupler_elliptical = base_cell(
    "grating_coupler_elliptical",
    partial(
        gf.components.grating_coupler_elliptical_trenches,
        polarization="te",
        taper_length=16.6,
        taper_angle=30.0,
        trenches_extra_angle=9.0,
        wavelength=1.53,
        fiber_angle=15.0,
        grating_line_width=0.343,
        layer_trench=LAYER.GRA,
        p_start=26,
        n_periods=30,
        end_straight_length=0.2,
        cross_section="xs_rc",
    ),
)

grating_coupler_elliptical_rc = partial(
    grating_coupler_elliptical,
    cross_section="xs_rc",
)


################
# MZI
################

# TODO: (needs gdsfactory fix) currently function arguments need to be
# supplied as ComponentSpec strings, because when supplied as function they get
# serialized weirdly in the netlist

mzi = base_cell(
    "mzi",
    partial(
        gf.components.mzi,
        delta_length=10.0,
        length_y=2.0,
        length_x=0.1,
        port_e1_splitter="o2",
        port_e0_splitter="o3",
        port_e1_combiner="o3",
        port_e0_combiner="o4",
        bend="bend_rc",
        straight="straight_rc",
        splitter="mmi1x2_rc",
        combiner="mmi2x2_rc",
        cross_section="xs_rc",
    ),
)

mzi_rc = partial(
    mzi,
    bend="bend_rc",
    straight="straight_rc",
    splitter="mmi1x2_rc",
    combiner="mmi2x2_rc",
    cross_section="xs_rc",
)


################
# Packaging
################

pad = base_cell("pad", partial(gf.c.pad, layer="PAD", size=(100.0, 100.0)))
rectangle = base_cell(
    "rectangle", partial(gf.components.rectangle, layer=LAYER.FLOORPLAN)
)
grating_coupler_array = base_cell(
    "grating_coupler_array",
    partial(
        gf.components.grating_coupler_array,
        pitch=127,
        n=6,
        port_name="o1",
        rotation=-90,
        with_loopback=False,
        grating_coupler="grating_coupler_rectangular_rc",
        cross_section="xs_rc",
    ),
)


die = base_cell(
    "die",
    partial(
        gf.c.die_with_pads,
        layer_floorplan=LAYER.FLOORPLAN,
        size=(11470.0, 4900.0),
        ngratings=14,
        npads=31,
        grating_pitch=250.0,
        grating_coupler="grating_coupler_rectangular_rc",
        pad_pitch=300.0,
        cross_section="xs_rc",
    ),
)
die_rc = partial(
    die,
    grating_coupler="grating_coupler_rectangular_rc",
    cross_section="xs_rc",
)
array = base_cell("array", partial(gf.components.array))

if __name__ == "__main__":
    c = grating_coupler_elliptical()
    c.show()
