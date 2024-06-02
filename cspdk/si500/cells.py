from functools import partial

import gdsfactory as gf
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from cspdk.si500.tech import LAYER

################
# Straights
################


straight = partial(gf.components.straight, cross_section="xs_sc")
straight_rc = partial(straight, cross_section="xs_rc")


################
# Bends
################


bend_s = partial(gf.components.bend_s, size=(11.0, 1.8), cross_section="xs_rc")
wire_corner = partial(gf.components.wire_corner, cross_section="metal_routing")
bend_euler = partial(gf.components.bend_euler, cross_section="xs_rc")
bend_rc = partial(bend_euler, cross_section="xs_rc")

################
# Transitions
################


taper = taper_sc = partial(
    gf.components.taper, cross_section="xs_rc", length=10.0, width1=0.5, width2=None
)
taper_rc = partial(
    gf.components.taper, cross_section="xs_rc", length=10.0, width1=0.5, width2=None
)

################
# MMIs
################
mmi1x2 = partial(
    gf.components.mmi1x2,
    cross_section="xs_rc",
    width_mmi=6.0,
    width_taper=1.5,
    length_taper=20.0,
)
mmi2x2 = partial(
    gf.components.mmi2x2,
    cross_section="xs_rc",
    width_mmi=6.0,
    width_taper=1.5,
    length_taper=20.0,
)

mmi1x2_rc = partial(mmi1x2, length_mmi=37.5, gap_mmi=1.47)
mmi2x2_rc = partial(mmi2x2, length_mmi=50.2, gap_mmi=0.4)


##############################
# Evanescent couplers
##############################


coupler = partial(
    gf.c.coupler, gap=0.234, length=20.0, dx=10.0, dy=4.0, cross_section="xs_sc"
)
coupler_rc = partial(coupler, cross_section="xs_rc", dx=15, dy=4.0)
coupler_ro = partial(coupler, cross_section="xs_ro", dx=15, dy=4.0)

##############################
# grating couplers Rectangular
##############################

grating_coupler_rectangular = partial(
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

grating_coupler_elliptical = partial(
    gf.components.grating_coupler_elliptical,
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
)


################
# MZI
################


mzi_rc = partial(
    gf.components.mzi,
    combiner=mmi1x2_rc,
    splitter=mmi1x2_rc,
    cross_section="xs_rc",
)


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
    grating_coupler="grating_coupler_rectangular_rc",
    cross_section="xs_rc",
)


@gf.cell
def _die(
    size: tuple[float, float] = (11470.0, 4900.0),
    ngratings: int = 14,
    npads: int = 31,
    grating_pitch: float = 250.0,
    pad_pitch: float = 300.0,
    grating_coupler: ComponentSpec = "grating_coupler_rectangular_rc",
    cross_section: CrossSectionSpec = "xs_rc",
    pad: ComponentSpec = "pad",
) -> gf.Component:
    """A die with grating couplers and pads.

    Args:
        size: the size of the die, in um.
        ngratings: the number of grating couplers.
        npads: the number of pads.
        grating_pitch: the pitch of the grating couplers, in um.
        pad_pitch: the pitch of the pads, in um.
        grating_coupler: the grating coupler component.
        cross_section: the cross section.
        pad: the pad component.
    """
    c = gf.Component()

    fp = c << rectangle(size=size, layer=LAYER.FLOORPLAN, centered=True)

    # Add optical ports
    x0 = -4925 + 2.827
    y0 = 1650

    gca = grating_coupler_array(
        n=ngratings,
        pitch=grating_pitch,
        with_loopback=True,
        grating_coupler=grating_coupler,
        cross_section=cross_section,
    )
    left = c << gca
    left.drotate(-90)
    left.dxmax = x0
    left.dy = fp.dy
    c.add_ports(left.ports, prefix="W")

    right = c << gca
    right.drotate(+90)
    right.dxmax = -x0
    right.dy = fp.dy
    c.add_ports(right.ports, prefix="E")

    # Add electrical ports
    x0 = -4615
    y0 = 2200
    pad = gf.get_component(pad)

    for i in range(npads):
        pad_ref = c << pad
        pad_ref.dxmin = x0 + i * pad_pitch
        pad_ref.dymin = y0
        c.add_port(
            name=f"N{i}",
            port=pad_ref.ports["e4"],
        )

    for i in range(npads):
        pad_ref = c << pad
        pad_ref.dxmin = x0 + i * pad_pitch
        pad_ref.dymax = -y0
        c.add_port(
            name=f"S{i}",
            port=pad_ref.ports["e2"],
        )

    c.auto_rename_ports()
    return c


die_rc = partial(
    _die, grating_coupler="grating_coupler_rectangular_rc", cross_section="xs_rc"
)
array = gf.components.array

if __name__ == "__main__":
    c = die_rc()
    c.show()
