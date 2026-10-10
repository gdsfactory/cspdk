"""This module contains the building blocks of the cspdk.si340 library.

Foundry-matched defaults follow the CORNERSTONE 340 nm SOI standard
components (49th call) and the reference GDS files in ``cspdk/si340/gds``.
The foundry library has strip (``_sc``/``_so``) MMIs, gratings and bends,
and only a rib (``_rc``) waveguide, 90 degree bend and rib-to-strip
transition. The rib MMI, grating and coupler variants reuse the strip C-band
geometry on the 800 nm rib and have no foundry basis.
"""

from functools import partial

import gdsfactory as gf
from gdsfactory.cross_section import CrossSection
from gdsfactory.typings import (
    CrossSectionSpec,
    Ints,
    LayerSpec,
    Size,
)

from cspdk._cells import _grating_coupler_rectangular
from cspdk.si340._schematic import (
    bend_circular_schematic,
    bend_euler_schematic,
    bend_s_schematic,
    coupler_schematic,
    coupler_straight_schematic,
    grating_coupler_elliptical_schematic,
    grating_coupler_rectangular_schematic,
    mmi1x2_schematic,
    mmi2x2_schematic,
    mzi_schematic,
    pad_schematic,
    straight_schematic,
    taper_rib_to_strip_schematic,
    taper_schematic,
    wire_corner_schematic,
)
from cspdk.si340.tech import LAYER, Tech

################
# Straights
################


@gf.cell(tags=["cells"], schematic_function=straight_schematic)
def straight(
    length: float = 10.0,
    cross_section: CrossSectionSpec = "xs_sc340",
    **kwargs,
) -> gf.Component:
    """A straight waveguide.

    Args:
        length: the length of the waveguide.
        cross_section: a cross section or its name or a function generating a cross section.
        kwargs: additional arguments to pass to the straight function.
    """
    return gf.c.straight(length=length, cross_section=cross_section, **kwargs)


straight_sc = partial(straight, cross_section="xs_sc340")
straight_so = partial(straight, cross_section="xs_so340")
straight_rc = partial(straight, cross_section="xs_rc340")


################
# Bends
################


@gf.cell(tags=["cells"], schematic_function=wire_corner_schematic)
def wire_corner(cross_section="metal_routing", **kwargs) -> gf.Component:
    """A wire corner.

    A wire corner is a bend for electrical routes.

    Args:
        cross_section: "metal_routing".
        **kwargs: additional arguments.
    """
    return gf.components.wire_corner(cross_section=cross_section, **kwargs)


@gf.cell(tags=["cells"], schematic_function=bend_s_schematic)
def bend_s(
    size: tuple[float, float] = (20.0, 1.8),
    cross_section: CrossSectionSpec = "xs_sc340",
    allow_min_radius_violation: bool = True,
    width: float | None = None,
) -> gf.Component:
    """An S-bend.

    Args:
        size: the width and height of the s-bend.
        cross_section: a cross section or its name or a function generating a cross section.
        allow_min_radius_violation: if True, allows the s-bend to have a smaller radius than the minimum radius.
        width: waveguide width; defaults to the cross-section width.
    """
    return gf.components.bend_s(
        size=size,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
        width=width,
    )


@gf.cell(tags=["cells"], schematic_function=bend_euler_schematic)
def bend_euler(
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.5,
    width: float | None = None,
    cross_section: CrossSectionSpec = "xs_sc340",
) -> gf.Component:
    """An euler bend.

    Args:
        radius: the effective radius of the bend.
        angle: the angle of the bend (usually 90 degrees).
        p: the fraction of the bend that's represented by a polar bend.
        width: the width of the waveguide forming the bend.
        cross_section: a cross section or its name or a function generating a cross section.
    """
    return gf.components.bend_euler(
        radius=radius,
        angle=angle,
        p=p,
        with_arc_floorplan=True,
        npoints=None,
        layer=None,
        width=width,
        cross_section=cross_section,
        allow_min_radius_violation=False,
    )


bend_euler_sc = partial(bend_euler, cross_section="xs_sc340")
bend_euler_so = partial(bend_euler, cross_section="xs_so340")
bend_euler_rc = partial(bend_euler, cross_section="xs_rc340")


@gf.cell(tags=["cells"], schematic_function=bend_circular_schematic)
def bend_circular(
    radius: float | None = None,
    angle: float = 90.0,
    width: float | None = None,
    cross_section: CrossSectionSpec = "xs_sc340",
) -> gf.Component:
    """A circular bend, as the foundry 90 degree bends.

    The SOI340nm strip bends (R = 10 um) and rib bend (R = 100 um) are
    circular arcs, so this is the default bend for routing and the MZI. An
    Euler bend (``bend_euler``) with the same effective radius curves more
    tightly than the foundry radius.

    Args:
        radius: the radius of the bend; defaults to the cross-section radius.
        angle: the angle of the bend (usually 90 degrees).
        width: the width of the waveguide forming the bend.
        cross_section: a cross section or its name or a function generating a cross section.
    """
    return gf.components.bend_circular(
        radius=radius,
        angle=angle,
        width=width,
        cross_section=cross_section,
        allow_min_radius_violation=False,
    )


bend_circular_sc = partial(bend_circular, cross_section="xs_sc340")
bend_circular_so = partial(bend_circular, cross_section="xs_so340")
bend_circular_rc = partial(bend_circular, cross_section="xs_rc340")

################
# Transitions
################


@gf.cell(tags=["cells"], schematic_function=taper_schematic)
def taper(
    length: float = 10.0,
    width1: float | None = None,
    width2: float | None = None,
    port: gf.Port | None = None,
    cross_section: CrossSectionSpec = "xs_sc340",
) -> gf.Component:
    """A taper.

    A taper is a transition between two waveguide widths

    Args:
        length: the length of the taper.
        width1: the input width of the taper (defaults to the cross-section width).
        width2: the output width of the taper (if not given, use port).
        port: the port (with certain width) to taper towards (if not given, use width2).
        cross_section: a cross section or its name or a function generating a cross section.
    """
    if width1 is None:
        width1 = gf.get_cross_section(cross_section).width
    return gf.c.taper(
        length=length,
        width1=width1,
        width2=width2,
        port=port,
        cross_section=cross_section,
    )


taper_sc = partial(
    taper,
    cross_section="xs_sc340",
    length=10.0,
    width1=Tech.width_sc,
    width2=None,
)
taper_so = partial(
    taper,
    cross_section="xs_so340",
    length=10.0,
    width1=Tech.width_so,
    width2=None,
)
taper_rc = partial(
    taper,
    cross_section="xs_rc340",
    length=10.0,
    width1=Tech.width_rc,
    width2=None,
)


@gf.cell(tags=["cells"], schematic_function=taper_rib_to_strip_schematic)
def taper_rib_to_strip(
    length: float = 200.0,
    width_rib: float = Tech.width_rc,
    width_strip: float = Tech.width_sc,
    width_slab: float = Tech.width_rc + 2 * Tech.width_slab,
    cross_section_rib: CrossSectionSpec = "xs_rc340",
    cross_section_strip: CrossSectionSpec = "xs_sc340",
) -> gf.Component:
    """A rib (o1) to strip (o2) transition.

    Matches SOI340nm_1550nm_TE_RIB_to_STRIP: over 200 um the waveguide (GDS
    layer 3) tapers linearly from 0.8 to 0.45 um and the rib protect slab
    (GDS layer 5) from 10.8 um down to the 0.45 um strip.

    Place it explicitly between xs_rc340 and xs_sc340 waveguides. It is not
    an automatic layer transition: both cross-sections are drawn on WG, and
    gdsfactory selects auto-tapers by port layer, which cannot tell a rib
    port from a strip port.

    Args:
        length: the length of the transition.
        width_rib: the rib waveguide width at o1.
        width_strip: the strip waveguide width at o2.
        width_slab: the slab (rib protect) width at o1.
        cross_section_rib: the cross section of o1.
        cross_section_strip: the cross section of o2.
    """
    c = gf.Component()
    for layer, width in ((LAYER.WG, width_rib), (LAYER.SLAB, width_slab)):
        c.add_polygon(
            [
                (0, -width / 2),
                (length, -width_strip / 2),
                (length, width_strip / 2),
                (0, width / 2),
            ],
            layer=layer,
        )
    xs_rib = gf.get_cross_section(cross_section_rib, width=width_rib)
    xs_strip = gf.get_cross_section(cross_section_strip, width=width_strip)
    c.add_port(
        name="o1",
        center=(0, 0),
        width=width_rib,
        orientation=180,
        layer=LAYER.WG,
        cross_section=xs_rib,
    )
    c.add_port(
        name="o2",
        center=(length, 0),
        width=width_strip,
        orientation=0,
        layer=LAYER.WG,
        cross_section=xs_strip,
    )
    c.info["length"] = length
    return c


################
# MMIs
################


@gf.cell(tags=["cells"], schematic_function=mmi1x2_schematic)
def mmi1x2(
    width: float | None = None,
    width_taper=1.5,
    length_taper=20.0,
    length_mmi: float = 34.7,
    width_mmi=6.0,
    gap_mmi: float = 1.65,
    cross_section: CrossSectionSpec = "xs_sc340",
) -> gf.Component:
    """An mmi1x2.

    An mmi1x2 is a splitter that splits a single input to two outputs

    Args:
        width: the width of the waveguides connecting at the mmi ports.
        width_taper: the width at the base of the mmi body.
        length_taper: the length of the tapers going towards the mmi body.
        length_mmi: the length of the mmi body.
        width_mmi: the width of the mmi body.
        gap_mmi: the gap between the tapers at the mmi body.
        cross_section: a cross section or its name or a function generating a cross section.
    """
    return gf.c.mmi1x2(
        width=width,
        width_taper=width_taper,
        length_taper=length_taper,
        length_mmi=length_mmi,
        width_mmi=width_mmi,
        gap_mmi=gap_mmi,
        taper=taper,
        straight=straight_sc,
        cross_section=cross_section,
    )


# dimensions from the Cornerstone SOI 340nm standard components reference GDS
mmi1x2_sc = partial(mmi1x2, cross_section="xs_sc340")
mmi1x2_so = partial(mmi1x2, length_mmi=42.6, gap_mmi=1.51, cross_section="xs_so340")
# no foundry reference exists for rib MMIs; defaults follow the strip C-band values
mmi1x2_rc = partial(mmi1x2, cross_section="xs_rc340")


@gf.cell(tags=["cells"], schematic_function=mmi2x2_schematic)
def mmi2x2(
    width: float | None = None,
    width_taper: float = 1.5,
    length_taper: float = 20.0,
    length_mmi: float = 46.5,
    width_mmi: float = 6.0,
    gap_mmi: float = 0.53,
    cross_section: CrossSectionSpec = "xs_sc340",
) -> gf.Component:
    """An mmi2x2.

    An mmi2x2 is a 2x2 splitter

    Args:
        width: the width of the waveguides connecting at the mmi ports.
        width_taper: the width at the base of the mmi body.
        length_taper: the length of the tapers going towards the mmi body.
        length_mmi: the length of the mmi body.
        width_mmi: the width of the mmi body.
        gap_mmi: the gap between the tapers at the mmi body.
        cross_section: a cross section or its name or a function generating a cross section.
    """
    return gf.c.mmi2x2(
        width=width,
        width_taper=width_taper,
        length_taper=length_taper,
        length_mmi=length_mmi,
        width_mmi=width_mmi,
        gap_mmi=gap_mmi,
        taper=taper,
        straight=straight,
        cross_section=cross_section,
    )


# dimensions from the Cornerstone SOI 340nm standard components reference GDS
mmi2x2_sc = partial(mmi2x2, cross_section="xs_sc340")
mmi2x2_so = partial(mmi2x2, length_mmi=57.0, gap_mmi=0.50, cross_section="xs_so340")
# no foundry reference exists for rib MMIs; defaults follow the strip C-band values
mmi2x2_rc = partial(mmi2x2, cross_section="xs_rc340")

##############################
# Evanescent couplers
##############################


@gf.cell(tags=["cells"], schematic_function=coupler_straight_schematic)
def coupler_straight(
    length: float = 20.0,
    gap: float = 0.236,
    cross_section: CrossSectionSpec = "xs_sc340",
) -> gf.Component:
    """The straight part of a coupler.

    Args:
        length: the length of the straight part of the coupler.
        gap: the gap between the waveguides forming the straight part of the coupler.
        cross_section: a cross section or its name or a function generating a cross section.
    """
    return gf.c.coupler_straight(
        length=length,
        gap=gap,
        cross_section=cross_section,
    )


@gf.cell(tags=["cells"], schematic_function=coupler_schematic)
def coupler(
    gap: float = 0.236,
    length: float = 20.0,
    dy: float = 4.0,
    dx: float = 15.0,
    cross_section: CrossSectionSpec = "xs_sc340",
) -> gf.Component:
    """A coupler.

    a coupler is a 2x2 splitter

    Args:
        gap: the gap between the waveguides forming the straight part of the coupler.
        length: the length of the coupler.
        dy: the height of the s-bend.
        dx: the length of the s-bend.
        cross_section: a cross section or its name or a function generating a cross section.
    """
    return gf.c.coupler(
        gap=gap,
        length=length,
        dy=dy,
        dx=dx,
        cross_section=cross_section,
    )


coupler_sc = partial(coupler, cross_section="xs_sc340")
coupler_so = partial(coupler, cross_section="xs_so340")
# no foundry reference exists for couplers on any cross-section; the rib
# s-bends are stretched (dx=30) to keep the 100 um minimum rib bend radius
coupler_rc = partial(coupler, dx=30.0, cross_section="xs_rc340")


##############################
# grating couplers Rectangular
##############################


@gf.cell(tags=["cells"], schematic_function=grating_coupler_rectangular_schematic)
def grating_coupler_rectangular(
    period: float = 0.59,
    n_periods: int = 60,
    fill_factor: float = 0.5508,
    length_taper: float = 350.0,
    width_grating: float = 10.0,
    length_grating: float = 42.0,
    grating_offset: float = 2.697,
    teeth_overhang: float = 0.5,
    wavelength: float = 1.55,
    cross_section: CrossSectionSpec = "xs_sc340",
) -> gf.Component:
    """A grating coupler with straight and parallel teeth.

    Defaults reproduce SOI340nm_1550nm_TE_STRIP_Grating_Coupler: a 350 um
    linear taper from the waveguide to a 10 um wide, 42 um long grating
    section with 60 teeth, each 0.325 um etched (GDS layer 6, 140 nm deep) at
    a 0.59 um period and 11 um tall, the first 2.697 um after the taper.

    Args:
        period: the period of the grating.
        n_periods: the number of grating teeth.
        fill_factor: etched tooth width (GDS layer 6) as a fraction of the period.
        length_taper: the length of the taper tapering up to the grating.
        width_grating: the width of the waveguide under the grating.
        length_grating: the length of the full-width waveguide after the taper.
        grating_offset: distance from the end of the taper to the first tooth.
        teeth_overhang: how far the teeth extend beyond each side of the waveguide.
        wavelength: the center wavelength for which the grating is designed.
        cross_section: a cross section or its name or a function generating a cross section.
    """
    return _grating_coupler_rectangular(
        period=period,
        n_periods=n_periods,
        fill_factor=fill_factor,
        length_taper=length_taper,
        width_grating=width_grating,
        length_grating=length_grating,
        grating_offset=grating_offset,
        teeth_overhang=teeth_overhang,
        wavelength=wavelength,
        cross_section=cross_section,
        layer_grating=LAYER.GRA,
    )


# dimensions from the Cornerstone SOI 340nm standard components reference GDS
grating_coupler_rectangular_sc = partial(
    grating_coupler_rectangular,
    cross_section="xs_sc340",
)

grating_coupler_rectangular_so = partial(
    grating_coupler_rectangular,
    period=0.47,
    fill_factor=0.468,
    grating_offset=9.83,
    wavelength=1.31,
    cross_section="xs_so340",
)

# no foundry reference exists for a rib grating; this is the strip C-band
# grating fed by the 800 nm rib
grating_coupler_rectangular_rc = partial(
    grating_coupler_rectangular,
    cross_section="xs_rc340",
)


##############################
# grating couplers elliptical
##############################


@gf.cell(tags=["cells"], schematic_function=grating_coupler_elliptical_schematic)
def grating_coupler_elliptical(
    wavelength: float = 1.55,
    grating_line_width=0.315,
    cross_section="xs_sc340",
) -> gf.Component:
    """A grating coupler with curved but parallel teeth.

    Args:
        wavelength: the center wavelength for which the grating is designed.
        grating_line_width: the line width of the grating.
        cross_section: a cross section or its name or a function generating a cross section.
    """
    return gf.c.grating_coupler_elliptical_trenches(
        polarization="te",
        wavelength=wavelength,
        grating_line_width=grating_line_width,
        taper_length=16.6,
        taper_angle=30.0,
        trenches_extra_angle=9.0,
        fiber_angle=15.0,
        neff=2.638,
        ncladding=1.443,
        layer_trench=LAYER.GRA,
        p_start=26,
        n_periods=30,
        end_straight_length=0.2,
        cross_section=cross_section,
    )


grating_coupler_elliptical_sc = partial(
    grating_coupler_elliptical,
    grating_line_width=0.315,
    wavelength=1.55,
    cross_section="xs_sc340",
)

grating_coupler_elliptical_so = partial(
    grating_coupler_elliptical,
    grating_line_width=0.250,
    wavelength=1.31,
    cross_section="xs_so340",
)

grating_coupler_elliptical_rc = partial(
    grating_coupler_elliptical,
    grating_line_width=0.315,
    wavelength=1.55,
    cross_section="xs_rc340",
)

################
# MZI
################

# TODO: (needs gdsfactory fix) currently function arguments need to be
# supplied as gf.ComponentSpec strings, because when supplied as function they get
# serialized weirdly in the netlist


@gf.cell(tags=["cells"], schematic_function=mzi_schematic)
def mzi(
    delta_length: float = 10.0,
    bend="bend_circular_sc",
    straight="straight_sc",
    splitter="mmi1x2_sc",
    combiner="mmi2x2_sc",
    cross_section: CrossSectionSpec = "xs_sc340",
) -> gf.Component:
    """A Mach-Zehnder Interferometer.

    Args:
        delta_length: the difference in length between the upper and lower arms of the mzi.
        bend: the name of the default bend of the mzi.
        straight: the name of the default straight of the mzi.
        splitter: the name of the default splitter of the mzi.
        combiner: the name of the default combiner of the mzi.
        cross_section: a cross section or its name or a function generating a cross section.
    """
    return gf.c.mzi(
        delta_length=delta_length,
        length_y=1.0,
        length_x=0.1,
        straight_y=None,
        straight_x_top=None,
        straight_x_bot=None,
        with_splitter=True,
        port_e1_splitter="o2",
        port_e0_splitter="o3",
        port_e1_combiner="o3",
        port_e0_combiner="o4",
        nbends=2,
        cross_section=cross_section,
        cross_section_x_top=None,
        cross_section_x_bot=None,
        mirror_bot=False,
        add_optical_ports_arms=False,
        min_length=10e-3,
        auto_rename_ports=True,
        bend=bend,
        straight=straight,
        splitter=splitter,
        combiner=combiner,
    )


mzi_sc = partial(
    mzi,
    straight="straight_sc",
    bend="bend_circular_sc",
    splitter="mmi1x2_sc",
    combiner="mmi2x2_sc",
    cross_section="xs_sc340",
)

mzi_so = partial(
    mzi,
    straight="straight_so",
    bend="bend_circular_so",
    splitter="mmi1x2_so",
    combiner="mmi2x2_so",
    cross_section="xs_so340",
)

mzi_rc = partial(
    mzi,
    straight="straight_rc",
    bend="bend_circular_rc",
    splitter="mmi1x2_rc",
    combiner="mmi2x2_rc",
    cross_section="xs_rc340",
)


################
# Packaging
################


@gf.cell(tags=["cells"], schematic_function=pad_schematic)
def pad() -> gf.Component:
    """An electrical pad."""
    return gf.c.pad(layer=LAYER.PAD, size=(100.0, 100.0))


@gf.cell(tags=["cells"])
def rectangle(layer=LAYER.FLOORPLAN, **kwargs) -> gf.Component:
    """A rectangle.

    Args:
        layer: LAYER.FLOORPLAN.
        **kwargs: additional arguments.
    """
    return gf.c.rectangle(layer=layer, **kwargs)


@gf.cell(tags=["cells"])
def compass(
    size: Size = (4.0, 2.0),
    layer: LayerSpec = "PAD",
    port_type: str | None = "electrical",
    port_inclusion: float = 0.0,
    port_orientations: Ints | None = (180, 90, 0, -90),
    auto_rename_ports: bool = True,
) -> gf.Component:
    """Rectangle with ports on each edge (north, south, east, and west).

    Args:
        size: rectangle size.
        layer: tuple (int, int).
        port_type: optical, electrical.
        port_inclusion: from edge.
        port_orientations: list of port_orientations to add. None does not add ports.
        auto_rename_ports: auto rename ports.
    """
    return gf.c.compass(
        size=size,
        layer=layer,
        port_type=port_type,
        port_inclusion=port_inclusion,
        port_orientations=port_orientations,
        auto_rename_ports=auto_rename_ports,
    )


@gf.cell(tags=["cells"])
def grating_coupler_array(
    pitch: float = 127.0,
    n: int = 6,
    cross_section="xs_sc340",
    centered=True,
    grating_coupler=None,
    port_name="o1",
    with_loopback=False,
    rotation=-90,
    straight_to_grating_spacing=10.0,
    radius: float | None = None,
) -> gf.Component:
    """An array of grating couplers.

    Args:
        pitch: the pitch of the grating couplers
        n: the number of grating couplers
        centered: if True, centers the array around the origin.
        grating_coupler: the name of the grating coupler to use in the array.
        port_name: port name
        with_loopback: if True, adds a loopback between edge GCs. Only works for rotation = 90 for now.
        rotation: rotation angle for each reference.
        straight_to_grating_spacing: spacing between the last grating coupler and the loopback.
        cross_section: a cross section or its name or a function generating a cross section.
        radius: optional radius for routing the loopback.
    """
    if grating_coupler is None:
        if isinstance(cross_section, str):
            xs = cross_section
        elif callable(cross_section):
            xs = cross_section().name
        elif isinstance(cross_section, CrossSection):
            xs = cross_section.name
        else:
            xs = ""
        gcs = {
            "xs_sc340": "grating_coupler_rectangular_sc",
            "xs_so340": "grating_coupler_rectangular_so",
            "xs_rc340": "grating_coupler_rectangular_rc",
        }
        grating_coupler = gcs.get(xs, "grating_coupler_rectangular")
    assert grating_coupler is not None
    return gf.c.grating_coupler_array(
        grating_coupler=grating_coupler,
        pitch=pitch,
        n=n,
        with_loopback=with_loopback,
        rotation=rotation,
        straight_to_grating_spacing=straight_to_grating_spacing,
        port_name=port_name,
        centered=centered,
        cross_section=cross_section,
        radius=radius,
    )


@gf.cell(tags=["cells"])
def die(cross_section="xs_sc340") -> gf.Component:
    """A die template.

    Args:
        cross_section: a cross section or its name or a function generating a cross section.
    """
    if isinstance(cross_section, str):
        xs = cross_section
    elif callable(cross_section):
        xs = cross_section().name
    elif isinstance(cross_section, CrossSection):
        xs = cross_section.name
    else:
        xs = ""
    gcs = {
        "xs_sc340": "grating_coupler_rectangular_sc",
        "xs_so340": "grating_coupler_rectangular_so",
        "xs_rc340": "grating_coupler_rectangular_rc",
    }
    grating_coupler = gcs.get(xs, "grating_coupler_rectangular")
    return gf.c.die_with_pads(
        cross_section=cross_section,
        edge_to_grating_distance=150.0,
        edge_to_pad_distance=150.0,
        grating_coupler=grating_coupler,
        grating_pitch=250.0,
        layer_floorplan=LAYER.FLOORPLAN,
        ngratings=14,
        npads=31,
        pad="pad",
        pad_pitch=300.0,
        size=(11470.0, 4900.0),
    )


die_sc = partial(die, cross_section="xs_sc340")
die_so = partial(die, cross_section="xs_so340")
die_rc = partial(die, cross_section="xs_rc340")


@gf.cell(tags=["cells"])
def array(
    component="pad",
    columns: int = 6,
    rows: int = 1,
    add_ports: bool = True,
    size=None,
    centered: bool = False,
    column_pitch: float = 150,
    row_pitch: float = 150,
) -> gf.Component:
    """An array of components.

    Args:
        component: the component of which to create an array
        columns: the number of components to place in the x-direction
        rows: the number of components to place in the y-direction
        add_ports: add ports to the component
        size: Optional x, y size. Overrides columns and rows.
        centered: center the array around the origin.
        row_pitch: the pitch between rows.
        column_pitch: the pitch between columns.
    """
    return gf.c.array(
        component=component,
        columns=columns,
        rows=rows,
        size=size,
        centered=centered,
        add_ports=add_ports,
        column_pitch=column_pitch,
        row_pitch=row_pitch,
    )
