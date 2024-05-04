from functools import partial
from typing import Any

import gdsfactory as gf
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, LayerSpec

from cspdk.si500.config import PATH
from cspdk.si500.tech import LAYER, xs_rc

################
# Straights
################


@gf.cell
def _straight(
    length: float = 10.0,
    cross_section: CrossSectionSpec = "xs_sc",
) -> gf.Component:
    return gf.components.straight(
        length=length,
        cross_section=cross_section,
    )


@gf.cell
def straight_rc(length: float = 10.0, **kwargs) -> gf.Component:
    """Straight waveguide in rib, c-band.

    Args:
        length (float, optional): The length of the waveguide. Defaults to 10.0.

    Returns:
        gf.Component: the component
    """
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_rc"
    return _straight(
        length=length,
        **kwargs,
    )


################
# Bends
################


@gf.cell
def bend_s(
    size: tuple[float, float] = (11.0, 1.8),
    cross_section: CrossSectionSpec = "xs_rc",
    **kwargs,
) -> gf.Component:
    """An S-bend.

    Args:
        size (tuple[float, float], optional): The size of the s-bend, in x and y. Defaults to (11.0, 1.8).
        cross_section (CrossSectionSpec, optional): the bend cross-section. Defaults to "xs_sc" (strip, c-band).

    Returns:
        gf.Component: the component
    """
    return gf.components.bend_s(
        size=size,
        cross_section=cross_section,
        **kwargs,
    )


@gf.cell
def _bend(
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.5,
    with_arc_floorplan: bool = True,
    npoints: int | None = None,
    direction: str = "ccw",
    cross_section: CrossSectionSpec = "xs_sc",
) -> gf.Component:
    return gf.components.bend_euler(
        radius=radius,
        angle=angle,
        p=p,
        with_arc_floorplan=with_arc_floorplan,
        npoints=npoints,
        direction=direction,
        cross_section=cross_section,
    )


@gf.cell
def wire_corner(
    cross_section: CrossSectionSpec = "xs_metal_routing",
) -> gf.Component:
    """The bend equivalent for electrical wires, which is a simple corner.

    Args:
        cross_section (CrossSectionSpec, optional): the bend cross-section. Defaults to "xs_metal_routing".

    Returns:
        gf.Component: the component
    """
    return gf.components.wire_corner(cross_section=cross_section)


def _float(x: Any) -> float:
    return float(x)


@gf.cell
def bend_rc(
    radius: float = _float(xs_rc.radius), angle: float = 90.0, **kwargs
) -> gf.Component:
    """An euler bend in rib, c-band.

    Args:
        radius (float, optional): the radius of the bend. Defaults to the PDK's default value for that cross-section.
        angle (float, optional): the angle of the bend. Defaults to 90.0.

    Returns:
        gf.Component: the component
    """
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_rc"
    return _bend(
        radius=radius,
        angle=angle,
        **kwargs,
    )


################
# Transitions
################


@gf.cell
def _taper(
    length: float = 10.0,
    width1: float = 0.5,
    width2: float | None = None,
    port: gf.Port | None = None,
    cross_section: CrossSectionSpec = "xs_sc",
    **kwargs,
) -> gf.Component:
    return gf.components.taper(
        length=length,
        width1=width1,
        width2=width2,
        port=port,
        cross_section=cross_section,
        **kwargs,
    )


@gf.cell
def taper_rc(
    length: float = 10.0,
    width1: float = 0.5,
    width2: float | None = None,
    port: gf.Port | None = None,
    **kwargs,
) -> gf.Component:
    """A width taper in rib, c-band.

    Args:
        length (float, optional): the length of the taper, in um. Defaults to 10.0.
        width1 (float, optional): the width of the taper input, in um. Defaults to 0.5.
        width2 (float | None, optional): the width of the taper output, in um. Defaults to None.
        port (gf.Port | None, optional): if given, starts from the port's width and transitions to width1. Defaults to None.

    Returns:
        gf.Component: the component
    """
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_rc"
    return _taper(length=length, width1=width1, width2=width2, port=port, **kwargs)


################
# MMIs
################


@gf.cell
def _mmi1x2(
    width_mmi: float = 6.0,
    width_taper: float = 1.6,
    length_taper: float = 20.0,
    cross_section: CrossSectionSpec = "xs_sc",
    **kwargs,
) -> gf.Component:
    return gf.components.mmi1x2(
        width_mmi=width_mmi,
        length_taper=length_taper,
        width_taper=width_taper,
        cross_section=cross_section,
        **kwargs,
    )


@gf.cell
def _mmi2x2(
    width_mmi: float = 6.0,
    width_taper: float = 1.6,
    length_taper: float = 20.0,
    cross_section: CrossSectionSpec = "xs_sc",
    **kwargs,
) -> gf.Component:
    return gf.components.mmi2x2(
        width_mmi=width_mmi,
        length_taper=length_taper,
        width_taper=width_taper,
        cross_section=cross_section,
        **kwargs,
    )


################
# MMIs rib cband
################


@gf.cell
def mmi1x2_rc(**kwargs) -> gf.Component:
    """A 1x2 MMI in rib, c-band.

    Returns:
        gf.Component: the component
    """
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_rc"
    return _mmi1x2(
        length_mmi=37.5,
        gap_mmi=1.47,
        **kwargs,
    )


@gf.cell
def mmi2x2_rc(**kwargs) -> gf.Component:
    """A 2x2 MMI in rib, c-band.

    Returns:
        gf.Component: the component
    """
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_rc"
    return _mmi2x2(
        length_mmi=50.2,
        gap_mmi=0.4,
        **kwargs,
    )


##############################
# Evanescent couplers
##############################


@gf.cell
def _coupler_symmetric(
    bend: ComponentSpec = bend_s,
    gap: float = 0.234,
    dx: float = 10.0,
    dy: float = 4.0,
    cross_section: CrossSectionSpec = "xs_sc",
) -> gf.Component:
    return gf.components.coupler_symmetric(
        bend=bend,
        gap=gap,
        dx=dx,
        dy=dy,
        cross_section=cross_section,
    )


@gf.cell
def _coupler_straight(
    length: float = 10.0,
    gap: float = 0.27,
    straight: ComponentSpec = _straight,
    cross_section: CrossSectionSpec = "xs_sc",
) -> gf.Component:
    return gf.components.coupler_straight(
        length=length,
        gap=gap,
        straight=straight,
        cross_section=cross_section,
    )


@gf.cell
def _coupler(
    gap: float = 0.236,
    length: float = 20.0,
    coupler_symmetric: ComponentSpec = _coupler_symmetric,
    coupler_straight: ComponentSpec = _coupler_straight,
    dx: float = 10.0,
    dy: float = 4.0,
    cross_section: CrossSectionSpec = "xs_sc",
) -> gf.Component:
    return gf.components.coupler(
        gap=gap,
        length=length,
        coupler_symmetric=coupler_symmetric,
        coupler_straight=coupler_straight,
        dx=dx,
        dy=dy,
        cross_section=cross_section,
    )


@gf.cell
def coupler_rc(
    gap: float = 0.236,
    length: float = 20.0,
    dx: float = 10.0,
    dy: float = 4.0,
    **kwargs,
) -> gf.Component:
    """A symmetric coupler in rib, c-band.

    Args:
        gap (float, optional): the coupling gap, in um. Defaults to 0.236.
        length (float, optional): the length of the coupling section, in um. Defaults to 20.0.
        dx (float, optional): the port-to-port horizontal spacing. Defaults to 10.0.
        dy (float, optional): the port-to-port vertical spacing. Defaults to 4.0.

    Returns:
        gf.Component: the component
    """
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_rc"
    return _coupler(
        gap=gap,
        length=length,
        dx=dx,
        dy=dy,
        **kwargs,
    )


##############################
# grating couplers Rectangular
##############################


@gf.cell
def _gc_rectangular(
    n_periods: int = 30,
    fill_factor: float = 0.5,
    length_taper: float = 350.0,
    fiber_angle: float = 10.0,
    layer_grating: LayerSpec = LAYER.GRA,
    layer_slab: LayerSpec = LAYER.WG,
    slab_offset: float = 0.0,
    period: float = 0.57,
    width_grating: float = 10.0,
    polarization: str = "te",
    wavelength: float = 1.55,
    taper: ComponentSpec = _taper,
    slab_xmin: float = -1.0,
    cross_section: CrossSectionSpec = "xs_rc",
) -> gf.Component:
    return gf.components.grating_coupler_rectangular(
        n_periods=n_periods,
        fill_factor=fill_factor,
        length_taper=length_taper,
        fiber_angle=fiber_angle,
        layer_grating=layer_grating,
        layer_slab=layer_slab,
        slab_offset=slab_offset,
        period=period,
        width_grating=width_grating,
        polarization=polarization,
        wavelength=wavelength,
        taper=taper,
        slab_xmin=slab_xmin,
        cross_section=cross_section,
    ).flatten()


@gf.cell
def gc_rectangular_rc() -> gf.Component:
    """A rectangular grating coupler in rib, c-band.

    Returns:
        gf.Component: the component
    """
    return _gc_rectangular(
        period=0.57,
        cross_section="xs_rc",
        n_periods=60,
    )


##############################
# grating couplers elliptical
##############################


@gf.cell
def _gc_elliptical(
    polarization: str = "te",
    taper_length: float = 16.6,
    taper_angle: float = 30.0,
    trenches_extra_angle: float = 9.0,
    wavelength: float = 1.53,
    fiber_angle: float = 15.0,
    grating_line_width: float = 0.343,
    neff: float = 2.638,
    ncladding: float = 1.443,
    layer_trench: LayerSpec = LAYER.GRA,
    p_start: int = 26,
    n_periods: int = 30,
    end_straight_length: float = 0.2,
    cross_section: CrossSectionSpec = "xs_sc",
) -> gf.Component:
    return gf.components.grating_coupler_elliptical_trenches(
        polarization=polarization,
        taper_length=taper_length,
        taper_angle=taper_angle,
        trenches_extra_angle=trenches_extra_angle,
        wavelength=wavelength,
        fiber_angle=fiber_angle,
        grating_line_width=grating_line_width,
        neff=neff,
        ncladding=ncladding,
        layer_trench=layer_trench,
        p_start=p_start,
        n_periods=n_periods,
        end_straight_length=end_straight_length,
        cross_section=cross_section,
    )


################
# MZI
################


@gf.cell
def _mzi(
    delta_length: float = 10.0,
    length_y: float = 2.0,
    length_x: float = 0.1,
    cross_section: CrossSectionSpec = "xs_sc",
    add_electrical_ports_bot: bool = True,
    bend: ComponentSpec = _bend,
    straight: ComponentSpec = _straight,
    splitter: ComponentSpec = _mmi1x2,
    combiner: ComponentSpec = _mmi2x2,
) -> gf.Component:
    return gf.components.mzi(
        delta_length=delta_length,
        length_y=length_y,
        length_x=length_x,
        bend=bend,
        straight=straight,
        straight_y=straight,
        straight_x_top=straight,
        straight_x_bot=straight,
        splitter=splitter,
        combiner=combiner,
        with_splitter=True,
        port_e1_splitter="o2",
        port_e0_splitter="o3",
        port_e1_combiner="o2",
        port_e0_combiner="o3",
        nbends=2,
        cross_section=cross_section,
        cross_section_x_top=cross_section,
        cross_section_x_bot=cross_section,
        mirror_bot=False,
        add_optical_ports_arms=False,
        add_electrical_ports_bot=add_electrical_ports_bot,
        min_length=0.01,
        extend_ports_straight_x=None,
    )


@gf.cell
def mzi_rc(
    delta_length: float = 10.0,
    length_y: float = 2.0,
    length_x: float = 0.1,
    add_electrical_ports_bot: bool = True,
    **kwargs,
) -> gf.Component:
    """A Mach-Zehnder Interferometer (MZI) in rib, c-band.

    Args:
        delta_length (float, optional): the length differential between the two arms. Defaults to 10.0.
        length_y (float, optional): the common vertical length, in um. Defaults to 2.0.
        length_x (float, optional): the common horizontal length, in um. Defaults to 0.1.
        add_electrical_ports_bot (bool, optional): if true, adds electrical ports to the bottom. Defaults to True.

    Returns:
        gf.Component: the component
    """
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_rc"
    return _mzi(
        delta_length=delta_length,
        length_y=length_y,
        length_x=length_x,
        add_electrical_ports_bot=add_electrical_ports_bot,
        straight=straight_rc,
        bend=bend_rc,
        combiner=mmi1x2_rc,
        splitter=mmi1x2_rc,
        **kwargs,
    )


################
# Packaging
################


@gf.cell
def pad(
    size: tuple[float, float] = (100.0, 100.0),
    layer: LayerSpec = LAYER.PAD,
    bbox_layers: None = None,
    bbox_offsets: None = None,
    port_inclusion: float = 0.0,
    port_orientation: None = None,
) -> gf.Component:
    """An electrical pad.

    Args:
        size (tuple[float, float], optional): Size of the pad in (x, y). Defaults to (100.0, 100.0).
        layer (LayerSpec, optional): the layer to draw the pad on. Defaults to LAYER.PAD.
        bbox_layers (None, optional): if set, draws a box around the pad with the given layers. Defaults to None.
        bbox_offsets (None, optional): if set, applies an offset to grow the bbox's specified with `bbox_layers`. Defaults to None.
        port_inclusion (float, optional): if set, insets the port from the edge by the specified amount. Defaults to 0.0.
        port_orientation (None, optional): if set, gives the port a fixed orientation. Defaults to None.

    Returns:
        gf.Component: the component
    """
    return gf.components.pad(
        size=size,
        layer=layer,
        bbox_layers=bbox_layers,
        bbox_offsets=bbox_offsets,
        port_inclusion=port_inclusion,
        port_orientation=port_orientation,
    )


@gf.cell
def rectangle(
    size: tuple[float, float] = (4.0, 2.0),
    layer: LayerSpec = LAYER.FLOORPLAN,
    centered: bool = False,
    port_type: str = "electrical",
    port_orientations: tuple[float, float, float, float] = (180.0, 90.0, 0.0, -90.0),
    round_corners_east_west: bool = False,
    round_corners_north_south: bool = False,
) -> gf.Component:
    """A simple rectangle on the given layer.

    Args:
        size (tuple[float, float], optional): the size of the rectangle in (x, y). Defaults to (4.0, 2.0).
        layer (LayerSpec, optional): the layer to draw the rectangle on. Defaults to LAYER.FLOORPLAN.
        centered (bool, optional): if true, the rectangle's origin will be placed at the center (otherwise it will be bottom-left). Defaults to False.
        port_type (str, optional): the port type for ports automatically added to edges of the rectangle. Defaults to "electrical".
        port_orientations (tuple[float, float, float, float], optional): orientations of the ports to be automatically added. Defaults to (180.0, 90.0, 0.0, -90.0).
        round_corners_east_west (bool, optional): if True, circles are added to the east and west edges, forming a horizontal pill shape. Defaults to False.
        round_corners_north_south (bool, optional): if True, circles are added to the north and south edges, forming a vertical pill shape. Defaults to False.

    Returns:
        gf.Component: the component
    """
    return gf.components.rectangle(
        size=size,
        layer=layer,
        centered=centered,
        port_type=port_type,
        port_orientations=port_orientations,
        round_corners_east_west=round_corners_east_west,
        round_corners_north_south=round_corners_north_south,
    )


@gf.cell
def grating_coupler_array(
    pitch: float = 127.0,
    n: int = 6,
    port_name: str = "o1",
    rotation: float = 0.0,
    with_loopback: bool = False,
    bend: ComponentSpec = _bend,
    grating_coupler_spacing: float = 0.0,
    grating_coupler: ComponentSpec = gc_rectangular_rc,
    cross_section: CrossSectionSpec = "xs_sc",
) -> gf.Component:
    """An array of grating couplers.

    Args:
        pitch (float, optional): the center-center pitch between grating couplers. Defaults to 127.0.
        n (int, optional): the number of grating couplers to place. Defaults to 6.
        port_name (str, optional): the routing port of the grating coupler to be placed. Defaults to "o1".
        rotation (float, optional): rotation of the grating couplers, in degrees. Defaults to 0.0.
        with_loopback (bool, optional): if True, adds a loopback. Defaults to False.
        bend (ComponentSpec, optional): the bend to be used for the loopback. Defaults to _bend.
        grating_coupler_spacing (float, optional): the spacing to be used in the loopback. Defaults to 0.0.
        grating_coupler (ComponentSpec, optional): the grating coupler component to use.
        cross_section (CrossSectionSpec, optional): the cross section to be used for routing in the loopback.

    Returns:
        gf.Component: the component
    """
    return gf.components.grating_coupler_array(
        pitch=pitch,
        n=n,
        port_name=port_name,
        rotation=rotation,
        with_loopback=with_loopback,
        bend=bend,
        grating_coupler_spacing=grating_coupler_spacing,
        grating_coupler=grating_coupler,
        cross_section=cross_section,
    )


@gf.cell
def _die(
    size: tuple[float, float] = (11470.0, 4900.0),
    ngratings: int = 14,
    npads: int = 31,
    grating_pitch: float = 250.0,
    pad_pitch: float = 300.0,
    grating_coupler: ComponentSpec = gc_rectangular_rc,
    cross_section: CrossSectionSpec = "xs_sc",
    pad: ComponentSpec = pad,
) -> gf.Component:
    c = gf.Component()

    fp = c << rectangle(size=size, layer=LAYER.FLOORPLAN, centered=True)

    # Add optical ports
    x0 = -4925 + 2.827
    y0 = 1650

    gca = grating_coupler_array(
        n=ngratings,
        pitch=grating_pitch,
        with_loopback=True,
        rotation=90,
        grating_coupler=grating_coupler,
        cross_section=cross_section,
    )
    left = c << gca
    left.rotate(90)
    left.xmax = x0
    left.y = fp.y
    c.add_ports(left.ports, prefix="W")

    right = c << gca
    right.rotate(-90)
    right.xmax = -x0
    right.y = fp.y
    c.add_ports(right.ports, prefix="E")

    # Add electrical ports
    x0 = -4615
    y0 = 2200
    pad = pad()

    for i in range(npads):
        pad_ref = c << pad
        pad_ref.xmin = x0 + i * pad_pitch
        pad_ref.ymin = y0
        c.add_port(
            name=f"N{i}",
            port=pad_ref.ports["e4"],
        )

    for i in range(npads):
        pad_ref = c << pad
        pad_ref.xmin = x0 + i * pad_pitch
        pad_ref.ymax = -y0
        c.add_port(
            name=f"S{i}",
            port=pad_ref.ports["e2"],
        )

    c.auto_rename_ports()
    return c


@gf.cell
def die_rc() -> gf.Component:
    """The standard die template for rib, c-band. This has 24 grating couplers, split evenly between the left and right sides of the chip and 62 electrical pads split between the top and bottom.

    Returns:
        gf.Component: the component
    """
    return _die(
        grating_coupler=gc_rectangular_rc,
        cross_section="xs_rc",
    )


################
# Imported from Cornerstone MPW SOI 220nm GDSII Template
################
_import_gds = partial(gf.import_gds, gdsdir=PATH.gds)


array = gf.components.array


if __name__ == "__main__":
    c = die_rc()
    c.show()
