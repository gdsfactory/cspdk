"""A die with grating couplers and pads."""

import gdsfactory as gf
from gdsfactory.typings import (
    ComponentSpec,
    CrossSectionSpec,
    Ints,
    LayerSpec,
    Size,
)

from cspdk.si220._schematic import die_schematic, pad_schematic
from cspdk.si220._utils import get_band_component as _get_band_component
from cspdk.si220.tech import LAYER


@gf.cell(tags=["die"])
def compass(
    size: Size = (4, 2),
    layer: LayerSpec = "PAD",
    port_type: str | None = None,
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


@gf.cell(tags=["die"])
def rectangle(
    size: Size = (4, 2),
    layer: LayerSpec = "PAD",
    centered: bool = False,
    port_type: str | None = None,
    port_orientations: Ints | None = (180, 90, 0, -90),
) -> gf.Component:
    """Returns a rectangle.

    Args:
        size: (tuple) Width and height of rectangle.
        layer: Specific layer to put polygon geometry on.
        centered: True sets center to (0, 0), False sets south-west to (0, 0).
        port_type: optical, electrical.
        port_orientations: list of port_orientations to add. None adds no ports.
    """
    return gf.c.rectangle(
        size=size,
        layer=layer,
        centered=centered,
        port_type=port_type,
        port_orientations=port_orientations,
    )


@gf.cell(tags=["die"], schematic_function=pad_schematic)
def pad(
    size: tuple[float, float] = (90.0, 90.0),
    layer: LayerSpec = "PAD",
    port_inclusion: float = 0,
    port_orientation: float = 0,
) -> gf.Component:
    """Returns rectangular pad with ports.

    Args:
        size: x, y size.
        layer: pad layer.
        bbox_layers: list of layers.
        bbox_offsets: Optional offsets for each layer with respect to size.
            positive grows, negative shrinks the size.
        port_inclusion: from edge.
        port_orientation: in degrees.
    """
    return gf.components.pad(
        size=size,
        layer=layer,
        port_inclusion=port_inclusion,
        port_orientation=port_orientation,
    )


@gf.cell(tags=["die"])
def die(size: tuple[float, float] = (11470.0, 4900.0)) -> gf.Component:
    """Return the die floorplan (GDS 99), centred on the origin.

    The default is the Run-50 full design area, 11470 x 4900 um; ``die_half`` is the
    5500 x 4900 um half design area.

    Args:
        size: Die width and height in micrometres.
    """
    c = gf.Component()
    _ = c << gf.c.rectangle(
        size=size, layer=LAYER.FLOORPLAN, centered=True, port_type=None
    )
    return c


@gf.cell(tags=["die"])
def die_half() -> gf.Component:
    """Return the Run-50 half design area floorplan, 5500 x 4900 um."""
    return die(size=(5500.0, 4900.0))


def _loopback_path(
    gc_length: float, span: float, radius: float, straight: float, clearance: float
) -> gf.Path:
    """Centreline from the bottom grating port (heading east) to the top one.

    It leaves the port for ``straight``, turns back under the grating with a 180
    degree arc, runs up ``clearance`` behind the far end of the gratings and returns
    the same way into the top grating. ``span`` is the distance between the ports.
    """
    back = straight + gc_length + clearance - radius
    p = gf.Path()
    p.append(gf.path.straight(length=straight))
    p.append(gf.path.arc(radius=radius, angle=-180, npoints=373))
    p.append(gf.path.straight(length=back))
    p.append(gf.path.arc(radius=radius, angle=-90, npoints=187))
    p.append(gf.path.straight(length=span + 2 * radius))
    p.append(gf.path.arc(radius=radius, angle=-90, npoints=187))
    p.append(gf.path.straight(length=back))
    p.append(gf.path.arc(radius=radius, angle=-180, npoints=373))
    p.append(gf.path.straight(length=straight))
    return p


@gf.cell(tags=["die"], schematic_function=die_schematic)
def die_with_pads(
    size: tuple[float, float] = (11470.0, 4900.0),
    ngratings: int = 16,
    npads: int = 31,
    grating_pitch: float = 250.0,
    pad_pitch: float = 300.0,
    grating_coupler: ComponentSpec | None = "grating_coupler_rectangular",
    cross_section: CrossSectionSpec = "strip_cband",
    pad: ComponentSpec = "pad",
    layer_floorplan: LayerSpec = "FLOORPLAN",
    edge_to_pad_distance: float = 50.0,
    edge_to_grating_distance: float = 420.0,
    with_loopback: bool = True,
    pad_size: tuple[float, float] | None = (150.0, 200.0),
    pad_offset_x: float = -115.0,
    loopback_radius: float = 30.0,
) -> gf.Component:
    """A die with grating couplers and pads: the Run-50 full packaging template.

    The defaults reproduce ``Cell0_SOI220_cband_TE_Full_Packaging_Template``: an
    11470 x 4900 um floorplan, 16 C-band strip gratings per side at 250 um pitch
    (y = -1875 ... 1875 um, ports at x = +-4923 um) and 31 pads of 150 x 200 um per
    side at 300 um pitch (x = -4615 ... 4385 um, y = +-2300 um). With
    ``with_loopback`` the outermost gratings of each side are joined by a loopback
    waveguide that runs 70 um behind the gratings, with 30 um radius arcs.

    Args:
        size: the size of the die, in um.
        ngratings: the number of grating couplers per side, loopback ones included.
        npads: the number of pads per side.
        grating_pitch: the pitch of the grating couplers, in um.
        pad_pitch: the pitch of the pads, in um.
        grating_coupler: the grating coupler component. None skips the grating couplers.
        cross_section: the cross section.
        pad: the pad component.
        layer_floorplan: the layer of the floorplan.
        edge_to_pad_distance: the distance from the top/bottom die edge to the pads, in um.
        edge_to_grating_distance: the distance from the left/right die edge to the
            far end of the grating couplers, in um.
        with_loopback: if True, joins the outermost grating couplers of each side
            with a loopback waveguide.
        pad_size: size of the pads, passed to ``pad``. None keeps the pad's own size.
        pad_offset_x: x of the centre of the pad rows, in um.
        loopback_radius: bend radius of the loopback waveguide, in um.
    """
    c = gf.Component()
    _ = c << gf.c.rectangle(
        size=size, layer=layer_floorplan, centered=True, port_type=None
    )
    xs, ys = size
    if grating_coupler and ngratings:
        gc = _get_band_component(grating_coupler, cross_section)
        x_port = xs / 2 - edge_to_grating_distance - gc.xsize
        loopback = with_loopback and ngratings > 1
        for side, rotation, x in (("W", 180, -x_port), ("E", 0, x_port)):
            for i in range(ngratings):
                ref = c << gc
                ref.rotate(rotation)
                y = (i - (ngratings - 1) / 2) * grating_pitch
                ref.move(ref.ports["o1"].center, (x, y))
                if not (loopback and i in (0, ngratings - 1)):
                    c.add_port(f"{side}{i}", port=ref.ports["o1"])
        if loopback:
            span = (ngratings - 1) * grating_pitch
            path = _loopback_path(gc.xsize, span, loopback_radius, 10.0, 70.0)
            # Drawn into the die (not as an anonymous child cell) so the cell and
            # its netlist only reference named PDK cells.
            loop = path.extrude(cross_section=cross_section)
            west = c << loop
            west.move((-x_port, -span / 2))
            east = c << loop
            east.dmirror_x()
            east.move((x_port, -span / 2))
            west.flatten()
            east.flatten()
    pad = gf.get_component(pad, size=pad_size) if pad_size else gf.get_component(pad)
    for i in range(npads):
        x = pad_offset_x + (i - (npads - 1) / 2) * pad_pitch
        top = c << pad
        top.x = x
        top.ymax = ys / 2 - edge_to_pad_distance
        c.add_port(name=f"N{i}", port=top.ports["e4"])
    for i in range(npads):
        x = pad_offset_x + (i - (npads - 1) / 2) * pad_pitch
        bottom = c << pad
        bottom.x = x
        bottom.ymin = -ys / 2 + edge_to_pad_distance
        c.add_port(name=f"S{i}", port=bottom.ports["e2"])
    c.auto_rename_ports()
    return c


##############################
# Run-50 foundry cells
##############################


@gf.cell(tags=["die"])
def Cell0_SOI220_cband_TE_Full_Packaging_Template() -> gf.Component:
    """Run-50 full-die packaging template: gratings, loopbacks, pads and outline."""
    return die_with_pads()


@gf.cell(tags=["die"])
def Cell0_SOI_Full_Institution() -> gf.Component:
    """Run-50 full design area outline (GDS 99), 11470 x 4900 um."""
    return die()


@gf.cell(tags=["die"])
def Cell0_SOI_Half_Institution() -> gf.Component:
    """Run-50 half design area outline (GDS 99), 5500 x 4900 um."""
    return die_half()
