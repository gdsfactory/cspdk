"""Building blocks for the cspdk.si_sus library."""

import gdsfactory as gf
import numpy as np
from gdsfactory.component_functions.bends.bend_s import bezier_curve
from gdsfactory.typings import CrossSectionSpec, LayerSpec

from cspdk.si_sus._schematic import (
    bend_circular_schematic,
    bend_euler_schematic,
    bend_s_schematic,
    grating_coupler_rectangular_schematic,
    straight_schematic,
    taper_schematic,
)
from cspdk.si_sus.tech import LAYER, Tech, add_tethers


@gf.cell(tags=["cells"], schematic_function=straight_schematic)
def straight(
    length: float = 10.0,
    cross_section: CrossSectionSpec = "xs_sus",
    **kwargs,
) -> gf.Component:
    """A straight waveguide.

    Args:
        length: the length of the waveguide.
        cross_section: a cross section or its name or a function generating a cross section.
        kwargs: additional arguments to pass to the straight function.
    """
    c = gf.component_functions.straight(
        length=length, cross_section=cross_section, **kwargs
    )
    path = gf.path.straight(length=length, npoints=kwargs.get("npoints", 2))
    add_tethers(c, path, cross_section)
    return c


@gf.cell(tags=["cells"], schematic_function=bend_s_schematic)
def bend_s(
    size: tuple[float, float] = (20.0, 1.8),
    cross_section: CrossSectionSpec = "xs_sus",
    allow_min_radius_violation: bool = True,
) -> gf.Component:
    """An S-bend.

    Args:
        size: the width and height of the s-bend.
        cross_section: a cross section or its name or a function generating a cross section.
        allow_min_radius_violation: if True, allows the s-bend to have a smaller radius than the minimum radius.
    """
    dx, dy = size
    if dy == 0:
        return straight(length=dx, cross_section=cross_section)
    control_points = ((0, 0), (dx / 2, 0), (dx / 2, dy), (dx, dy))
    npoints = 99
    c = gf.component_functions.bezier(
        control_points=control_points,
        npoints=npoints,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
    )
    path = gf.Path(bezier_curve(np.linspace(0, 1, npoints), control_points))
    add_tethers(c, path, cross_section)
    return c


@gf.cell(tags=["cells"], schematic_function=bend_euler_schematic)
def bend_euler(
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.5,
    width: float | None = None,
    cross_section: CrossSectionSpec = "xs_sus",
) -> gf.Component:
    """An euler bend.

    Args:
        radius: the effective radius of the bend.
        angle: the angle of the bend (usually 90 degrees).
        p: the fraction of the bend that's represented by a polar bend.
        width: the width of the waveguide forming the bend.
        cross_section: a cross section or its name or a function generating a cross section.
    """
    c = gf.component_functions.bend_euler(
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
    path = gf.path.euler(
        radius=radius or gf.get_cross_section(cross_section).radius,
        angle=angle,
        p=p,
        use_eff=True,
        npoints=None,
    )
    add_tethers(c, path, cross_section)
    return c


@gf.cell(tags=["cells"], schematic_function=bend_circular_schematic)
def bend_circular(
    radius: float | None = None,
    angle: float = 90.0,
    width: float | None = None,
    cross_section: CrossSectionSpec = "xs_sus",
) -> gf.Component:
    """A circular bend, matching the foundry reference bend geometry.

    The Suspendedsilicon500nm_3800nm_TE_90_DegreeBend reference GDS is a
    circular arc with a 40.75um center-line radius (the xs_sus default).

    Args:
        radius: the radius of the bend (defaults to the cross-section radius).
        angle: the angle of the bend (usually 90 degrees).
        width: the width of the waveguide forming the bend.
        cross_section: a cross section or its name or a function generating a cross section.
    """
    c = gf.component_functions.bend_circular(
        radius=radius,
        angle=angle,
        width=width,
        cross_section=cross_section,
        allow_min_radius_violation=False,
    )
    path = gf.path.arc(
        radius=radius or gf.get_cross_section(cross_section).radius, angle=angle
    )
    add_tethers(c, path, cross_section)
    return c


@gf.cell(tags=["cells"], schematic_function=taper_schematic)
def taper(
    length: float = 10.0,
    width1: float = Tech.width_sus,
    width2: float | None = None,
    port: gf.Port | None = None,
    cross_section: CrossSectionSpec = "xs_sus",
    layer: LayerSpec | None = None,
    port_names: tuple[str, str] = ("o1", "o2"),
    port_types: tuple[str, str] = ("optical", "optical"),
    with_bbox: bool = True,
) -> gf.Component:
    """A taper.

    Args:
        length: the length of the taper.
        width1: the input width of the taper.
        width2: the output width of the taper (if not given, use port).
        port: the port (with certain width) to taper towards (if not given, use width2).
        cross_section: a cross section or its name or a function generating a cross section.
        layer: layer of the taper. Defaults to the cross section's main layer.
        port_names: input and output port names.
        port_types: input and output port types.
        with_bbox: whether to draw the cross section's bbox layers.
    """
    return gf.c.taper(
        length=length,
        width1=width1,
        width2=width2,
        port=port,
        cross_section=cross_section,
        layer=layer,
        port_names=port_names,
        port_types=port_types,
        with_bbox=with_bbox,
    )


@gf.cell(tags=["cells"], schematic_function=grating_coupler_rectangular_schematic)
def grating_coupler_rectangular(
    period=2.5,
    n_periods: int = 20,
    length_taper: float = 350.0,
    wavelength: float = 3.8,
    cross_section="xs_sus",
) -> gf.Component:
    """A grating coupler with straight and parallel teeth.

    Note: not matched to the foundry reference GDS
    (Suspendedsilicon500nm_3800nm_TE_GratingCoupler); the generic gdsfactory
    grating with layer_slab=WG only approximates the etch-slot drawing
    convention of this platform.

    Args:
        period: the period of the grating.
        n_periods: the number of grating teeth.
        length_taper: the length of the taper tapering up to the grating.
        wavelength: the center wavelength for which the grating is designed.
        cross_section: a cross section or its name or a function generating a cross section.
    """
    return gf.c.grating_coupler_rectangular(
        n_periods=n_periods,
        period=period,
        fill_factor=0.5,
        width_grating=20.0,
        length_taper=length_taper,
        polarization="te",
        wavelength=wavelength,
        taper=taper,
        layer_slab=LAYER.WG,
        fiber_angle=10.0,
        slab_xmin=-1.0,
        slab_offset=0.0,
        cross_section=cross_section,
    )


@gf.cell(tags=["cells"])
def rectangle(layer=LAYER.FLOORPLAN, **kwargs) -> gf.Component:
    """A rectangle.

    Args:
        layer: LAYER.FLOORPLAN.
        **kwargs: additional arguments.
    """
    return gf.c.rectangle(layer=layer, **kwargs)


@gf.cell(tags=["cells"])
def array(
    component="straight",
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
        component: the component of which to create an array.
        columns: the number of components to place in the x-direction.
        rows: the number of components to place in the y-direction.
        add_ports: add ports to the component.
        size: Optional x, y size. Overrides columns and rows.
        centered: center the array around the origin.
        column_pitch: the pitch between columns.
        row_pitch: the pitch between rows.
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
