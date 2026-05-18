"""This module contains cells that contain other cells."""

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import (
    ComponentSpec,
    CrossSectionSpec,
    Strs,
)

gc = "grating_coupler_elliptical"

pack_doe = gf.c.pack_doe
pack_doe_grid = gf.c.pack_doe_grid


@gf.cell(tags=["containers"])
def add_fiber_array(
    component: ComponentSpec = "straight",
    grating_coupler=gc,
    gc_port_name: str = "o1",
    cross_section: CrossSectionSpec = "strip",
    **kwargs,
) -> Component:
    """Returns component with south routes and grating_couplers.

    You can also use pads or other terminations instead of grating couplers.

    Args:
        component: component spec to connect to grating couplers.
        grating_coupler: spec for route terminations.
        gc_port_name: grating coupler input port name.
        cross_section: cross_section function.
        kwargs: additional arguments.

    .. plot::
        :include-source:

        import gdsfactory as gf

        c = gf.components.crossing()
        cc = gf.routing.add_fiber_array(
            component=c,
            grating_coupler=gf.components.grating_coupler_elliptical_te,
        )
        cc.plot()

    """
    return gf.routing.add_fiber_array(
        component=component,
        grating_coupler=grating_coupler,
        gc_port_name=gc_port_name,
        cross_section=cross_section,
        **kwargs,
    )


@gf.cell(tags=["containers"])
def add_fiber_single(
    component: ComponentSpec = "straight",
    grating_coupler=gc,
    gc_port_name: str = "o1",
    cross_section: CrossSectionSpec = "strip",
    input_port_names: list[str] | tuple[str, ...] | None = None,
    pitch: float = 70,
    with_loopback: bool = True,
    loopback_spacing: float = 100.0,
    **kwargs,
) -> Component:
    """Returns component with south routes and grating_couplers.

    You can also use pads or other terminations instead of grating couplers.

    Args:
        component: component spec to connect to grating couplers.
        grating_coupler: spec for route terminations.
        gc_port_name: grating coupler input port name.
        cross_section: cross_section function.
        input_port_names: list of input port names to connect to grating couplers.
        pitch: spacing between fibers.
        with_loopback: adds loopback structures.
        loopback_spacing: spacing between loopback and test structure.
        kwargs: additional arguments.

    .. plot::
        :include-source:

        import gdsfactory as gf

        c = gf.components.crossing()
        cc = gf.routing.add_fiber_single(
            component=c,
            grating_coupler=gf.components.grating_coupler_elliptical_te,
        )
        cc.plot()

    """
    return gf.routing.add_fiber_single(
        component=component,
        grating_coupler=grating_coupler,
        gc_port_name=gc_port_name,
        cross_section=cross_section,
        input_port_names=input_port_names,
        pitch=pitch,
        with_loopback=with_loopback,
        loopback_spacing=loopback_spacing,
        **kwargs,
    )


@gf.cell(tags=["containers"])
def add_pads_top(
    component: ComponentSpec = "straight_metal",
    port_names: Strs | None = None,
    cross_section: CrossSectionSpec = "metal_routing",
    pad_port_name: str = "e1",
    pad: ComponentSpec = "pad",
    bend: ComponentSpec = "wire_corner",
    straight_separation: float = 15.0,
    pad_pitch: float = 100.0,
    port_type: str = "electrical",
    allow_width_mismatch: bool = True,
    fanout_length: float | None = 80,
    route_width: float | list[float] = 0,
    **kwargs,
) -> Component:
    """Returns new component with ports connected top pads.

    Args:
        component: component spec to connect to.
        port_names: list of port names to connect to pads.
        cross_section: cross_section function.
        pad_port_name: pad port name.
        pad: pad function.
        bend: bend function.
        straight_separation: from edge to edge.
        pad_pitch: spacing between pads.
        port_type: port type.
        allow_width_mismatch: if True, allows width mismatch.
        fanout_length: length of the fanout.
        route_width: width of the route.
        kwargs: additional arguments.

    .. plot::
        :include-source:

        import gdsfactory as gf
        c = gf.c.nxn(
            xsize=600,
            ysize=200,
            north=2,
            south=3,
            wg_width=10,
            layer="M3",
            port_type="electrical",
        )
        cc = gf.routing.add_pads_top(component=c, port_names=("e1", "e4"), fanout_length=50)
        cc.plot()

    """
    return gf.routing.add_pads_top(
        component=component,
        port_names=port_names,
        cross_section=cross_section,
        pad_port_name=pad_port_name,
        pad=pad,
        bend=bend,
        straight_separation=straight_separation,
        pad_pitch=pad_pitch,
        port_type=port_type,
        allow_width_mismatch=allow_width_mismatch,
        fanout_length=fanout_length,
        route_width=route_width,
        **kwargs,
    )
