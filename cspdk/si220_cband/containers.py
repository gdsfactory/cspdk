"""This module contains cells that contain other cells."""

from typing import Any

import gdsfactory as gf
from gdsfactory.port import select_ports_electrical, select_ports_optical
from gdsfactory.typings import (
    BoundingBoxes,
    ComponentSpec,
    ComponentSpecOrList,
    CrossSectionSpec,
    Strs,
)

_grating_coupler_silicon = "AMF_PSOI_SiGC1D_Cband_v3p0"
_grating_coupler_nitride = "AMF_PSOI_PSiNGC1D1_Cband_preview"

pack_doe = gf.c.pack_doe
pack_doe_grid = gf.c.pack_doe_grid


@gf.cell
def add_fiber_array_strip(
    component: ComponentSpec = "straight_strip",
    grating_coupler: ComponentSpecOrList = _grating_coupler_silicon,
    cross_section: CrossSectionSpec = "strip",
    start_straight_length: float = 0,
    end_straight_length: float = 0,
    **kwargs: Any,
) -> gf.Component:
    """Returns component with south routes and grating_couplers.

    You can also use pads or other terminations instead of grating couplers.

    Args:
        component: component spec to connect to grating couplers.
        grating_coupler: spec for route terminations.
        cross_section: cross_section function.
        start_straight_length: length of the straight section before the grating coupler.
        end_straight_length: length of the straight section after the grating coupler.
        kwargs: additional arguments.
    """
    from gdsfactory.routing.add_fiber_array import add_fiber_array

    return add_fiber_array(
        component=component,
        grating_coupler=grating_coupler,
        cross_section=cross_section,
        gc_port_name="o1",
        select_ports=select_ports_optical,
        start_straight_length=start_straight_length,
        end_straight_length=end_straight_length,
        **kwargs,
    )


@gf.cell
def add_fiber_array_nitride(
    component: ComponentSpec = "straight_nitride",
    grating_coupler: ComponentSpecOrList = _grating_coupler_nitride,
    cross_section: CrossSectionSpec = "nitride",
    start_straight_length: float = 0,
    end_straight_length: float = 0,
    **kwargs: Any,
) -> gf.Component:
    """Returns component with south routes and grating_couplers.

    You can also use pads or other terminations instead of grating couplers.

    Args:
        component: component spec to connect to grating couplers.
        grating_coupler: spec for route terminations.
        cross_section: cross_section function.
        start_straight_length: length of the straight section before the grating coupler.
        end_straight_length: length of the straight section after the grating coupler.
        kwargs: additional arguments.
    """
    from gdsfactory.routing.add_fiber_array import add_fiber_array

    return add_fiber_array(
        component=component,
        grating_coupler=grating_coupler,
        cross_section=cross_section,
        gc_port_name="o1",
        select_ports=select_ports_optical,
        start_straight_length=start_straight_length,
        end_straight_length=end_straight_length,
        **kwargs,
    )


@gf.cell
def add_fiber_single_strip(
    component: ComponentSpec = "straight_strip",
    grating_coupler: ComponentSpecOrList = _grating_coupler_silicon,
    cross_section: CrossSectionSpec = "strip",
    with_loopback: bool = True,
    pitch: float = 70,
    loopback_spacing: float = 100.0,
    **kwargs: Any,
) -> gf.Component:
    """Returns component with south routes and grating_couplers.

    You can also use pads or other terminations instead of grating couplers.

    Args:
        component: component spec to connect to grating couplers.
        grating_coupler: spec for route terminations.
        cross_section: cross_section function.
        with_loopback: adds loopback structures.
        pitch: in um. Defaults to 70um.
        loopback_spacing: in um. Defaults to 100um.
        kwargs: additional arguments.
    """
    from gdsfactory.routing.add_fiber_single import add_fiber_single

    return add_fiber_single(
        component=component,
        grating_coupler=grating_coupler,
        cross_section=cross_section,
        with_loopback=with_loopback,
        gc_port_name="o1",
        gc_port_name_fiber="o2",
        select_ports=select_ports_optical,
        input_port_names=None,
        straight="straight",
        pitch=pitch,
        loopback_spacing=loopback_spacing,
        **kwargs,
    )


@gf.cell
def add_fiber_single_nitride(
    component: ComponentSpec = "straight_nitride",
    grating_coupler: ComponentSpecOrList = _grating_coupler_nitride,
    cross_section: CrossSectionSpec = "nitride",
    with_loopback: bool = True,
    pitch: float = 70,
    loopback_spacing: float = 100.0,
    **kwargs: Any,
) -> gf.Component:
    """Returns component with south routes and grating_couplers.

    You can also use pads or other terminations instead of grating couplers.

    Args:
        component: component spec to connect to grating couplers.
        grating_coupler: spec for route terminations.
        cross_section: cross_section function.
        with_loopback: adds loopback structures.
        pitch: in um. Defaults to 70um.
        loopback_spacing: in um. Defaults to 100um.
        kwargs: additional arguments.
    """
    from gdsfactory.routing.add_fiber_single import add_fiber_single

    return add_fiber_single(
        component=component,
        grating_coupler=grating_coupler,
        cross_section=cross_section,
        with_loopback=with_loopback,
        gc_port_name="o1",
        gc_port_name_fiber="o2",
        select_ports=select_ports_optical,
        input_port_names=None,
        straight="straight",
        pitch=pitch,
        loopback_spacing=loopback_spacing,
        **kwargs,
    )


@gf.cell
def add_pads(
    component: ComponentSpec = "straight_heater_metal_undercut",
    port_names: Strs | None = None,
    pad: ComponentSpec = "pad",
    straight_separation: float = 15,
    pad_pitch: float = 150,
    allow_width_mismatch: bool = True,
    fanout_length: float | None = 80,
    route_width: float | None = 0,
    bboxes: BoundingBoxes | None = None,
    avoid_component_bbox: bool = False,
    **kwargs: Any,
) -> gf.Component:
    """Returns new component with ports connected top pads.

    Args:
        component: component spec to connect to.
        port_names: optional port names. Overrides select_ports.
        get_input_labels_function: function to get input labels. None skips labels.
        layer_label: optional layer for grating coupler label.
        pad_port_labels: pad list of labels.
        pad: spec for route terminations.
        straight_separation: from wire edge to edge. Defaults to xs.width+xs.gap
        pad_pitch: in um. Defaults to pad_pitch constant from the PDK.
        allow_width_mismatch: True
        fanout_length: if None, automatic calculation of fanout length.
        route_width: width of the route. If None, defaults to cross_section.width.
        bboxes: list of bounding boxes to avoid.
        avoid_component_bbox: True
        kwargs: additional arguments.
    """
    from gdsfactory.routing.add_pads import add_pads_top

    return add_pads_top(
        component=component,
        port_names=port_names,
        pad=pad,
        straight_separation=straight_separation,
        pad_pitch=pad_pitch,
        allow_width_mismatch=allow_width_mismatch,
        fanout_length=fanout_length,
        route_width=route_width,
        bboxes=bboxes,
        avoid_component_bbox=avoid_component_bbox,
        select_ports=select_ports_electrical,
        cross_section="metal_routing",
        pad_port_name="e1",
        bend="wire_corner",
        port_type="electrical",
        **kwargs,
    )


@gf.cell
def add_pads_bottom(
    component: ComponentSpec = "straight_heater_metal_undercut",
    port_names: Strs | None = None,
    pad: ComponentSpec = "pad",
    straight_separation: float = 15,
    pad_pitch: float = 150,
    allow_width_mismatch: bool = True,
    fanout_length: float | None = 80,
    route_width: float | None = 0,
    bboxes: BoundingBoxes | None = None,
    avoid_component_bbox: bool = False,
    **kwargs: Any,
) -> gf.Component:
    """Returns new component with ports connected bottom pads.

    Args:
        component: component spec to connect to.
        port_names: optional port names. Overrides select_ports.
        get_input_labels_function: function to get input labels. None skips labels.
        layer_label: optional layer for grating coupler label.
        pad_port_labels: pad list of labels.
        pad: spec for route terminations.
        straight_separation: from wire edge to edge. Defaults to xs.width+xs.gap
        pad_pitch: in um. Defaults to pad_pitch constant from the PDK.
        allow_width_mismatch: True
        fanout_length: if None, automatic calculation of fanout length.
        route_width: width of the route. If None, defaults to cross_section.width.
        bboxes: list bounding boxes to avoid for routing.
        avoid_component_bbox: avoid component bbox for routing.
        kwargs: additional arguments.
    """
    from gdsfactory.routing.add_pads import add_pads_bot

    return add_pads_bot(
        component=component,
        port_names=port_names,
        pad=pad,
        straight_separation=straight_separation,
        pad_pitch=pad_pitch,
        allow_width_mismatch=allow_width_mismatch,
        fanout_length=fanout_length,
        route_width=route_width,
        bboxes=bboxes,
        avoid_component_bbox=avoid_component_bbox,
        select_ports=select_ports_electrical,
        cross_section="metal_routing",
        pad_port_name="e1",
        bend="wire_corner",
        port_type="electrical",
        **kwargs,
    )


if __name__ == "__main__":
    from amf.cband import PDK

    PDK.activate()
    c = add_pads_bottom()
    c.pprint_ports()
    c.show()
