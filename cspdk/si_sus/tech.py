"""Technology definitions."""

import sys

import gdsfactory as gf
from gdsfactory.cross_section import (
    CrossSection,
    CrossSectionSpec,
    get_cross_sections,
)
from gdsfactory.routing.route_bundle import ManhattanRoute
from gdsfactory.technology import (
    LayerLevel,
    LayerMap,
    LayerStack,
    LayerViews,
    LogicalLayer,
)
from gdsfactory.typings import (
    ComponentSpec,
    Floats,
    Layer,
    LayerSpec,
    LayerSpecs,
)

from cspdk.si_sus.config import PATH

nm = 1e-3


class LayerMapCornerstone(LayerMap):
    """Layer mapping for Cornerstone Suspended Si technology."""

    WG: Layer = (404, 0)
    SLAB: Layer = (405, 0)
    FLOORPLAN: Layer = (99, 0)
    LBL: Layer = (100, 0)

    LABEL_SETTINGS: Layer = (100, 0)
    LABEL_INSTANCE: Layer = (101, 0)
    routing_error_marker: Layer = (1000, 0)


LAYER = LayerMapCornerstone


def get_layer_stack(
    thickness_wg: float = 450 * nm,
    thickness_slab: float = 150 * nm,
) -> LayerStack:
    """Returns LayerStack.

    Args:
        thickness_wg: Si waveguide thickness in um.
        thickness_slab: Si slab thickness in um.
    """
    return LayerStack(
        layers=dict(
            core=LayerLevel(
                layer=LogicalLayer(layer=LAYER.WG),
                thickness=thickness_wg,
                zmin=0.0,
                material="si",
                info={"mesh_order": 1},
                sidewall_angle=10,
                width_to_z=0.5,
            ),
            slab=LayerLevel(
                layer=LogicalLayer(layer=LAYER.SLAB),
                thickness=thickness_slab,
                zmin=0.0,
                material="si",
                info={"mesh_order": 2},
                sidewall_angle=10,
                width_to_z=0.5,
            ),
        )
    )


LAYER_STACK = get_layer_stack()
LAYER_VIEWS = gf.technology.LayerViews(PATH.lyp_yaml)


class Tech:
    """Technology parameters."""

    radius_sus = 20
    width_sus = 8.5


TECH = Tech()


############################
# Cross-sections functions
############################
xsection = gf.xsection


@xsection
def xs_sus(
    width: float = TECH.width_sus,
    layer: LayerSpec = "WG",
    radius: float = TECH.radius_sus,
    radius_min: float = TECH.radius_sus,
    bbox_layers: LayerSpecs = ("SLAB",),
    bbox_offsets: Floats = (5.0,),
    **kwargs,
) -> CrossSection:
    """Return Suspended Si cross_section for 3800nm TE."""
    return gf.cross_section.cross_section(
        width=width,
        layer=layer,
        radius=radius,
        radius_min=radius_min,
        bbox_layers=bbox_layers,
        bbox_offsets=bbox_offsets,
        **kwargs,
    )


############################
# Routing functions
############################


def route_single(
    component: gf.Component,
    port1: gf.Port,
    port2: gf.Port,
    start_straight_length: float = 0.0,
    end_straight_length: float = 0.0,
    waypoints: list[tuple[float, float]] | None = None,
    port_type: str | None = None,
    allow_width_mismatch: bool = False,
    radius: float | None = None,
    route_width: float | None = None,
    cross_section: CrossSectionSpec = "xs_sus",
    straight: ComponentSpec = "straight",
    bend: ComponentSpec = "bend_euler",
    taper: ComponentSpec = "taper",
) -> ManhattanRoute:
    """Route two ports with a single route."""
    return gf.routing.route_single(
        component=component,
        port1=port1,
        port2=port2,
        start_straight_length=start_straight_length,
        end_straight_length=end_straight_length,
        cross_section=cross_section,
        waypoints=waypoints,
        port_type=port_type,
        allow_width_mismatch=allow_width_mismatch,
        radius=radius,
        route_width=route_width,
        straight=straight,
        bend=bend,
        taper=taper,
    )


def route_bundle(
    component: gf.Component,
    ports1: list[gf.Port],
    ports2: list[gf.Port],
    separation: float = 10.0,
    sort_ports: bool = False,
    start_straight_length: float = 0.0,
    end_straight_length: float = 0.0,
    min_straight_taper: float = 100.0,
    port_type: str | None = None,
    cross_section: CrossSectionSpec = "xs_sus",
    straight: ComponentSpec = "straight",
    bend: ComponentSpec = "bend_euler",
    taper: ComponentSpec = "taper",
    **kwargs,
) -> list[ManhattanRoute]:
    """Route two bundles of ports."""
    return gf.routing.route_bundle(
        component=component,
        ports1=ports1,
        ports2=ports2,
        separation=separation,
        sort_ports=sort_ports,
        start_straight_length=start_straight_length,
        end_straight_length=end_straight_length,
        min_straight_taper=min_straight_taper,
        port_type=port_type,
        cross_section=cross_section,
        straight=straight,
        bend=bend,
        taper=taper,
        sbend="bend_s",
        **kwargs,
    )


cross_sections = get_cross_sections(sys.modules[__name__])


routing_strategies = dict(
    route_single=route_single,
    route_bundle=route_bundle,
)


if __name__ == "__main__":
    from gdsfactory.technology.klayout_tech import KLayoutTechnology

    LAYER_VIEWS = LayerViews(PATH.lyp_yaml)

    t = KLayoutTechnology(
        name="Cornerstone_si_sus",
        layer_map=LAYER,
        layer_views=LAYER_VIEWS,
        layer_stack=LAYER_STACK,
    )
    t.write_tech(tech_dir=PATH.klayout)
