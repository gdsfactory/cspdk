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
from gdsfactory.typings import ComponentSpec, Layer, LayerSpec

from cspdk.ge_on_si.config import PATH

nm = 1e-3


class LayerMapCornerstone(LayerMap):
    """Layer mapping for Cornerstone Ge-on-Si technology."""

    WG: Layer = (303, 0)
    WG_DF: Layer = (304, 0)
    BLEED: Layer = (98, 0)
    FLOORPLAN: Layer = (99, 0)
    LBL: Layer = (100, 0)

    LABEL_SETTINGS: Layer = (100, 0)
    LABEL_INSTANCE: Layer = (101, 0)
    routing_error_marker: Layer = (1000, 0)


LAYER = LayerMapCornerstone


def get_layer_stack(
    thickness_slab: float = 1200 * nm,
    thickness_wg: float = 3000 * nm,
    trench_width: float = 20.0,
) -> LayerStack:
    """Returns LayerStack.

    The wafer is a 3 um Ge film on a Si substrate with a single 1.8 um partial
    Ge etch (MPW-4/MPW-7 sec. 4). The etch is NOT blanket: per sec. 5.2 and 8
    the foundry grows layer 303 by 20 um, subtracts 303 to form the trenches,
    and merges in the dark-field layer 304 and the label layer 100. So:

    - core: drawn 303 not etched by 304/100, full Ge thickness.
    - slab: the etched regions (trenches + 304 + 100), 1.2 um Ge left.
    - field: everything else inside the cell outline (99) keeps the full
      3 um Ge. It only renders where a floorplan (e.g. ``die``) is drawn,
      since the unetched wafer has no drawn layer of its own.

    The Si substrate is not drawn on any layer and is not modelled here.
    ``derived_layer`` (where simulators/3D export put each level's shapes)
    reuses 303 (core), 304 (etched, slab) and 99 (field).

    Args:
        thickness_slab: Ge thickness left in etched regions in um.
        thickness_wg: unetched Ge film thickness in um.
        trench_width: foundry trench width around layer 303 in um.
    """
    wg = LogicalLayer(layer=LAYER.WG)
    etched_by_mask = LogicalLayer(layer=LAYER.WG_DF) + LogicalLayer(layer=LAYER.LBL)
    wg_grown = wg.sized(round(trench_width / nm))
    trench = wg_grown - wg
    return LayerStack(
        layers=dict(
            core=LayerLevel(
                layer=wg - etched_by_mask,
                thickness=thickness_wg,
                zmin=0.0,
                material="Ge",
                info={"mesh_order": 1},
                # sidewall_angle/width_to_z are not given by Cornerstone;
                # kept from the original PDK pending foundry confirmation.
                sidewall_angle=10,
                width_to_z=0.5,
                derived_layer=wg,
            ),
            slab=LayerLevel(
                layer=trench + etched_by_mask,
                thickness=thickness_slab,
                zmin=0.0,
                material="Ge",
                info={"mesh_order": 2},
                derived_layer=LogicalLayer(layer=LAYER.WG_DF),
            ),
            field=LayerLevel(
                layer=LogicalLayer(layer=LAYER.FLOORPLAN) - wg_grown - etched_by_mask,
                thickness=thickness_wg,
                zmin=0.0,
                material="Ge",
                info={"mesh_order": 3},
                derived_layer=LogicalLayer(layer=LAYER.FLOORPLAN),
            ),
        )
    )


LAYER_STACK = get_layer_stack()
LAYER_VIEWS = gf.technology.LayerViews(PATH.lyp_yaml)


class Tech:
    """Technology parameters."""

    radius_rib = 300
    width_rib = 3.2


TECH = Tech()


############################
# Cross-sections functions
############################
xsection = gf.xsection


@xsection
def xs_rib(
    width: float = TECH.width_rib,
    layer: LayerSpec = "WG",
    radius: float = TECH.radius_rib,
    radius_min: float = TECH.radius_rib,
    **kwargs,
) -> CrossSection:
    """Return Ge rib cross_section for 3800nm TE."""
    return gf.cross_section.cross_section(
        width=width,
        layer=layer,
        radius=radius,
        radius_min=radius_min,
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
    cross_section: CrossSectionSpec = "xs_rib",
    straight: ComponentSpec = "straight",
    bend: ComponentSpec = "bend_circular",
) -> ManhattanRoute:
    """Route two ports with a single route.

    Bends default to ``bend_circular``, the foundry's r=300 um circular 90 deg
    bend (Ge_on_Si_3800nm_TE_RIB_90_Degree_Bend).
    """
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
    cross_section: CrossSectionSpec = "xs_rib",
    straight: ComponentSpec = "straight",
    bend: ComponentSpec = "bend_circular",
    taper: ComponentSpec | None = None,
    **kwargs,
) -> list[ManhattanRoute]:
    """Route two bundles of ports.

    Bends default to ``bend_circular`` (the foundry bend). ``taper`` defaults to
    None: the PDK has a single 3.2 um rib width, so long-straight tapers would be
    3.2 -> 3.2 um no-ops.
    """
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


# Required by the pdk-ci check-tech-structure hook.
cross_sections = get_cross_sections(sys.modules[__name__])


routing_strategies = dict(
    route_single=route_single,
    route_bundle=route_bundle,
)


if __name__ == "__main__":
    from gdsfactory.technology.klayout_tech import KLayoutTechnology

    LAYER_VIEWS = LayerViews(PATH.lyp_yaml)

    t = KLayoutTechnology(
        name="Cornerstone_ge_on_si",
        layer_map=LAYER,
        layer_views=LAYER_VIEWS,
        layer_stack=LAYER_STACK,
    )
    t.write_tech(tech_dir=PATH.klayout)
