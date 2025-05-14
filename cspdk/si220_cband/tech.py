"""Technology definitions."""

from collections.abc import Iterable
from functools import partial

import gdsfactory as gf
from gdsfactory import Section
from gdsfactory.cross_section import (
    CrossSection,
    cross_section,
    port_names_electrical,
    port_types_electrical,
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
    ConnectivitySpec,
    CrossSectionSpec,
    Floats,
    Layer,
    LayerSpec,
    LayerSpecs,
)

from cspdk.si220_cband.config import PATH

nm = 1e-3


class LayerMapCornerstone(LayerMap):
    """Layer map for Cornerstone technology."""

    WG: Layer = (3, 0)  # type: ignore
    SLAB: Layer = (5, 0)  # type: ignore
    FLOORPLAN: Layer = (99, 0)  # type: ignore
    HEATER: Layer = (39, 0)  # type: ignore
    GRA: Layer = (6, 0)  # type: ignore
    LBL: Layer = (100, 0)  # type: ignore
    PAD: Layer = (41, 0)  # type: ignore

    # labels for gdsfactory
    LABEL_SETTINGS: Layer = (100, 0)  # type: ignore
    LABEL_INSTANCE: Layer = (101, 0)  # type: ignore


LAYER = LayerMapCornerstone


def get_layer_stack(
    thickness_wg: float = 220 * nm,
    thickness_slab: float = 100 * nm,
    zmin_heater: float = 1.1,
    thickness_heater: float = 700 * nm,
    zmin_metal: float = 1.1,
    thickness_metal: float = 700 * nm,
) -> LayerStack:
    """Returns LayerStack.

    based on paper https://www.degruyter.com/document/doi/10.1515/nanoph-2013-0034/html

    Args:
        thickness_wg: waveguide thickness in um.
        thickness_slab: slab thickness in um.
        zmin_heater: TiN heater.
        thickness_heater: TiN thickness.
        zmin_metal: metal thickness in um.
        thickness_metal: metal2 thickness.
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
                info={"mesh_order": 1},
                sidewall_angle=10,
                width_to_z=0.5,
            ),
            heater=LayerLevel(
                layer=LogicalLayer(layer=LAYER.HEATER),
                thickness=thickness_heater,
                zmin=zmin_heater,
                material="TiN",
                info={"mesh_order": 1},
            ),
            metal=LayerLevel(
                layer=LogicalLayer(layer=LAYER.PAD),
                thickness=thickness_metal,
                zmin=zmin_metal + thickness_metal,
                material="Aluminum",
                info={"mesh_order": 2},
            ),
        )
    )


LAYER_STACK = get_layer_stack()
LAYER_VIEWS = gf.technology.LayerViews(PATH.lyp_yaml)


class Tech:
    """Technology parameters."""

    radius = 5
    radius_rib = 25
    radius_ro = 25
    width = 0.45
    width_rib = 0.5
    width_ro = 0.5

    width_slab = 5
    width_heater = 2.5


TECH = Tech()

############################
# Cross-sections functions
############################

xsection = gf.xsection


@xsection
def strip(
    width: float = TECH.width,
    layer: LayerSpec = "WG",
    radius: float = TECH.radius,
    radius_min: float = TECH.radius,
) -> CrossSection:
    """Return Strip cross_section."""
    return gf.cross_section.cross_section(
        width=width,
        layer=layer,
        radius=radius,
        radius_min=radius_min,
    )


@xsection
def rib(
    width: float = TECH.width_rib,
    layer: LayerSpec = "WG",
    radius: float = TECH.radius_rib,
    radius_min: float = TECH.radius_rib,
    bbox_layers: LayerSpecs = ("SLAB",),
    bbox_offsets: Floats = (TECH.width_slab,),
    **kwargs,
) -> CrossSection:
    """Return Rib cross_section."""
    return gf.cross_section.cross_section(
        width=width,
        layer=layer,
        radius=radius,
        radius_min=radius_min,
        bbox_layers=bbox_layers,
        bbox_offsets=bbox_offsets,
        **kwargs,
    )


@xsection
def strip_heater_metal(
    width: float = TECH.width,
    layer: LayerSpec = "WG",
    heater_width: float = TECH.width_heater,
    layer_heater: LayerSpec = "HEATER",
) -> CrossSection:
    """Returns strip cross_section with top heater metal."""

    return gf.cross_section.strip_heater_metal(
        width=width,
        layer=layer,
        heater_width=heater_width,
        layer_heater=layer_heater,
    )


@xsection
def metal_routing(
    width: float = 10,
    layer: LayerSpec = "PAD",
    radius: float | None = None,
) -> CrossSection:
    """Return Metal Strip cross_section."""
    return cross_section(
        width=width,
        layer=layer,
        radius=radius,
        port_names=port_names_electrical,
        port_types=port_types_electrical,
    )


@xsection
def heater_metal(width=TECH.width_heater):
    """Heater cross-section."""
    return gf.cross_section.heater_metal(width=width, layer=LAYER.HEATER)


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
    cross_section: CrossSectionSpec = "strip",
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
    separation: float = 3.0,
    sort_ports: bool = False,
    start_straight_length: float = 0.0,
    end_straight_length: float = 0.0,
    min_straight_taper: float = 100.0,
    port_type: str | None = None,
    collision_check_layers: Iterable[LayerSpec] = (),
    on_collision: str | None = None,
    bboxes: list | None = None,
    allow_width_mismatch: bool = False,
    radius: float | None = None,
    route_width: float | list[float] | None = None,
    cross_section: CrossSectionSpec = "strip",
    straight: ComponentSpec = "straight",
    bend: ComponentSpec = "bend_euler",
    taper: ComponentSpec | None = "taper",
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
        collision_check_layers=tuple(collision_check_layers),
        on_collision=on_collision,
        bboxes=bboxes,
        allow_width_mismatch=allow_width_mismatch,
        radius=radius,
        route_width=route_width,
        cross_section=cross_section,
        straight=straight,
        bend=bend,
        taper=taper,
    )


route_single = partial(
    route_single,
    straight="straight",
    bend="bend_euler",
    taper="taper",
    cross_section="strip",
    port_type="optical",
)
route_bundle_rib = partial(
    route_bundle,
    straight="straight_rib",
    bend="bend_euler_rib",
    taper="taper_rib",
    cross_section="rib",
    port_type="optical",
)
route_bundle_metal = partial(
    route_bundle,
    straight="straight_metal",
    bend="bend_metal",
    taper=None,
    cross_section="metal_routing",
    port_type="electrical",
)
route_bundle_metal_corner = partial(
    route_bundle,
    straight="straight_metal",
    bend="wire_corner",
    taper=None,
    cross_section="metal_routing",
    port_type="electrical",
)

routing_strategies = dict(
    route_bundle=route_bundle,
    route_bundle_rib=route_bundle_rib,
    route_bundle_metal=route_bundle_metal,
    route_bundle_metal_corner=route_bundle_metal_corner,
)

if __name__ == "__main__":
    from typing import cast

    from gdsfactory.technology.klayout_tech import KLayoutTechnology

    LAYER_VIEWS = LayerViews(PATH.lyp_yaml)
    # LAYER_VIEWS.to_lyp(PATH.lyp)

    connectivity = cast(list[ConnectivitySpec], [("HEATER", "HEATER", "PAD")])

    t = KLayoutTechnology(
        name="Cornerstone_si220",
        layer_map=LAYER,
        layer_views=LAYER_VIEWS,
        layer_stack=LAYER_STACK,
        connectivity=connectivity,
    )
    t.write_tech(tech_dir=PATH.klayout)
    # print(DEFAULT_CROSS_SECTION_NAMES)
    # print(strip() is strip())
    # print(strip().name, strip().name)
    # c = gf.c.bend_euler(cross_section="metal_routing")
    # c.pprint_ports()
