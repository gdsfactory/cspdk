"""Technology definitions."""

import sys
from collections.abc import Iterable
from functools import partial
from typing import cast

import gdsfactory as gf
from gdsfactory.cross_section import CrossSectionSpec, LayerSpec, get_cross_sections
from gdsfactory.routing.route_bundle import ManhattanRoute
from gdsfactory.technology import (
    LayerLevel,
    LayerMap,
    LayerStack,
    LayerViews,
    LogicalLayer,
)
from gdsfactory.typings import ComponentSpec, ConnectivitySpec, Layer

from cspdk.si500.config import PATH

nm = 1e-3


class LayerMapCornerstone(LayerMap):
    """Layer mapping for Cornerstone technology."""

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
    thickness_wg: float = 500 * nm,
    thickness_slab: float = 200 * nm,
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

    radius_rc = 25
    radius_ro = 25
    width_rc = 0.45
    width_ro = 0.40


TECH = Tech()


############################
# Cross-sections functions
############################

# will be filled after all cross sections are defined:
DEFAULT_CROSS_SECTION_NAMES: dict[str, str] = {}


def xs_rc(width=Tech.width_rc, radius=Tech.radius_rc, **kwargs) -> gf.CrossSection:
    """Returns a rib Cband cross-section."""
    kwargs["layer"] = kwargs.get("layer", LAYER.WG)
    kwargs["bbox_layers"] = kwargs.get("bbox_layers", (LAYER.SLAB,))
    kwargs["bbox_offsets"] = kwargs.get("bbox_offsets", (5,))
    kwargs["radius_min"] = kwargs.get("radius_min", radius)
    kwargs["sections"] = kwargs.get(
        "sections",
        (gf.Section(width=10.45, layer=LAYER.SLAB, name="slab", simplify=50 * nm),),
    )
    xs = gf.cross_section.strip(width=width, radius=radius, **kwargs)
    if xs.name in DEFAULT_CROSS_SECTION_NAMES:
        xs._name = DEFAULT_CROSS_SECTION_NAMES[xs.name]
    return xs


def xs_rc_tip(width=Tech.width_rc, radius=Tech.radius_rc, **kwargs) -> gf.CrossSection:
    """Returns a rib Cband cross-section."""
    kwargs["layer"] = kwargs.get("layer", LAYER.WG)
    kwargs["bbox_layers"] = kwargs.get("bbox_layers", (LAYER.SLAB,))
    kwargs["bbox_offsets"] = kwargs.get("bbox_offsets", (5,))
    kwargs["radius_min"] = kwargs.get("radius_min", radius)
    kwargs["sections"] = kwargs.get(
        "sections",
        (gf.Section(width=0.2, layer=LAYER.SLAB, name="slab"),),
    )
    xs = gf.cross_section.strip(width=width, radius=radius, **kwargs)
    if xs.name in DEFAULT_CROSS_SECTION_NAMES:
        xs._name = DEFAULT_CROSS_SECTION_NAMES[xs.name]
    return xs


def xs_ro(width=Tech.width_ro, radius=Tech.radius_ro, **kwargs) -> gf.CrossSection:
    """Returns a rib Oband cross-section."""
    kwargs["layer"] = kwargs.get("layer", LAYER.WG)
    kwargs["bbox_layers"] = kwargs.get("bbox_layers", (LAYER.SLAB,))
    kwargs["bbox_offsets"] = kwargs.get("bbox_offsets", (5,))
    kwargs["radius_min"] = kwargs.get("radius_min", radius)
    kwargs["sections"] = kwargs.get(
        "sections",
        (gf.Section(width=10.45, layer=LAYER.SLAB, name="slab", simplify=50 * nm),),
    )
    xs = gf.cross_section.strip(width=width, radius=radius, **kwargs)
    if xs.name in DEFAULT_CROSS_SECTION_NAMES:
        xs._name = DEFAULT_CROSS_SECTION_NAMES[xs.name]
    return xs


def metal_routing(width=10.0, **kwargs) -> gf.CrossSection:
    """Returns a metal routing cross-section."""
    kwargs["layer"] = kwargs.get("layer", LAYER.PAD)
    kwargs["port_names"] = kwargs.get(
        "port_names", gf.cross_section.port_names_electrical
    )
    kwargs["port_types"] = kwargs.get(
        "port_types", gf.cross_section.port_types_electrical
    )
    kwargs["radius"] = kwargs.get("radius", 0)
    kwargs["radius_min"] = kwargs.get("radius_min", kwargs["radius"])
    xs = gf.cross_section.strip_heater_metal(width=width, **kwargs)
    if xs.name in DEFAULT_CROSS_SECTION_NAMES:
        xs._name = DEFAULT_CROSS_SECTION_NAMES[xs.name]
    return xs


def heater_metal(width=4.0, **kwargs) -> gf.CrossSection:
    """Returns a heater metal cross-section."""
    kwargs["layer"] = kwargs.get("layer", LAYER.HEATER)
    xs = metal_routing(width=width, **kwargs).copy()
    if xs.name in DEFAULT_CROSS_SECTION_NAMES:
        xs._name = DEFAULT_CROSS_SECTION_NAMES[xs.name]
    return xs


def populate_default_cross_section_names() -> None:
    """Populates the default cross-section names."""
    xss = {k: v() for k, v in get_cross_sections(sys.modules[__name__]).items()}
    for k, xs in xss.items():
        xs._name = ""
        _k = xs.name
        xs._name = k
        DEFAULT_CROSS_SECTION_NAMES[_k] = xs.name


populate_default_cross_section_names()


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
    cross_section: CrossSectionSpec = "xs_rc",
    straight: ComponentSpec = "straight_rc",
    bend: ComponentSpec = "bend_euler_rc",
    taper: ComponentSpec = "taper_rc",
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
    on_collision: str | None = "show_error",
    bboxes: list | None = None,
    allow_width_mismatch: bool = False,
    radius: float | None = None,
    route_width: float | list[float] | None = None,
    cross_section: CrossSectionSpec = "xs_rc",
    straight: ComponentSpec = "straight_rc",
    bend: ComponentSpec = "bend_euler_rc",
    taper: ComponentSpec = "taper_rc",
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


routing_strategies = dict(
    route_single=route_single,
    route_single_rc=partial(
        route_single,
        straight="straight_rc",
        bend="bend_euler_rc",
        taper="taper_rc",
        cross_section="xs_rc",
    ),
    route_single_ro=partial(
        route_single,
        straight="straight_ro",
        bend="bend_euler_ro",
        taper="taper_ro",
        cross_section="xs_ro",
    ),
    route_bundle=route_bundle,
    route_bundle_rc=partial(
        route_bundle,
        straight="straight_rc",
        bend="bend_euler_rc",
        taper="taper_rc",
        cross_section="xs_rc",
    ),
    route_bundle_ro=partial(
        route_bundle,
        straight="straight_ro",
        bend="bend_euler_ro",
        taper="taper_ro",
        cross_section="xs_ro",
    ),
)


if __name__ == "__main__":
    from gdsfactory.technology.klayout_tech import KLayoutTechnology

    LAYER_VIEWS = LayerViews(PATH.lyp_yaml)
    # LAYER_VIEWS.to_lyp(PATH.lyp)

    connectivity = cast(list[ConnectivitySpec], [("HEATER", "HEATER", "PAD")])

    t = KLayoutTechnology(
        name="Cornerstone_si500",
        layer_map=LAYER,
        layer_views=LAYER_VIEWS,
        layer_stack=LAYER_STACK,
        connectivity=connectivity,
    )
    t.write_tech(tech_dir=PATH.klayout)
