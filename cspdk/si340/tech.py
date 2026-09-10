"""Technology definitions."""

import sys
from collections.abc import Iterable
from functools import partial
from typing import cast

import gdsfactory as gf
from gdsfactory.cross_section import (
    CrossSection,
    cross_section,
    get_cross_sections,
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

from cspdk.si340.config import PATH

nm = 1e-3


class LayerMapCornerstone(LayerMap):
    """Layer mapping for Cornerstone Si340 technology."""

    WG: Layer = (3, 0)  # type: ignore
    WG_DF: Layer = (4, 0)  # type: ignore
    SLAB: Layer = (5, 0)  # type: ignore
    GRA: Layer = (6, 0)  # type: ignore
    HEATER: Layer = (39, 0)  # type: ignore
    PAD: Layer = (41, 0)  # type: ignore
    ISOLATION: Layer = (46, 0)  # type: ignore
    FLOORPLAN: Layer = (99, 0)  # type: ignore
    LBL: Layer = (100, 0)  # type: ignore

    # labels for gdsfactory
    LABEL_SETTINGS: Layer = (100, 0)  # type: ignore
    LABEL_INSTANCE: Layer = (101, 0)  # type: ignore
    routing_error_marker: Layer = (1000, 0)  # type: ignore


LAYER = LayerMapCornerstone


def get_layer_stack(
    thickness_wg: float = 340 * nm,
    thickness_slab: float = 200 * nm,
    thickness_grating: float = 200 * nm,
    zmin_heater: float = 1.1,
    thickness_heater: float = 150 * nm,
    zmin_metal: float = 1.1,
    thickness_metal: float = 220 * nm,
) -> LayerStack:
    """Returns LayerStack.

    Args:
        thickness_wg: waveguide thickness in um.
        thickness_slab: slab thickness in um.
        thickness_grating: grating residual Si thickness in um.
        zmin_heater: TiN heater.
        thickness_heater: TiN thickness.
        zmin_metal: metal thickness in um.
        thickness_metal: top metal thickness.
    """
    return LayerStack(
        layers=dict(
            core=LayerLevel(
                layer=LogicalLayer(layer=LAYER.WG) - LogicalLayer(layer=LAYER.GRA),
                thickness=thickness_wg,
                zmin=0.0,
                material="Si",
                info={"mesh_order": 1},
                sidewall_angle=10,
                width_to_z=0.5,
                derived_layer=LogicalLayer(layer=LAYER.WG),
            ),
            grating=LayerLevel(
                layer=LogicalLayer(layer=LAYER.WG) & LogicalLayer(layer=LAYER.GRA),
                thickness=thickness_grating,
                zmin=0.0,
                material="Si",
                info={"mesh_order": 1},
                sidewall_angle=10,
                width_to_z=0.5,
                derived_layer=LogicalLayer(layer=LAYER.GRA),
            ),
            slab=LayerLevel(
                layer=LogicalLayer(layer=LAYER.SLAB),
                thickness=thickness_slab,
                zmin=0.0,
                material="Si",
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
                material="Al",
                info={"mesh_order": 2},
            ),
        )
    )


LAYER_STACK = get_layer_stack()
LAYER_VIEWS = gf.technology.LayerViews(PATH.lyp_yaml)


class Tech:
    """Technology parameters."""

    radius_sc = 10
    radius_so = 10
    radius_rc = 25
    width_sc = 0.45
    width_so = 0.40
    width_rc = 0.80

    width_slab = 5


TECH = Tech()


############################
# Cross-sections functions
############################
xsection = gf.xsection


@xsection
def xs_sc(
    width: float = TECH.width_sc,
    layer: LayerSpec = "WG",
    radius: float = TECH.radius_sc,
    radius_min: float = TECH.radius_sc,
    **kwargs,
) -> CrossSection:
    """Return Strip C-band cross_section."""
    return gf.cross_section.cross_section(
        width=width,
        layer=layer,
        radius=radius,
        radius_min=radius_min,
        **kwargs,
    )


@xsection
def xs_so(
    width: float = TECH.width_so,
    layer: LayerSpec = "WG",
    radius: float = TECH.radius_so,
    radius_min: float = TECH.radius_so,
    **kwargs,
) -> CrossSection:
    """Return Strip O-band cross_section."""
    return gf.cross_section.cross_section(
        width=width,
        layer=layer,
        radius=radius,
        radius_min=radius_min,
        **kwargs,
    )


@xsection
def xs_rc(
    width: float = TECH.width_rc,
    layer: LayerSpec = "WG",
    radius: float = TECH.radius_rc,
    radius_min: float = TECH.radius_rc,
    bbox_layers: LayerSpecs = ("SLAB",),
    bbox_offsets: Floats = (TECH.width_slab,),
    **kwargs,
) -> CrossSection:
    """Return Rib C-band cross_section."""
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
def heater_metal(
    width: float = 2.5,
    layer: LayerSpec = "HEATER",
    radius: float | None = None,
    port_names=port_names_electrical,
    port_types=port_types_electrical,
    **kwargs,
) -> CrossSection:
    """Return Heater metal cross_section."""
    return gf.cross_section.cross_section(
        width=width,
        layer=layer,
        radius=radius,
        port_names=port_names,
        port_types=port_types,
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
    cross_section: CrossSectionSpec = "xs_sc",
    straight: ComponentSpec = "straight_sc",
    bend: ComponentSpec = "bend_euler_sc",
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
    cross_section: CrossSectionSpec = "xs_sc",
    straight: ComponentSpec = "straight_sc",
    bend: ComponentSpec = "bend_euler_sc",
    taper: ComponentSpec = "taper_sc",
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
        sbend="bend_s",
    )


cross_sections = get_cross_sections(sys.modules[__name__])


routing_strategies = dict(
    route_single=route_single,
    route_single_sc=partial(
        route_single,
        straight="straight_sc",
        bend="bend_euler_sc",
        cross_section="xs_sc",
    ),
    route_single_so=partial(
        route_single,
        straight="straight_so",
        bend="bend_euler_so",
        cross_section="xs_so",
    ),
    route_single_rc=partial(
        route_single,
        straight="straight_rc",
        bend="bend_euler_rc",
        cross_section="xs_rc",
    ),
    route_bundle=route_bundle,
    route_bundle_sc=partial(
        route_bundle,
        straight="straight_sc",
        bend="bend_euler_sc",
        taper="taper_sc",
        cross_section="xs_sc",
    ),
    route_bundle_so=partial(
        route_bundle,
        straight="straight_so",
        bend="bend_euler_so",
        taper="taper_so",
        cross_section="xs_so",
    ),
    route_bundle_rc=partial(
        route_bundle,
        straight="straight_rc",
        bend="bend_euler_rc",
        taper="taper_rc",
        cross_section="xs_rc",
    ),
)


if __name__ == "__main__":
    from gdsfactory.technology.klayout_tech import KLayoutTechnology

    LAYER_VIEWS = LayerViews(PATH.lyp_yaml)

    connectivity = cast(list[ConnectivitySpec], [("HEATER", "HEATER", "PAD")])

    t = KLayoutTechnology(
        name="Cornerstone_si340",
        layer_map=LAYER,
        layer_views=LAYER_VIEWS,
        layer_stack=LAYER_STACK,
        connectivity=connectivity,
    )
    t.write_tech(tech_dir=PATH.klayout)
