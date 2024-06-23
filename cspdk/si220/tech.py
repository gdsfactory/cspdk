"""Technology definitions."""

import sys
from functools import partial
from typing import cast

import gdsfactory as gf
from gdsfactory.cross_section import get_cross_sections
from gdsfactory.technology import (
    LayerLevel,
    LayerMap,
    LayerStack,
    LayerViews,
    LogicalLayer,
)
from gdsfactory.typings import ConnectivitySpec, Layer

from cspdk.si220.config import PATH

nm = 1e-3


class LayerMapCornerstone(LayerMap):
    WG: Layer = (3, 0)
    SLAB: Layer = (5, 0)
    FLOORPLAN: Layer = (99, 0)
    HEATER: Layer = (39, 0)
    GRA: Layer = (6, 0)
    LBL: Layer = (100, 0)
    PAD: Layer = (41, 0)

    # labels for gdsfactory
    LABEL_SETTINGS: Layer = (100, 0)
    LABEL_INSTANCE: Layer = (101, 0)


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
    radius_sc = 5
    radius_so = 5
    radius_rc = 25
    radius_ro = 25


TECH = Tech()

############################
# Cross-sections functions
############################

strip = xs_sc = partial(gf.cross_section.strip, layer=LAYER.WG, width=0.45)
xs_so = partial(xs_sc, width=0.40)

xs_rc = partial(
    gf.cross_section.strip,
    layer=LAYER.WG,
    width=0.45,
    bbox_layers=(LAYER.SLAB,),
    bbox_offsets=(5,),
    radius=25,
    radius_min=25,
)
xs_ro = partial(xs_rc, width=0.40)


xs_sc_heater_metal = partial(
    gf.cross_section.strip_heater_metal,
    layer=LAYER.WG,
    heater_width=2.5,
    layer_heater=LAYER.HEATER,
    width=0.45,
)
metal_routing = partial(
    gf.cross_section.cross_section,
    layer=LAYER.PAD,
    width=10.0,
    port_names=gf.cross_section.port_names_electrical,
    port_types=gf.cross_section.port_types_electrical,
    radius=None,
)
heater_metal = partial(metal_routing, width=4, layer=LAYER.HEATER)
cross_sections = get_cross_sections(sys.modules[__name__])

############################
# Routing functions
############################

_settings_sc = dict(
    straight="straight_sc",
    cross_section="xs_sc",
    bend="bend_sc",
)
_settings_so = dict(
    straight="straight_so",
    cross_section="xs_so",
    bend="bend_so",
)
_settings_rc = dict(
    straight="straight_rc",
    cross_section="xs_rc",
    bend="bend_rc",
)
_settings_ro = dict(
    straight="straight_ro",
    cross_section="xs_ro",
    bend="bend_ro",
)

route_single_sc = partial(gf.routing.route_single, **_settings_sc)
route_single_so = partial(gf.routing.route_single, **_settings_so)
route_single_rc = partial(gf.routing.route_single, **_settings_rc)
route_single_ro = partial(gf.routing.route_single, **_settings_ro)

route_bundle_sc = partial(gf.routing.route_bundle, **_settings_sc)
route_bundle_so = partial(gf.routing.route_bundle, **_settings_so)
route_bundle_rc = partial(gf.routing.route_bundle, **_settings_rc)
route_bundle_ro = partial(gf.routing.route_bundle, **_settings_ro)


routing_strategies = dict(
    route_single_sc=route_single_sc,
    route_single_so=route_single_so,
    route_single_rc=route_single_rc,
    route_single_ro=route_single_ro,
    route_bundle_sc=route_bundle_sc,
    route_bundle_so=route_bundle_so,
    route_bundle_rc=route_bundle_rc,
    route_bundle_ro=route_bundle_ro,
)


if __name__ == "__main__":
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
