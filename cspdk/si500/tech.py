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

from cspdk.si500.config import PATH

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


class Tech:
    radius_rc = 25
    radius_ro = 25


TECH = Tech()


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


############################
# Cross-sections functions
############################
cladding_layers_rib = (LAYER.SLAB,)
cladding_offsets_rib = (5,)

strip = xs_rc = partial(
    gf.cross_section.strip,
    layer=LAYER.WG,
    width=0.45,
    sections=(gf.Section(width=10.45, layer="SLAB", name="slab", simplify=50 * nm),),
    radius=25,
    radius_min=25,
)
xs_ro = partial(xs_rc, width=0.40)
xs_rc_tip = partial(
    gf.cross_section.strip,
    sections=(gf.Section(width=0.2, layer="SLAB", name="slab"),),
)


metal_routing = partial(
    gf.cross_section.cross_section,
    layer=LAYER.PAD,
    width=10.0,
    port_names=gf.cross_section.port_names_electrical,
    port_types=gf.cross_section.port_types_electrical,
    radius=None,
)
xs_heater_metal = heater_metal = partial(metal_routing, width=4, layer=LAYER.HEATER)
cross_sections = get_cross_sections(sys.modules[__name__])

############################
# Routing functions
############################

_settings_rc = dict(
    straight="straight_rc", cross_section="xs_rc", bend="bend_rc", taper="taper_rc"
)

route_single_rc = partial(gf.routing.route_single, **_settings_rc)
route_bundle_rc = partial(gf.routing.route_bundle, **_settings_rc)

routing_strategies = dict(
    route_single_rc=route_single_rc,
    route_bundle_rc=route_bundle_rc,
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
