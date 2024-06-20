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

from cspdk.sin300.config import PATH

nm = 1e-3


class LayerMapCornerstone(LayerMap):
    WG: Layer = (3, 0)
    SLAB: Layer = (5, 0)
    FLOORPLAN: Layer = (99, 0)
    HEATER: Layer = (39, 0)
    GRA: Layer = (6, 0)
    LBL: Layer = (100, 0)
    PAD: Layer = (41, 0)
    NITRIDE: Layer = (203, 0)
    NITRIDE_ETCH: Layer = (204, 0)

    # labels for gdsfactory
    LABEL_SETTINGS: Layer = (100, 0)
    LABEL_INSTANCE: Layer = (101, 0)


LAYER = LayerMapCornerstone


class Tech:
    radius_nc = 25
    radius_no = 25


TECH = Tech()


def get_layer_stack(
    thickness_nitride: float = 300 * nm,
    zmin_heater: float = 1.1,
    thickness_heater: float = 700 * nm,
    zmin_metal: float = 1.1,
    thickness_metal: float = 700 * nm,
) -> LayerStack:
    """Returns LayerStack.

    based on paper https://www.degruyter.com/document/doi/10.1515/nanoph-2013-0034/html

    Args:
        thickness_nitride: nitride thickness in um.
        zmin_heater: TiN heater.
        thickness_heater: TiN thickness.
        zmin_metal: metal thickness in um.
        thickness_metal: metal2 thickness.
    """

    return LayerStack(
        layers=dict(
            nitride=LayerLevel(
                layer=LogicalLayer(layer=LAYER.NITRIDE),
                thickness=thickness_nitride,
                zmin=0.0,
                material="sin",
                info={"mesh_order": 2},
                sidewall_angle=10,
                width_to_z=0.5,
            ),
            nitride_etch=LayerLevel(
                layer=LogicalLayer(layer=LAYER.NITRIDE_ETCH),
                thickness=thickness_nitride,
                zmin=0.0,
                material="sin",
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
strip = xs_nc = partial(
    gf.cross_section.strip, layer=LAYER.NITRIDE, width=1.20, radius=25
)
xs_no = partial(gf.cross_section.strip, layer=LAYER.NITRIDE, width=0.95, radius=25)

xs_nc_heater_metal = partial(
    gf.cross_section.strip_heater_metal,
    layer=LAYER.NITRIDE,
    heater_width=2.5,
    layer_heater=LAYER.HEATER,
    width=1.20,
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
_settings_nc = dict(
    straight="straight_nc", cross_section="xs_nc", bend="bend_nc", taper="taper_nc"
)
_settings_no = dict(
    straight="straight_no", cross_section="xs_no", bend="bend_no", taper="taper_no"
)

route_single_nc = partial(gf.routing.route_single, **_settings_nc)
route_single_no = partial(gf.routing.route_single, **_settings_no)


route_bundle_nc = partial(gf.routing.route_bundle, **_settings_nc)
route_bundle_no = partial(gf.routing.route_bundle, **_settings_no)


routing_strategies = dict(
    route_single_nc=route_single_nc,
    route_single_no=route_single_no,
    route_bundle_nc=route_bundle_nc,
    route_bundle_no=route_bundle_no,
)


if __name__ == "__main__":
    from gdsfactory.technology.klayout_tech import KLayoutTechnology

    LAYER_VIEWS = LayerViews(PATH.lyp_yaml)
    # LAYER_VIEWS.to_lyp(PATH.lyp)

    connectivity = cast(list[ConnectivitySpec], [("HEATER", "HEATER", "PAD")])

    t = KLayoutTechnology(
        name="Cornerstone_sin300",
        layer_map=LAYER,
        layer_views=LAYER_VIEWS,
        layer_stack=LAYER_STACK,
        connectivity=connectivity,
    )
    t.write_tech(tech_dir=PATH.klayout)

if __name__ == "__main__":
    print(type(LAYER.NITRIDE), LAYER.NITRIDE)
