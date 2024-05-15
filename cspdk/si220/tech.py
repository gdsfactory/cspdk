"""Technology definitions."""

import sys
from functools import partial
from typing import cast

import gdsfactory as gf
from gdsfactory.cross_section import get_cross_sections
from gdsfactory.technology import LayerLevel, LayerMap, LayerStack, LayerViews
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


LAYER = LayerMapCornerstone()


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
                layer=LAYER.WG,
                thickness=thickness_wg,
                zmin=0.0,
                material="si",
                info={"mesh_order": 1},
                sidewall_angle=10,
                width_to_z=0.5,
            ),
            slab=LayerLevel(
                layer=LAYER.SLAB,
                thickness=thickness_slab,
                zmin=0.0,
                material="si",
                info={"mesh_order": 1},
                sidewall_angle=10,
                width_to_z=0.5,
            ),
            heater=LayerLevel(
                layer=LAYER.HEATER,
                thickness=thickness_heater,
                zmin=zmin_heater,
                material="TiN",
                info={"mesh_order": 1},
            ),
            metal=LayerLevel(
                layer=LAYER.PAD,
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

xf_sc = partial(gf.cross_section.strip, layer=LAYER.WG, width=0.45)
xf_so = partial(xf_sc, width=0.40)

xf_rc = partial(
    gf.cross_section.strip,
    layer=LAYER.WG,
    width=0.45,
    sections=(gf.Section(width=10.45, layer="SLAB", name="slab", simplify=50 * nm),),
    radius=25,
    radius_min=25,
)
xf_ro = partial(xf_rc, width=0.40)
xf_rc_tip = partial(
    gf.cross_section.strip,
    sections=(gf.Section(width=0.2, layer="SLAB", name="slab"),),
)


xf_sc_heater_metal = partial(
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

############################
# Cross-sections
############################
xs_sc = xf_sc()
xs_rc = xf_rc()
xs_so = xf_so()
xs_ro = xf_ro()
xs_rc_tip = xf_rc_tip()

xs_sc_heater_metal = xf_sc_heater_metal()
xs_metal_routing = metal_routing()
xs_heater_metal = heater_metal()

cross_sections = get_cross_sections(sys.modules[__name__])

############################
# Routing functions
############################

_settings_sc = dict(
    straight="straight_sc", cross_section=xs_sc, bend="bend_sc", taper="taper_sc"
)
_settings_so = dict(
    straight="straight_so", cross_section=xs_so, bend="bend_so", taper="taper_so"
)
_settings_rc = dict(
    straight="straight_rc", cross_section=xs_rc, bend="bend_rc", taper="taper_rc"
)
_settings_ro = dict(
    straight="straight_ro", cross_section=xs_ro, bend="bend_ro", taper="taper_ro"
)

get_route_sc = partial(gf.routing.get_route, **_settings_sc)
get_route_so = partial(gf.routing.get_route, **_settings_so)
get_route_rc = partial(gf.routing.get_route, **_settings_rc)
get_route_ro = partial(gf.routing.get_route, **_settings_ro)

get_route_from_steps_sc = partial(
    gf.routing.get_route_from_steps,
    **_settings_sc,
)
get_route_from_steps_so = partial(
    gf.routing.get_route_from_steps,
    **_settings_so,
)
get_route_from_steps_rc = partial(
    gf.routing.get_route_from_steps,
    **_settings_rc,
)
get_route_from_steps_ro = partial(
    gf.routing.get_route_from_steps,
    **_settings_ro,
)

get_bundle_sc = partial(gf.routing.get_bundle, **_settings_sc)
get_bundle_so = partial(gf.routing.get_bundle, **_settings_so)
get_bundle_rc = partial(gf.routing.get_bundle, **_settings_rc)
get_bundle_ro = partial(gf.routing.get_bundle, **_settings_ro)

get_bundle_from_steps_sc = partial(
    gf.routing.get_bundle_from_steps,
    **_settings_sc,
)
get_bundle_from_steps_so = partial(
    gf.routing.get_bundle_from_steps,
    **_settings_so,
)
get_bundle_from_steps_rc = partial(
    gf.routing.get_bundle_from_steps,
    **_settings_rc,
)
get_bundle_from_steps_ro = partial(
    gf.routing.get_bundle_from_steps,
    **_settings_ro,
)


routing_strategies = dict(
    get_route_sc=get_route_sc,
    get_route_so=get_route_so,
    get_route_rc=get_route_rc,
    get_route_ro=get_route_ro,
    get_route_from_steps_sc=get_route_from_steps_sc,
    get_route_from_steps_so=get_route_from_steps_so,
    get_route_from_steps_rc=get_route_from_steps_rc,
    get_route_from_steps_ro=get_route_from_steps_ro,
    get_bundle_sc=get_bundle_sc,
    get_bundle_so=get_bundle_so,
    get_bundle_rc=get_bundle_rc,
    get_bundle_ro=get_bundle_ro,
    get_bundle_from_steps_sc=get_bundle_from_steps_sc,
    get_bundle_from_steps_so=get_bundle_from_steps_so,
    get_bundle_from_steps_rc=get_bundle_from_steps_rc,
    get_bundle_from_steps_ro=get_bundle_from_steps_ro,
)


if __name__ == "__main__":
    from gdsfactory.technology.klayout_tech import KLayoutTechnology

    LAYER_VIEWS = LayerViews(PATH.lyp_yaml)
    # LAYER_VIEWS.to_lyp(PATH.lyp)

    connectivity = cast(list[ConnectivitySpec], [("HEATER", "HEATER", "PAD")])

    t = KLayoutTechnology(
        name="Cornerstone_si220",
        layer_map=dict(LAYER),
        layer_views=LAYER_VIEWS,
        layer_stack=LAYER_STACK,
        connectivity=connectivity,
    )
    t.write_tech(tech_dir=PATH.klayout)
