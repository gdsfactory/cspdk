"""Technology definitions."""

import sys
from collections.abc import Iterable
from functools import partial

import gdsfactory as gf
from gdsfactory.cross_section import CrossSectionSpec, get_cross_sections
from gdsfactory.routing.route_bundle import ManhattanRoute
from gdsfactory.technology import (
    LayerLevel,
    LayerMap,
    LayerStack,
    LayerViews,
    LogicalLayer,
)
from gdsfactory.typings import ComponentSpec, ConnectivitySpec, Layer, LayerSpec

from cspdk.sin200.config import PATH

nm = 1e-3


class LayerMapCornerstone(LayerMap):
    """Layer map for Cornerstone SiN200 technology."""

    NITRIDE: Layer = (203, 0)  # type: ignore
    NITRIDE_ETCH: Layer = (204, 0)  # type: ignore
    FLOORPLAN: Layer = (99, 0)  # type: ignore
    HEATER: Layer = (39, 0)  # type: ignore
    LBL: Layer = (100, 0)  # type: ignore
    PAD: Layer = (41, 0)  # type: ignore
    CLAD_OPEN: Layer = (22, 0)  # type: ignore

    # labels for gdsfactory
    LABEL_SETTINGS: Layer = (100, 0)  # type: ignore
    LABEL_INSTANCE: Layer = (101, 0)  # type: ignore
    routing_error_marker: Layer = (1000, 0)  # type: ignore


LAYER = LayerMapCornerstone

CONNECTIVITY: list[ConnectivitySpec] = [("HEATER", "HEATER", "PAD")]


def get_layer_stack(
    thickness_nitride: float = 200 * nm,
    thickness_cladding: float = 2.0,
    thickness_heater: float = 150 * nm,
    thickness_metal: float = 220 * nm,
    zmin_heater: float | None = None,
    zmin_metal: float | None = None,
) -> LayerStack:
    """Returns LayerStack.

    Per the SiN200-Visible MPW design guidelines (section 4): 200 nm LPCVD SiN
    on a 2 um thermal BOX, fully etched to the BOX, under a 2 um SiO2 top
    cladding. Heater filaments (GDS 39) and heater contact pads (GDS 41) are
    both lifted off on top of the cladding, so they start at the same height.

    GDS 203 is light field (drawn SiN remains) and GDS 204 is dark field (drawn
    shapes are etched through the SiN, e.g. grating trenches), so the nitride
    level is NITRIDE minus NITRIDE_ETCH.

    The heater and pad metal materials and thicknesses are not published by
    the foundry; the values here are placeholders.

    Args:
        thickness_nitride: nitride core thickness in um.
        thickness_cladding: SiO2 top-cladding thickness over the nitride in um.
        thickness_heater: heater filament (GDS 39) thickness in um.
        thickness_metal: heater contact pad (GDS 41) thickness in um.
        zmin_heater: bottom of the heater filaments in um; defaults to the top
            of the cladding (thickness_nitride + thickness_cladding).
        zmin_metal: bottom of the contact pads in um; defaults to zmin_heater.
    """
    if zmin_heater is None:
        zmin_heater = thickness_nitride + thickness_cladding
    if zmin_metal is None:
        zmin_metal = zmin_heater
    return LayerStack(
        layers=dict(
            nitride=LayerLevel(
                layer=LogicalLayer(layer=LAYER.NITRIDE)
                - LogicalLayer(layer=LAYER.NITRIDE_ETCH),
                thickness=thickness_nitride,
                zmin=0.0,
                material="SiN",
                info={"mesh_order": 2},
                sidewall_angle=10,
                width_to_z=0.5,
                derived_layer=LogicalLayer(layer=LAYER.NITRIDE),
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
                zmin=zmin_metal,
                material="Al",
                info={"mesh_order": 2},
            ),
        )
    )


LAYER_STACK = get_layer_stack()
LAYER_VIEWS = gf.technology.LayerViews(PATH.lyp_yaml)


class Tech:
    """Technology parameters."""

    radius_n780 = 60
    radius_n638 = 40
    radius_n520 = 30
    width_n780 = 0.50
    width_n638 = 0.36
    width_n520 = 0.27


TECH = Tech()


############################
# Cross-sections functions
############################

# will be filled after all cross sections are defined:
DEFAULT_CROSS_SECTION_NAMES: dict[str, str] = {}


def xs_n780(
    width=Tech.width_n780, radius=Tech.radius_n780, **kwargs
) -> gf.CrossSection:
    """Returns nitride 780nm cross-section."""
    kwargs["layer"] = kwargs.get("layer", LAYER.NITRIDE)
    kwargs["radius_min"] = kwargs.get("radius_min", radius)
    xs = gf.cross_section.strip(width=width, radius=radius, **kwargs)
    if xs.name in DEFAULT_CROSS_SECTION_NAMES:
        xs._name = DEFAULT_CROSS_SECTION_NAMES[xs.name]
    return xs


def xs_n638(
    width=Tech.width_n638, radius=Tech.radius_n638, **kwargs
) -> gf.CrossSection:
    """Returns nitride 638nm cross-section."""
    kwargs["layer"] = kwargs.get("layer", LAYER.NITRIDE)
    kwargs["radius_min"] = kwargs.get("radius_min", radius)
    xs = gf.cross_section.strip(width=width, radius=radius, **kwargs)
    if xs.name in DEFAULT_CROSS_SECTION_NAMES:
        xs._name = DEFAULT_CROSS_SECTION_NAMES[xs.name]
    return xs


def xs_n520(
    width=Tech.width_n520, radius=Tech.radius_n520, **kwargs
) -> gf.CrossSection:
    """Returns nitride 520nm cross-section."""
    kwargs["layer"] = kwargs.get("layer", LAYER.NITRIDE)
    kwargs["radius_min"] = kwargs.get("radius_min", radius)
    xs = gf.cross_section.strip(width=width, radius=radius, **kwargs)
    if xs.name in DEFAULT_CROSS_SECTION_NAMES:
        xs._name = DEFAULT_CROSS_SECTION_NAMES[xs.name]
    return xs


def metal_routing(width=10.0, **kwargs) -> gf.CrossSection:
    """Returns metal routing cross-section."""
    kwargs["layer"] = kwargs.get("layer", LAYER.PAD)
    kwargs["port_names"] = kwargs.get(
        "port_names", gf.cross_section.port_names_electrical
    )
    kwargs["port_types"] = kwargs.get(
        "port_types", gf.cross_section.port_types_electrical
    )
    kwargs["radius"] = kwargs.get("radius", 0)
    kwargs["radius_min"] = kwargs.get("radius_min", kwargs["radius"])
    xs = gf.cross_section.cross_section(width=width, **kwargs)
    if xs.name in DEFAULT_CROSS_SECTION_NAMES:
        xs._name = DEFAULT_CROSS_SECTION_NAMES[xs.name]
    return xs


def heater_metal_sin200(width=2.0, **kwargs) -> gf.CrossSection:
    """Returns heater filament (GDS 39) cross-section.

    The 2 um default is the filament width of the foundry Heater.gds and the
    width recommended in the design guidelines (section 5.2).
    """
    kwargs["layer"] = kwargs.get("layer", LAYER.HEATER)
    xs = metal_routing(width=width, **kwargs).copy()
    if xs.name in DEFAULT_CROSS_SECTION_NAMES:
        xs._name = DEFAULT_CROSS_SECTION_NAMES[xs.name]
    return xs


def populate_default_cross_section_names():
    """Populates default cross-section names."""
    xss = {k: v() for k, v in get_cross_sections(sys.modules[__name__]).items()}
    for k, xs in xss.items():
        xs._name = ""
        _k = xs.name
        xs._name = k
        DEFAULT_CROSS_SECTION_NAMES[_k] = xs.name


populate_default_cross_section_names()
cross_sections = get_cross_sections(sys.modules[__name__])


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
    cross_section: CrossSectionSpec = "xs_n780",
    straight: ComponentSpec = "straight_n780",
    bend: ComponentSpec = "bend_euler_n780",
) -> ManhattanRoute:
    """Route single optical path."""
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
    cross_section: CrossSectionSpec = "xs_n780",
    straight: ComponentSpec = "straight_n780",
    bend: ComponentSpec = "bend_euler_n780",
    taper: ComponentSpec = "taper_n780",
) -> list[ManhattanRoute]:
    """Route bundle of optical paths."""
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


# Electrical routing on the contact-pad metal (GDS 41). Explicit straight
# lengths avoid the upstream electrical router's handling of None as an empty
# list in the installed gdsfactory/kfactory versions.
route_single_metal = partial(
    gf.routing.route_single_electrical,
    cross_section="metal_routing",
    start_straight_length=0.0,
    end_straight_length=0.0,
)
route_bundle_metal = partial(
    gf.routing.route_bundle_electrical,
    cross_section="metal_routing",
)


routing_strategies = dict(
    route_single_metal=route_single_metal,
    route_bundle_metal=route_bundle_metal,
    route_single=route_single,
    route_single_n780=partial(
        route_single,
        straight="straight_n780",
        bend="bend_euler_n780",
        cross_section="xs_n780",
    ),
    route_single_n638=partial(
        route_single,
        straight="straight_n638",
        bend="bend_euler_n638",
        cross_section="xs_n638",
    ),
    route_single_n520=partial(
        route_single,
        straight="straight_n520",
        bend="bend_euler_n520",
        cross_section="xs_n520",
    ),
    route_bundle=route_bundle,
    route_bundle_n780=partial(
        route_bundle,
        straight="straight_n780",
        bend="bend_euler_n780",
        taper="taper_n780",
        cross_section="xs_n780",
    ),
    route_bundle_n638=partial(
        route_bundle,
        straight="straight_n638",
        bend="bend_euler_n638",
        taper="taper_n638",
        cross_section="xs_n638",
    ),
    route_bundle_n520=partial(
        route_bundle,
        straight="straight_n520",
        bend="bend_euler_n520",
        taper="taper_n520",
        cross_section="xs_n520",
    ),
)


if __name__ == "__main__":
    from gdsfactory.technology.klayout_tech import KLayoutTechnology

    LAYER_VIEWS = LayerViews(PATH.lyp_yaml)

    t = KLayoutTechnology(
        name="Cornerstone_sin200",
        layer_map=LAYER,
        layer_views=LAYER_VIEWS,
        layer_stack=LAYER_STACK,
        connectivity=CONNECTIVITY,
    )
    t.write_tech(tech_dir=PATH.klayout)
