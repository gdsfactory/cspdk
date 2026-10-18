"""Technology definitions of the EVENT-ONLY CORNERSTONE e-beam flavour.

Every number below comes from the CORNERSTONE e-beam workshop specification
(Bilge, 17 Oct 2026) unless marked as a placeholder. The rules are provisional
and apply to the e-beam event only, not to regular CORNERSTONE MPW runs.
"""

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

from cspdk.si220_ebeam_event.config import PATH

nm = 1e-3


class LayerMapCornerstone(LayerMap):
    """Layer map of the CORNERSTONE e-beam event.

    Only WG (3/0), EXPOSURE (4/0) and GRA (6/0) are design layers. TRENCH_TMP
    (3/1) is the temporary layer of the foundry post-processing
    (:func:`cspdk.si220_ebeam_event.mask.post_process`) and must not be drawn.
    """

    WG: Layer = (3, 0)  # waveguide cores, light field
    TRENCH_TMP: Layer = (3, 1)  # post-processing only: WG sized by +5 um
    EXPOSURE: Layer = (4, 0)  # e-beam exposure, dark field (extra etch allowed)
    GRA: Layer = (6, 0)  # grating teeth
    FLOORPLAN: Layer = (99, 0)

    # labels for gdsfactory
    LABEL_SETTINGS: Layer = (100, 0)
    LABEL_INSTANCE: Layer = (101, 0)
    routing_error_marker: Layer = (1000, 0)


LAYER = LayerMapCornerstone


class Tech:
    """Process numbers and design rules of the e-beam event, in um.

    The single source of the rules used by the cells, ``mask.post_process`` and
    ``drc.check``.
    """

    design_rules = "CORNERSTONE e-beam event rules (17 Oct 2026, provisional)"

    # Process: SOI 220 nm with one 120 nm (+-10 nm) etch, 100 nm slab left.
    thickness_si = 220 * nm
    etch_depth = 120 * nm
    thickness_slab = 100 * nm
    trench_width = 5.0  # WG (3/0) is sized by +5 um to form the trenches

    # Design rules (apply to WG 3/0, EXPOSURE 4/0 and GRA 6/0).
    min_width = 0.2
    min_space = 0.2
    max_exposure_area = 2_000_000.0  # um^2 on 4/0 after post-processing
    radius_min = 25.0

    # Rib waveguide, as the CORNERSTONE student-batch rib (450 nm, R = 25 um).
    width_rib = 0.45
    radius_rib = 25.0
    # Euler bend default: its tightest curvature is ~0.706 x radius at p=0.5,
    # so radius 36 keeps it at 25.4 um, above radius_min.
    radius_euler = 36.0

    # Default coupling gap of couplers and rings (placeholder, >= min_space).
    gap = 0.25

    # Design area and provisional placeholder I/O.
    die_size = (6000.0, 3000.0)
    io_pitch = 250.0  # vertical pitch between inputs (and between outputs)
    io_min_separation = 500.0  # min horizontal distance, inputs to outputs


TECH = Tech()


def get_layer_stack(
    thickness_si: float = TECH.thickness_si,
    thickness_slab: float = TECH.thickness_slab,
    trench_width: float = TECH.trench_width,
) -> LayerStack:
    """Returns LayerStack.

    220 nm Si on BOX with a single 120 nm etch. The etched area is the
    post-processed exposure: the 5 um trenches around WG (3/0) plus EXPOSURE
    (4/0) and GRA (6/0). So:

    - core: WG not etched by 4/0 or 6/0, full 220 nm Si.
    - slab: the etched area, 100 nm Si left.
    - field: everything else inside FLOORPLAN keeps the full 220 nm Si. It
      only renders where a floorplan (e.g. ``die``) is drawn.

    Cladding and BOX thicknesses for the event are not known yet (TODO).

    Args:
        thickness_si: unetched Si thickness in um.
        thickness_slab: Si thickness left in etched regions in um.
        trench_width: trench width around WG in um.
    """
    wg = LogicalLayer(layer=LAYER.WG)
    etched_by_mask = LogicalLayer(layer=LAYER.EXPOSURE) + LogicalLayer(layer=LAYER.GRA)
    wg_grown = wg.sized(round(trench_width / nm))
    trench = wg_grown - wg
    return LayerStack(
        layers=dict(
            core=LayerLevel(
                layer=wg - etched_by_mask,
                thickness=thickness_si,
                zmin=0.0,
                material="Si",
                info={"mesh_order": 1},
                derived_layer=wg,
            ),
            slab=LayerLevel(
                layer=trench + etched_by_mask,
                thickness=thickness_slab,
                zmin=0.0,
                material="Si",
                info={"mesh_order": 2},
                derived_layer=LogicalLayer(layer=LAYER.EXPOSURE),
            ),
            field=LayerLevel(
                layer=LogicalLayer(layer=LAYER.FLOORPLAN) - wg_grown - etched_by_mask,
                thickness=thickness_si,
                zmin=0.0,
                material="Si",
                info={"mesh_order": 3},
                derived_layer=LogicalLayer(layer=LAYER.FLOORPLAN),
            ),
        )
    )


LAYER_STACK = get_layer_stack()
LAYER_VIEWS = gf.technology.LayerViews(PATH.lyp_yaml)


############################
# Cross-sections functions
############################
xsection = gf.xsection


@xsection
def rib(
    width: float = TECH.width_rib,
    layer: LayerSpec = "WG",
    radius: float = TECH.radius_rib,
    radius_min: float = TECH.radius_min,
    **kwargs,
) -> CrossSection:
    """Return the 450 nm rib cross-section (core on WG 3/0, 1550 nm TE).

    Only the core is drawn: the foundry post-processing etches a 5 um trench
    around it, so the slab is not drawn.

    Args:
        width: core width in um.
        layer: core layer.
        radius: default bend radius in um.
        radius_min: minimum bend radius in um (design rule).
        kwargs: passed to gdsfactory's cross_section.
    """
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
    cross_section: CrossSectionSpec = "rib",
    straight: ComponentSpec = "straight",
    bend: ComponentSpec = "bend_circular",
) -> ManhattanRoute:
    """Route two ports with a single route.

    Bends default to ``bend_circular`` at the 25 um minimum radius (an Euler
    bend of the same radius would be tighter than 25 um in its middle).
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
    separation: float = 5.0,
    sort_ports: bool = False,
    start_straight_length: float = 0.0,
    end_straight_length: float = 0.0,
    port_type: str | None = None,
    cross_section: CrossSectionSpec = "rib",
    straight: ComponentSpec = "straight",
    bend: ComponentSpec = "bend_circular",
    **kwargs,
) -> list[ManhattanRoute]:
    """Route two bundles of ports with 25 um circular bends.

    There is a single rib width, so no tapers are inserted.
    """
    return gf.routing.route_bundle(
        component=component,
        ports1=ports1,
        ports2=ports2,
        separation=separation,
        sort_ports=sort_ports,
        start_straight_length=start_straight_length,
        end_straight_length=end_straight_length,
        port_type=port_type,
        cross_section=cross_section,
        straight=straight,
        bend=bend,
        taper=None,
        auto_taper=False,
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
        name="Cornerstone_si220_ebeam_event",
        layer_map=LAYER,
        layer_views=LAYER_VIEWS,
        layer_stack=LAYER_STACK,
    )
    t.write_tech(tech_dir=PATH.klayout)
