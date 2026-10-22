"""Technology definitions."""

import sys
from functools import partial

import gdsfactory as gf
from gdsfactory.cross_section import CrossSection, get_cross_sections
from gdsfactory.technology import (
    LayerLevel,
    LayerMap,
    LayerStack,
    LayerViews,
    LogicalLayer,
)
from gdsfactory.typings import Layer, LayerSpec

from cspdk._checks import check_min
from cspdk.si220_ebeam_event.config import PATH

nm = 1e-3


class LayerMapCornerstone(LayerMap):
    """Layer map. Design layers are WG, EXPOSURE and GRA."""

    WG: Layer = (3, 0)  # waveguide cores, light field
    TRENCH_TMP: Layer = (3, 1)  # foundry temporary layer: do not draw
    EXPOSURE: Layer = (4, 0)  # e-beam exposure, dark field (extra etch allowed)
    GRA: Layer = (6, 0)  # grating teeth
    FLOORPLAN: Layer = (99, 0)

    # labels for gdsfactory
    LABEL_SETTINGS: Layer = (100, 0)
    LABEL_INSTANCE: Layer = (101, 0)
    routing_error_marker: Layer = (1000, 0)


LAYER = LayerMapCornerstone


class Tech:
    """Process numbers and design rules, in um."""

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

    # Rib waveguide, as the CORNERSTONE student-batch rib.
    width_rib = 0.45
    radius_rib = radius_min
    # Euler bend default: its tightest curvature is ~0.706 x radius at p=0.5,
    # so radius 36 keeps it at 25.4 um, above radius_min.
    radius_euler = 36.0

    # Default coupling gap of couplers and rings (placeholder, >= min_space).
    gap = 0.25

    # Design area and placeholder I/O.
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

    core: unetched WG. slab: the etched area (trenches, 4/0, 6/0). field: the
    unetched Si elsewhere inside FLOORPLAN (only rendered where one is drawn).

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
LAYER_VIEWS = LayerViews(PATH.lyp_yaml)


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
    """Return the rib cross-section: the core only; the trench is etched at mask prep.

    Args:
        width: core width in um.
        layer: core layer.
        radius: default bend radius in um.
        radius_min: minimum bend radius in um (design rule).
        kwargs: passed to gdsfactory's cross_section.
    """
    check_min("width", width, TECH.min_width, TECH.design_rules)
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

# Circular bends: an Euler bend of radius 25 um is tighter than 25 um mid-bend.
route_single = partial(
    gf.routing.route_single, cross_section="rib", bend="bend_circular"
)
route_bundle = partial(
    gf.routing.route_bundle,
    cross_section="rib",
    bend="bend_circular",
    taper=None,
    auto_taper=False,
)


# Required by the pdk-ci check-tech-structure hook.
cross_sections = get_cross_sections(sys.modules[__name__])


routing_strategies = dict(
    route_single=route_single,
    route_bundle=route_bundle,
)


if __name__ == "__main__":
    from gdsfactory.technology.klayout_tech import KLayoutTechnology

    t = KLayoutTechnology(
        name="Cornerstone_si220_ebeam_event",
        layer_map=LAYER,
        layer_views=LAYER_VIEWS,
        layer_stack=LAYER_STACK,
    )
    t.write_tech(tech_dir=PATH.klayout)
