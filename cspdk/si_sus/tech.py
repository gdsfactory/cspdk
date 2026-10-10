"""Technology definitions."""

import sys

import gdsfactory as gf
from gdsfactory.cross_section import (
    ComponentAlongPath,
    CrossSection,
    CrossSectionSpec,
    Section,
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
from gdsfactory.typings import (
    ComponentSpec,
    Layer,
)

from cspdk.si_sus.config import PATH

nm = 1e-3


class LayerMapCornerstone(LayerMap):
    """Layer mapping for Cornerstone Suspended Si technology."""

    WG: Layer = (404, 0)
    SLAB: Layer = (405, 0)
    # abstract marker layer (not a foundry mask layer, datatype 10 is ignored
    # by mask prep): marks where the suspended waveguide core lies, since the
    # core itself is the UN-drawn Si between the etch windows on (404, 0)
    WG_MARK: Layer = (404, 10)
    FLOORPLAN: Layer = (99, 0)
    LBL: Layer = (100, 0)

    LABEL_SETTINGS: Layer = (100, 0)
    LABEL_INSTANCE: Layer = (101, 0)
    routing_error_marker: Layer = (1000, 0)


LAYER = LayerMapCornerstone


def get_layer_stack(
    thickness_wg: float = 450 * nm,
    thickness_slab: float = 150 * nm,
) -> LayerStack:
    """Returns LayerStack.

    Args:
        thickness_wg: Si waveguide thickness in um.
        thickness_slab: Si slab thickness in um.
    """
    # (404, 0) is a dark-field etch layer (drawn shapes are etched to BOX), so
    # the physical core is the UN-drawn Si between the etch windows. The core
    # level extrudes the abstract WG_MARK layer that xs_sus draws along the
    # waveguide center for exactly this purpose. The surrounding un-etched Si
    # (field minus 404) is not modeled: cells carry no field shape to
    # subtract from. The rib slab thickness (150nm, 404 AND 405 regions) has
    # no foundry source; the MPW #7 process figure suggests 175nm after HF.
    return LayerStack(
        layers=dict(
            core=LayerLevel(
                layer=LogicalLayer(layer=LAYER.WG_MARK),
                thickness=thickness_wg,
                zmin=0.0,
                material="Si",
                info={"mesh_order": 1},
                sidewall_angle=10,
                width_to_z=0.5,
            ),
            slab=LayerLevel(
                layer=LogicalLayer(layer=LAYER.SLAB),
                thickness=thickness_slab,
                zmin=0.0,
                material="Si",
                info={"mesh_order": 2},
                sidewall_angle=10,
                width_to_z=0.5,
            ),
        )
    )


LAYER_STACK = get_layer_stack()
LAYER_VIEWS = gf.technology.LayerViews(PATH.lyp_yaml)


class Tech:
    """Technology parameters.

    Geometry from the Cornerstone suspended-Si standard components library
    and its Suspendedsilicon500nm_3800nm_TE_Waveguide GDS: total
    cross-section width 8.5um = 3.5um etch window + 1.5um core + 3.5um etch
    window; each window is drawn as periodic 0.3um etch slots at 0.55um
    pitch, leaving 0.25um Si tethers that support the suspended core.

    Like the foundry library, all cells are drawn un-biased, for submission
    with the 'CORNERSTONE to bias' option (MPW #7 guidelines, Table 2: on
    layer 404 the drawn features must be >= 270nm and the gaps >= 180nm).
    """

    # radius_min_sus (20um) has no foundry source; the default is the 40.75um
    # center-line radius of the Suspendedsilicon500nm_3800nm_TE_90_DegreeBend
    # GDS (the library's 'suggested bend radius: 40um' is its inner core edge)
    radius_sus = 40.75
    radius_min_sus = 20
    width_sus = 1.5  # un-etched Si core between the two etch windows
    width_etch_window = 3.5  # width of each etch band drawn on (404, 0)
    tether_period = 0.55  # pitch of the etch slots along the waveguide
    etch_slot_length = 0.3  # etched slot length; 0.25um tether in between
    # first slot center >= 0.275um from a cell end, so the slot edge is at
    # least 0.125um inside the cell and abutting cells keep >= 0.25um tethers
    etch_slot_padding = 0.275
    # MPW #7 Table 2, 'CORNERSTONE to bias' option, layer 404
    min_feature_404 = 0.27
    min_gap_404 = 0.18
    max_suspended_width = 16.0


TECH = Tech()


############################
# Cross-sections functions
############################
xsection = gf.xsection


@gf.cell
def _etch_slot_pair(width: float = TECH.width_sus) -> gf.Component:
    """One pair of etch slots (one per etch window), centered at the origin.

    The windows start at the core edges (+-width/2), so the slot centers sit
    at +-(width + width_etch_window)/2. Built with raw kdb shapes (not
    gf.c.rectangle) so it can be created at import time, before any PDK is
    active.

    Args:
        width: width of the un-etched core between the two windows.
    """
    import kfactory as kf  # noqa: PLC0415

    c = gf.Component()
    dx = TECH.etch_slot_length / 2
    dy = TECH.width_etch_window / 2
    o = width / 2 + dy
    layer_index = c.kcl.layout.layer(*LAYER.WG)
    c.shapes(layer_index).insert(kf.kdb.DBox(-dx, o - dy, dx, o + dy))
    c.shapes(layer_index).insert(kf.kdb.DBox(-dx, -o - dy, dx, -o + dy))
    return c


@xsection
def xs_sus(
    width: float = TECH.width_sus,
    radius: float = TECH.radius_sus,
    radius_min: float = TECH.radius_min_sus,
) -> CrossSection:
    """Return the suspended-Si strip cross_section for 3800nm TE.

    (404, 0) is dark field: drawn shapes are etched to BOX, so the waveguide
    core is the UN-drawn Si between two 3.5um etch windows. Each window is
    drawn as periodic 0.3um etch slots (0.55um pitch), leaving 0.25um tethers
    that hold the suspended core, matching the
    Suspendedsilicon500nm_3800nm_TE_Waveguide GDS. The core Section is drawn
    on the abstract WG_MARK (404, 10) marker layer, which carries the
    optical ports and the LayerStack core level; it is not a foundry mask
    layer.

    The along-path slot pairs serve straights extruded by generic gdsfactory
    functions and are only correct on straight paths; the cspdk.si_sus cells
    draw their own slots at exact positions, following bends. The rib
    cross-section (solid etch windows + SLAB (405, 0) protect) is not
    implemented.

    Args:
        width: width of the un-etched core; the etch windows follow its edges.
        radius: bend radius.
        radius_min: minimum allowed bend radius.
    """
    return CrossSection(
        sections=(
            Section(
                width=width,
                layer=LAYER.WG_MARK,
                port_names=("o1", "o2"),
                port_types=("optical", "optical"),
            ),
        ),
        radius=radius,
        radius_min=radius_min,
        components_along_path=(
            ComponentAlongPath(
                component=_etch_slot_pair(width=width),
                spacing=TECH.tether_period,
                padding=TECH.etch_slot_padding,
            ),
        ),
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
    cross_section: CrossSectionSpec = "xs_sus",
    straight: ComponentSpec = "straight",
    bend: ComponentSpec = "bend_circular",
) -> ManhattanRoute:
    """Route two ports with a single route (circular bends, like the foundry)."""
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
    separation: float = 75.0,
    sort_ports: bool = False,
    start_straight_length: float = 0.0,
    end_straight_length: float = 0.0,
    min_straight_taper: float = 100.0,
    port_type: str | None = None,
    cross_section: CrossSectionSpec = "xs_sus",
    straight: ComponentSpec = "straight",
    bend: ComponentSpec = "bend_circular",
    taper: ComponentSpec = "taper",
    **kwargs,
) -> list[ManhattanRoute]:
    """Route two bundles of ports.

    The default separation is the MPW #7 guideline (section 5.3): at least
    75um between suspended waveguides to avoid collapse.
    """
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
        cross_section=cross_section,
        straight=straight,
        bend=bend,
        taper=taper,
        sbend="bend_s",
        **kwargs,
    )


cross_sections = get_cross_sections(sys.modules[__name__])


routing_strategies = dict(
    route_single=route_single,
    route_bundle=route_bundle,
)


if __name__ == "__main__":
    from gdsfactory.technology.klayout_tech import KLayoutTechnology

    LAYER_VIEWS = LayerViews(PATH.lyp_yaml)

    t = KLayoutTechnology(
        name="Cornerstone_si_sus",
        layer_map=LAYER,
        layer_views=LAYER_VIEWS,
        layer_stack=LAYER_STACK,
    )
    t.write_tech(tech_dir=PATH.klayout)
