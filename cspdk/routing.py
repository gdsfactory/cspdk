from functools import partial

from gdsfactory.routing import get_route
from gdsfactory.routing.factories import routing_strategy

from cspdk import cells
from cspdk.cells import wire_corner


def get_routing_strategies():
    return {
        **get_routing_strategies_for_cross_section(""),
        **get_routing_strategies_for_cross_section("xs_sc"),
        **get_routing_strategies_for_cross_section("xs_so"),
        **get_routing_strategies_for_cross_section("xs_rc"),
        **get_routing_strategies_for_cross_section("xs_ro"),
        **get_routing_strategies_for_cross_section("xs_nc"),
        **get_routing_strategies_for_cross_section("xs_no"),
    }


def get_routing_strategies_for_cross_section(cross_section):
    suffix = ""
    if "_" in cross_section:
        suffix = cross_section.split("_")[-1]
    straight = getattr(cells, f"straight_{suffix}", cells.straight_sc)
    bend = getattr(cells, f"bend_{suffix}", cells.straight_sc)
    taper = getattr(cells, f"taper_{suffix}", cells.straight_sc)
    if suffix:
        suffix = f"_{suffix}"
    return {
        f"get_bundle{suffix}": partial(
            routing_strategy["get_bundle"],
            straight=straight,
            bend=bend,
            cross_section=cross_section,
        ),
        f"get_bundle_electrical{suffix}": partial(
            routing_strategy["get_bundle_electrical"],
            straight=straight,
            bend=wire_corner,
            cross_section=cross_section,
        ),
        f"get_bundle_path_length_match{suffix}": partial(
            routing_strategy["get_bundle_path_length_match"],
            straight=straight,
            bend=bend,
            taper=taper,
            cross_section=cross_section,
        ),
        f"get_bundle_same_axis_no_grouping{suffix}": partial(
            routing_strategy["get_bundle_same_axis_no_grouping"],
            taper=taper,
            route_filter=partial(
                get_route,
                bend=bend,
                straight=straight,
            ),
            cross_section=cross_section,
        ),
        f"get_bundle_from_waypoints{suffix}": partial(
            routing_strategy["get_bundle_from_waypoints"],
            straight=straight,
            taper=taper,
            bend=bend,
            cross_section=cross_section,
        ),
        f"get_bundle_from_steps{suffix}": partial(
            routing_strategy["get_bundle_from_steps"],
            straight=straight,
            taper=taper,
            bend=bend,
            cross_section=cross_section,
        ),
        f"get_bundle_from_steps_electrical{suffix}": partial(
            routing_strategy["get_bundle_from_steps_electrical"],
            straight=straight,
            taper=taper,
            bend=wire_corner,
            cross_section=cross_section,
        ),
    }


if __name__ == "__main__":
    print(list(get_routing_strategies()))
