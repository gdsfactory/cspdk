from functools import partial

from gdsfactory.routing import get_route
from gdsfactory.routing.factories import routing_strategy

from cspdk.cells import _bend, _straight, _taper, wire_corner

routing_strategies = {
    "get_bundle": partial(
        routing_strategy["get_bundle"],
        straight=_straight,
        bend=_bend,
        cross_section="xs_sc",
    ),
    "get_bundle_electrical": partial(
        routing_strategy["get_bundle_electrical"],
        straight=_straight,
        bend=wire_corner,
        cross_section="xs_sc",
    ),
    "get_bundle_path_length_match": partial(
        routing_strategy["get_bundle_path_length_match"],
        straight=_straight,
        bend=_bend,
        taper=_taper,
        cross_section="xs_sc",
    ),
    "get_bundle_same_axis_no_grouping": partial(
        routing_strategy["get_bundle_same_axis_no_grouping"],
        taper=_taper,
        route_filter=partial(
            get_route,
            bend=_bend,
            straight=_straight,
        ),
        cross_section="xs_sc",
    ),
    "get_bundle_from_waypoints": partial(
        routing_strategy["get_bundle_from_waypoints"],
        straight=_straight,
        taper=_taper,
        bend=_bend,
        cross_section="xs_sc",
    ),
    "get_bundle_from_steps": partial(
        routing_strategy["get_bundle_from_steps"],
        straight=_straight,
        taper=_taper,
        bend=_bend,
        cross_section="xs_sc",
    ),
    "get_bundle_from_steps_electrical": partial(
        routing_strategy["get_bundle_from_steps_electrical"],
        straight=_straight,
        taper=_taper,
        bend=wire_corner,
        cross_section="xs_sc",
    ),
}
