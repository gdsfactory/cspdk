"""Shared helpers for the sample die-layout cells.

Used by ``sample_mzi_tree`` and ``sample_clements_mesh``: heater-fan-out
planning (``Column`` bookkeeping, escape-shelf constants), the default
edge coupler, the square metal corner used by the A* bundle bender, and
``route_heaters_to_pads`` — the comb-escape + doroutes A* pipeline that
wires heater columns to mirror-symmetric bond-pad rows.
"""

from __future__ import annotations

from functools import partial

import gdsfactory as gf
from doroutes import add_bundle_astar
from doroutes.routing import add_route_manual
from gdsfactory.typings import ComponentSpec

# Escape stub lengths from the l_e2 / r_e2 heater ports to their
# shelves; r_e2 shelves higher to pass over its sibling's via pad.
_SHELF_LO = 12.0
_SHELF_HI = 28.0

edge_coupler_strip = partial(
    gf.c.edge_coupler_silicon,
    length=300.0,
    width1=0.45,
    width2=0.2,
    cross_section="strip_cband",
)
"""Default edge coupler: gdsfactory inverse taper at cspdk strip width."""

Column = tuple[float, list[tuple[gf.Port, float]]]
"""(column_x_min, deepest-first list of (heater_port, shelf_y))."""


@gf.cell
def sharp_bend_metal(
    radius: float = 30.0, width: float = 10.0, layer: str = "PAD"
) -> gf.Component:
    """Sharp 90° metal corner with ``radius``-long arms.

    The arm length is the bend radius doroutes' A* planner derives from
    the port positions (allowed grid ≤ radius/2), so longer arms give a
    coarser, faster routing grid while the metal stays a square corner.

    Args:
        radius: arm length.
        width: trace width.
        layer: metal layer.
    """
    c = gf.Component()
    w = width / 2
    c.add_polygon(
        [(-radius, -w), (w, -w), (w, radius), (-w, radius), (-w, w), (-radius, w)],
        layer=layer,
    )
    c.add_port(
        "e1",
        center=(-radius, 0),
        width=width,
        orientation=180,
        layer=layer,
        port_type="electrical",
    )
    c.add_port(
        "e2",
        center=(0, radius),
        width=width,
        orientation=90,
        layer=layer,
        port_type="electrical",
    )
    c.info["length"] = 2 * radius
    c.info["radius"] = radius
    return c


def _pad_row_start(
    columns: list[Column],
    pad_pitch: float,
    wire_spacing: float,
    lane_margin: float,
) -> float:
    """Leftmost pad x such that every column's pad group ends left of that column's lane block (with margin)."""
    row_starts = []
    n_pads = 0
    for col_x_min, traces in columns:
        n_tr = len(traces)
        n_pads += n_tr
        lane_block_x = col_x_min - lane_margin - (n_tr + 2) * wire_spacing
        row_starts.append(lane_block_x - (n_pads - 1) * pad_pitch)
    return min(row_starts, default=0.0)


def _route_side(
    c: gf.Component,
    columns: list[Column],
    stage_ys: list[float],
    pad_ports: list[gf.Port],
    sign: int,
    wire_spacing: float,
    lane_margin: float,
    grid_unit: int,
    bend_radius: float,
) -> None:
    """Route one side's heater ports to its pad row.

    Stage 1: manual comb escapes bring each column's traces to an
    aligned virtual row (at exact bundle pitch) on its staging line.
    Stage 2: one doroutes A* bundle per column carries the staged
    traces to the column's pad group, rightmost column first.
    """
    dbu = 1000
    xs_metal = gf.get_cross_section("metal_routing")
    metal_layer = gf.get_layer_name(xs_metal.layer)
    bundle_bend = sharp_bend_metal(
        radius=bend_radius, width=xs_metal.width, layer=metal_layer
    )
    first_pad = 0
    bundles = []
    for (col_x_min, traces), stage_y in zip(columns, stage_ys):
        n_tr = len(traces)
        starts: list[tuple[int, int, float]] = []
        for t, (port, shelf_y) in enumerate(traces):
            lane_x = col_x_min - lane_margin - (n_tr - 1 - t) * wire_spacing
            stop = (
                int(round(lane_x * dbu)),
                int(round(stage_y * dbu)),
                90.0 if sign > 0 else 270.0,
            )
            add_route_manual(
                c,
                port,
                stop,
                corners=[(port.dx, shelf_y), (lane_x, shelf_y)],
                straight="straight_metal",
                bend="wire_corner",
            )
            starts.append(stop)
        bundles.append((starts, pad_ports[first_pad : first_pad + n_tr]))
        first_pad += n_tr

    for starts, targets in reversed(bundles):
        # Anchor both fans at center + half a pitch: zero-jog entry on
        # the staged side, maximal smallest jog on the pad side.
        lane_c = (min(s[0] for s in starts) + max(s[0] for s in starts)) / 2 / dbu
        pad_c = (min(p.dx for p in targets) + max(p.dx for p in targets)) / 2
        add_bundle_astar(
            c,
            starts,
            targets,
            straight="straight_metal",
            bend=bundle_bend,
            layers=[metal_layer],
            grid_unit=grid_unit,
            spacing=wire_spacing,
            fan_in={"type": "manhattan", "x_bundle": lane_c + wire_spacing / 2},
            fan_out={"type": "manhattan", "x_bundle": pad_c + wire_spacing / 2},
        )


def route_heaters_to_pads(
    c: gf.Component,
    north_cols: list[Column],
    south_cols: list[Column],
    *,
    pad: ComponentSpec = "pad",
    pad_pitch: float = 150.0,
    fan_room: float = 300.0,
    wire_spacing: float = 20.0,
    grid_unit: int = 15000,
    bend_radius: float = 30.0,
) -> None:
    """Place mirror-symmetric pad rows and wire the heater columns to them.

    Both rows share the same |y| and a common x span (the row-start
    constraint is one-sided, so the min feasible start is used).

    Args:
        c: component holding the placed heaters.
        north_cols: heater columns for the north side (see Column).
        south_cols: heater columns for the south side (see Column).
        pad: bond pad component.
        pad_pitch: bond pad pitch.
        fan_room: minimum gap between staging lines and the pad rows.
        wire_spacing: metal trace pitch.
        grid_unit: A* grid in dbu (≤ bend_radius/2 in dbu).
        bend_radius: arm length of the sharp metal corners.
    """
    lane_margin = 1.5 * wire_spacing
    stage_gap = 3.0 * wire_spacing
    bbox = c.dbbox()
    pad_cell = gf.get_component(pad)
    edge_abs = max(abs(bbox.top), abs(bbox.bottom))
    x0 = min(
        _pad_row_start(cols, pad_pitch, wire_spacing, lane_margin)
        for cols in (north_cols, south_cols)
    )

    for label, cols, sign, pname in (
        ("north", north_cols, +1, "e4"),  # e4 faces south
        ("south", south_cols, -1, "e2"),  # e2 faces north
    ):
        # Staging lines rise left→right so no bundle corridor crosses
        # another column's lane block.
        stage_ys: list[float] = []
        acc = 2.0 * wire_spacing
        for _, traces in cols:
            stage_ys.append(sign * (edge_abs + acc))
            acc += len(traces) * wire_spacing + stage_gap

        # The corridor between the top staging line and the pad row
        # must fit the widest bundle plus its pad fan-out comb.
        max_bundle = max(len(traces) for _, traces in cols) * wire_spacing
        room = max(fan_room, 2 * max_bundle + 4 * wire_spacing)
        pad_port_y = sign * (edge_abs + acc + room)

        n_traces = sum(len(traces) for _, traces in cols)
        pad_ports: list[gf.Port] = []
        for i in range(n_traces):
            pad_ref = c.add_ref(pad_cell, name=f"pad_{label}_{i}")
            pad_ref.dmove(
                (
                    x0 + i * pad_pitch - pad_cell.ports[pname].dx,
                    pad_port_y - pad_cell.ports[pname].dy,
                )
            )
            pad_ports.append(pad_ref.ports[pname])

        _route_side(
            c,
            cols,
            stage_ys,
            pad_ports,
            sign,
            wire_spacing,
            lane_margin,
            grid_unit,
            bend_radius,
        )
