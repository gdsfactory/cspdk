"""N×N Clements mesh of 2×2 thermo-optic MZI switches.

Components:
    sample_mzi2x2_switch: balanced 2×2 MZI unit cell (mmi2x2 splitter/
        combiner, heater on the upper arm, S-bend fan-in/out to the
        channel pitch).
    sample_clements_mesh: rectangular Clements arrangement — N columns
        of unit cells on alternating channel pairs plus an output phase
        shifter per channel. Heaters are wired to mirror-symmetric
        bond-pad rows with the same comb-escape + doroutes A* machinery
        as ``sample_opa_mzi_tree``.
    sample_clements_mesh_die: the mesh in an edge-coupled die frame with
        fiber arrays, alignment loopbacks, a pass-through, markers and
        label.

Shared routing helpers (``route_heaters_to_pads``, ``Column``,
``edge_coupler_strip``, …) live in ``_die_layout_common``.
"""

from __future__ import annotations

import gdsfactory as gf
from cspdk.si220 import cells
from gdsfactory.typings import ComponentSpec, LayerSpec

from mycspdk.samples._die_layout_common import (
    _SHELF_HI,
    _SHELF_LO,
    Column,
    edge_coupler_strip,
    route_heaters_to_pads,
)


@gf.cell
def sample_mzi2x2_switch(
    length_heater: float = 80.0,
    arm_offset: float = 15.0,
    port_pitch: float = 100.0,
    sbend_length: float = 80.0,
    cross_section: str = "strip_cband",
) -> gf.Component:
    """Balanced 2×2 thermo-optic MZI switch at channel pitch.

    Both arms have identical shape (S-bend, 80 µm run, S-bend), so the
    interferometer is path-balanced; the upper arm carries the heater.

    Args:
        length_heater: heater length (also the lower-arm straight).
        arm_offset: vertical offset of each arm from the MMI ports.
        port_pitch: vertical pitch of the cell's optical ports.
        sbend_length: length of the fan-in/out S-bends.
        cross_section: waveguide cross section.

    Ports:
        o1, o2: top / bottom inputs (west, at ±port_pitch/2).
        o3, o4: top / bottom outputs (east).
        l_e2, r_e2: heater pads (north).
    """
    c = gf.Component()
    xs = cross_section
    arm_len = 40.0

    mi = c << gf.c.mmi2x2(cross_section=xs)
    wt, wb = mi.ports["o2"], mi.ports["o1"]  # west top / bottom
    et, eb = mi.ports["o3"], mi.ports["o4"]  # east top / bottom

    # Input fan: S-bends from ±port_pitch/2 down to the MMI ports
    si_t = c << gf.c.bend_s(
        size=(sbend_length, wt.dy - port_pitch / 2), cross_section=xs
    )
    si_t.connect("o2", wt)
    si_b = c << gf.c.bend_s(
        size=(sbend_length, wb.dy + port_pitch / 2), cross_section=xs
    )
    si_b.connect("o2", wb)

    # Balanced arms: heater on top, plain straight on the bottom
    at1 = c << gf.c.bend_s(size=(arm_len, arm_offset), cross_section=xs)
    at1.connect("o1", et)
    ht = c << cells.straight_heater_metal(length=length_heater)
    ht.connect("o1", at1.ports["o2"])
    at2 = c << gf.c.bend_s(size=(arm_len, -arm_offset), cross_section=xs)
    at2.connect("o1", ht.ports["o2"])
    ab1 = c << gf.c.bend_s(size=(arm_len, -arm_offset), cross_section=xs)
    ab1.connect("o1", eb)
    sb = c << gf.c.straight(length=length_heater, cross_section=xs)
    sb.connect("o1", ab1.ports["o2"])
    ab2 = c << gf.c.bend_s(size=(arm_len, arm_offset), cross_section=xs)
    ab2.connect("o1", sb.ports["o2"])

    mo = c << gf.c.mmi2x2(cross_section=xs)
    mo.connect("o2", at2.ports["o2"])  # bottom arm aligns by symmetry

    # Output fan back to ±port_pitch/2
    so_t = c << gf.c.bend_s(
        size=(sbend_length, port_pitch / 2 - mo.ports["o3"].dy), cross_section=xs
    )
    so_t.connect("o1", mo.ports["o3"])
    so_b = c << gf.c.bend_s(
        size=(sbend_length, -port_pitch / 2 - mo.ports["o4"].dy), cross_section=xs
    )
    so_b.connect("o1", mo.ports["o4"])

    c.add_port("o1", port=si_t.ports["o1"])
    c.add_port("o2", port=si_b.ports["o1"])
    c.add_port("o3", port=so_t.ports["o2"])
    c.add_port("o4", port=so_b.ports["o2"])
    c.add_port("l_e2", port=ht.ports["l_e2"])
    c.add_port("r_e2", port=ht.ports["r_e2"])
    return c


def _west_ports(ref: gf.ComponentReference) -> list[gf.Port]:
    """Optical west ports, top to bottom (robust to mirroring)."""
    ps = [p for p in ref.ports if p.port_type == "optical" and p.orientation == 180]
    return sorted(ps, key=lambda p: -p.dy)


def _east_ports(ref: gf.ComponentReference) -> list[gf.Port]:
    """Optical east ports, top to bottom (robust to mirroring)."""
    ps = [p for p in ref.ports if p.port_type == "optical" and p.orientation == 0]
    return sorted(ps, key=lambda p: -p.dy)


def _traces(ref: gf.ComponentReference, sign: int) -> list[tuple[gf.Port, float]]:
    """Comb traces for one heater: l_e2 low shelf, r_e2 high shelf."""
    l_e2, r_e2 = ref.ports["l_e2"], ref.ports["r_e2"]
    return [
        (l_e2, l_e2.dy + sign * _SHELF_LO),
        (r_e2, r_e2.dy + sign * _SHELF_HI),
    ]


@gf.cell
def sample_clements_mesh(
    n: int = 4,
    port_pitch: float = 100.0,
    x_spacing: float = 700.0,
    length_heater: float = 80.0,
    pad: ComponentSpec = "pad",
    pad_pitch: float = 150.0,
    fan_room: float = 300.0,
    wire_spacing: float = 20.0,
    grid_unit: int = 15000,
    bend_radius: float = 30.0,
) -> gf.Component:
    """N×N Clements mesh with an output phase shifter per channel.

    N columns of 2×2 MZI cells on alternating channel pairs (N even),
    N(N-1)/2 cells total. Cells below the midline are mirrored so their
    heaters face the south pad row; midline cells alternate north/south
    to keep the rows balanced.

    Args:
        n: number of channels (even).
        port_pitch: vertical channel pitch.
        x_spacing: column pitch (cell is ~370 µm wide; the rest is the
            metal lane corridor).
        length_heater: heater length in cells and output shifters.
        pad: bond pad component.
        pad_pitch: bond pad pitch.
        fan_room: minimum gap between staging lines and the pad rows.
        wire_spacing: metal trace pitch.
        grid_unit: A* grid in dbu (≤ bend_radius/2 in dbu).
        bend_radius: arm length of the sharp metal corners.

    Ports:
        o_in_0..o_in_{n-1} / o_out_0..o_out_{n-1}: channel ports, top
        (channel 0) to bottom.
    """
    if n % 2 or n < 2:
        raise ValueError(f"n must be even and >= 2, got {n}")
    c = gf.Component()
    cell = sample_mzi2x2_switch(length_heater=length_heater, port_pitch=port_pitch)
    shifter = cells.straight_heater_metal(length=length_heater)

    ch_y = [(n - 1) / 2 * port_pitch - i * port_pitch for i in range(n)]
    cur: dict[int, gf.Port] = {}
    inputs: dict[int, gf.Port] = {}
    north_cols: list[Column] = []
    south_cols: list[Column] = []
    n_center = 0  # alternator for midline cells

    def consume(ch: int, port: gf.Port) -> None:
        """Wire channel ch into ``port`` (or register it as an input)."""
        if ch in cur:
            gf.routing.route_single(c, cur[ch], port, cross_section="strip_cband")
        else:
            inputs[ch] = port

    for col in range(n):
        x = col * x_spacing
        col_cells: list[tuple[gf.ComponentReference, float]] = []
        for i in range(col % 2, n - 1, 2):
            yc = (ch_y[i] + ch_y[i + 1]) / 2
            if yc == 0:
                mirror = n_center % 2 == 1
                n_center += 1
            else:
                mirror = yc < 0
            ref = c.add_ref(cell, name=f"cell_C{col}_T{i}")
            if mirror:
                ref.dmirror_y(0)
            ref.dmove((x, yc))
            top_in, bottom_in = _west_ports(ref)
            consume(i, top_in)
            consume(i + 1, bottom_in)
            top_out, bottom_out = _east_ports(ref)
            cur[i], cur[i + 1] = top_out, bottom_out
            col_cells.append((ref, yc))

        # Side split: cells route to the row their heaters face.
        ncells = sorted(
            (r for r, _ in col_cells if r.ports["l_e2"].orientation == 90),
            key=lambda r: abs(r.ports["l_e2"].dy),
        )
        scells = sorted(
            (r for r, _ in col_cells if r.ports["l_e2"].orientation == 270),
            key=lambda r: abs(r.ports["l_e2"].dy),
        )
        col_x_min = x + cell.dbbox().left
        if ncells:
            north_cols.append((col_x_min, [t for r in ncells for t in _traces(r, +1)]))
        if scells:
            south_cols.append((col_x_min, [t for r in scells for t in _traces(r, -1)]))

    # Output phase shifters, one per channel (lower half mirrored)
    x = n * x_spacing
    nsh, ssh = [], []
    for i in range(n):
        ref = c.add_ref(shifter, name=f"phase_{i}")
        if ch_y[i] < 0:
            ref.dmirror_y(0)
        ref.dmove((x, ch_y[i]))
        gf.routing.route_single(
            c, cur[i], _west_ports(ref)[0], cross_section="strip_cband"
        )
        cur[i] = _east_ports(ref)[0]
        (nsh if ch_y[i] > 0 else ssh).append(ref)
    col_x_min = x + shifter.dbbox().left
    key = lambda r: abs(r.ports["l_e2"].dy)  # noqa: E731
    north_cols.append(
        (col_x_min, [t for r in sorted(nsh, key=key) for t in _traces(r, +1)])
    )
    south_cols.append(
        (col_x_min, [t for r in sorted(ssh, key=key) for t in _traces(r, -1)])
    )

    route_heaters_to_pads(
        c,
        north_cols,
        south_cols,
        pad=pad,
        pad_pitch=pad_pitch,
        fan_room=fan_room,
        wire_spacing=wire_spacing,
        grid_unit=grid_unit,
        bend_radius=bend_radius,
    )

    for i in range(n):
        c.add_port(f"o_in_{i}", port=inputs[i])
        c.add_port(f"o_out_{i}", port=cur[i])
    return c


@gf.cell
def sample_clements_mesh_die(
    n: int = 4,
    edge_coupler: ComponentSpec = edge_coupler_strip,
    coupler_pitch: float = 127.0,
    fanout_length: float = 400.0,
    die_margin: float = 150.0,
    layer_floorplan: LayerSpec = "FLOORPLAN",
) -> gf.Component:
    """Clements mesh in an edge-coupled die frame.

    West and east fiber arrays (n signals each, alignment loopback pair
    at both ends), a straight pass-through south of the pad rows,
    corner markers, and a label. The outline is centered on the
    mirror-symmetric pad rows.

    Args:
        n: mesh channel count (even).
        edge_coupler: edge coupler component.
        coupler_pitch: facet coupler pitch.
        fanout_length: horizontal room for the facet fans.
        die_margin: clearance between structures and the die outline.
        layer_floorplan: die outline layer.

    Ports:
        o_in_i / o_out_i: facet ports, top (channel 0) to bottom.
        o_pass_w, o_pass_e: pass-through facet ports.
    """
    c = gf.Component()
    mesh = c << sample_clements_mesh(n=n)
    bbox = mesh.dbbox()
    ec_cell = gf.get_component(edge_coupler)
    facet_port = ec_cell.ports["o2"]  # facet-side (taper tip) port

    pad_boxes = [
        i.dbbox() for i in mesh.cell.insts if (i.name or "").startswith("pad_")
    ]
    row_center = (min(b.left for b in pad_boxes) + max(b.right for b in pad_boxes)) / 2

    def place_array(facet_x: float, west: bool) -> list[gf.Port]:
        """Coupler array with loopbacks; returns signal ports, top first."""
        ref = c.add_ref(
            gf.c.edge_coupler_array_with_loopback(
                edge_coupler=edge_coupler,
                n=n + 4,
                pitch=coupler_pitch,
                text=None,
                cross_section="strip_cband",
            ),
            name="ec_west" if west else "ec_east",
        )
        if west:
            ref.drotate(180)
            ref.dmove((facet_x - ref.dbbox().left, 0))
        else:
            ref.dmove((facet_x - ref.dbbox().right, 0))
        sig = [ref.ports[f"o{i + 1}"] for i in range(n)]
        ref.dmove((0, -(min(p.dy for p in sig) + max(p.dy for p in sig)) / 2))
        sig = [ref.ports[f"o{i + 1}"] for i in range(n)]  # re-fetch after move
        return sorted(sig, key=lambda p: -p.dy)

    # East facet first (its position is set by the mesh), then center
    # the outline on the pad rows and put the west facet at the mirror.
    # The coupler length plus 10 µm of slack clears the array from the fan.
    facet_e = bbox.right + fanout_length + ec_cell.dxsize + 10.0
    east_sig = place_array(facet_e, west=False)
    facet_w = min(2 * row_center - facet_e, bbox.left - die_margin)
    west_sig = place_array(facet_w, west=True)

    gf.routing.route_bundle(
        c,
        west_sig,
        [mesh.ports[f"o_in_{i}"] for i in range(n)],
        cross_section="strip_cband",
        separation=5.0,
        sort_ports=False,
    )
    gf.routing.route_bundle(
        c,
        [mesh.ports[f"o_out_{i}"] for i in range(n)],
        east_sig,
        cross_section="strip_cband",
        separation=5.0,
        sort_ports=False,
    )
    for i in range(n):
        c.add_port(
            f"o_in_{i}",
            center=(facet_w, west_sig[i].dy),
            width=facet_port.width,
            orientation=180,
            layer=facet_port.layer,
            port_type="optical",
        )
        c.add_port(
            f"o_out_{i}",
            center=(facet_e, east_sig[i].dy),
            width=facet_port.width,
            orientation=0,
            layer=facet_port.layer,
            port_type="optical",
        )

    # Floorplan, pass-through, markers, label
    half_h = (
        max(
            abs(bbox.top),
            abs(bbox.bottom),
            (n + 3) / 2 * coupler_pitch + coupler_pitch,
        )
        + 2 * die_margin
    )
    # Pass-through just below the coupler array but well inside the pad
    # rows — never in the die-edge margin under the pads.
    pass_y = -((n + 3) / 2 * coupler_pitch + coupler_pitch)
    pw = c.add_ref(ec_cell, name="ec_pass_w")
    pw.drotate(180)
    pw.dmove((facet_w - pw.ports["o2"].dx, pass_y - pw.ports["o2"].dy))
    pe = c.add_ref(ec_cell, name="ec_pass_e")
    pe.dmove((facet_e - pe.ports["o2"].dx, pass_y - pe.ports["o2"].dy))
    gf.routing.route_single(
        c, pw.ports["o1"], pe.ports["o1"], cross_section="strip_cband"
    )
    c.add_port("o_pass_w", port=pw.ports["o2"])
    c.add_port("o_pass_e", port=pe.ports["o2"])

    fp = c << gf.c.rectangle(
        size=(facet_e - facet_w, 2 * half_h),
        layer=layer_floorplan,
        centered=False,
        port_type=None,
    )
    fp.dmove((facet_w, -half_h))

    marker = gf.c.cross(length=100.0, width=10.0, layer="PAD")
    inset = 1.5 * die_margin
    for mx in (facet_w + inset, facet_e - inset):
        for my in (-half_h + inset, half_h - inset):
            m = c.add_ref(marker, name=f"marker_{int(mx)}_{int(my)}")
            m.dmove((mx, my))
    label = c << gf.c.text(text="CLEMENTS_MESH", size=100.0, layer="PAD")
    label.dmove((facet_w + 3 * die_margin, half_h - 2.5 * die_margin))

    return c
