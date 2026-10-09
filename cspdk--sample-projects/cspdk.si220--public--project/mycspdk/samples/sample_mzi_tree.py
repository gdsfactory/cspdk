"""1×2^N thermo-optic MZI switch tree with electrical fan-out and an edge-coupled die wrapper.

Components:
    sample_mzi_switch: 1×2 thermo-optic switch (mmi1x2 splitter, mmi2x2
        combiner).
    sample_opa_mzi_tree: binary switch tree. Optics are routed with
        playdough. Heaters are wired to bond-pad rows (north for the
        upper half, south for the mirrored lower half) with doroutes:
        deterministic comb escapes to per-column staging lines, then
        one A* bundle per column to that column's group of pads.
    sample_opa_mzi_tree_l5: thin wrapper fixing the tree at levels=5
        (1×32, 31 MZIs).
    sample_opa_mzi_tree_die: the tree in a die frame with edge-coupler
        fiber arrays (alignment loopbacks included), a reference MZI, a
        pass-through waveguide, corner markers, and a label.

Shared routing helpers (``route_heaters_to_pads``, ``Column``,
``edge_coupler_strip``, …) live in ``_die_layout_common``.
"""

from __future__ import annotations

import gdsfactory as gf
import jax
import playdough as pld
from cspdk.si220 import cells
from gdsfactory.typings import ComponentSpec, LayerSpec

from mycspdk.samples._die_layout_common import (
    _SHELF_HI,
    _SHELF_LO,
    Column,
    edge_coupler_strip,
    route_heaters_to_pads,
)

_CONFIGURED = False


def _ensure_configured() -> None:
    """Configure playdough lazily.

    playdough's precompiled routing kernels are float32; cspdk (via sax)
    switches jax to 64-bit on import — switch it back, but only when a
    playdough route is actually built, so importing this module does not
    change global jax precision for other users.
    """
    global _CONFIGURED  # noqa: PLW0603
    if _CONFIGURED:
        return
    jax.config.update("jax_enable_x64", False)
    pld.config.BACKEND = "hlo"
    _CONFIGURED = True


@gf.cell
def sample_mzi_switch(
    delta_length: float = 10.0,
    length_heater: float = 80.0,
    cross_section: str = "strip_cband",
) -> gf.Component:
    """1×2 thermo-optic MZI switch.

    Args:
        delta_length: arm length difference.
        length_heater: heater length on the upper arm.
        cross_section: waveguide cross section.

    Ports:
        o1: optical input (west).
        o2, o3: upper / lower optical outputs (east).
        l_e2, r_e2: heater pads (north).
    """
    c = gf.Component()

    sp = c << gf.c.mmi1x2(cross_section=cross_section)
    sp.name = "splitter"

    # Upper arm with heater
    b1 = c << gf.c.bend_euler(cross_section=cross_section)
    b1.connect("o1", sp.ports["o2"])
    sl = c << gf.c.straight(length=delta_length / 2, cross_section=cross_section)
    sl.connect("o1", b1.ports["o2"])
    b2 = c << gf.c.bend_euler(cross_section=cross_section)
    b2.connect("o2", sl.ports["o2"])
    h = c << cells.straight_heater_metal(length=length_heater)
    h.name = "heater"
    h.connect("o1", b2.ports["o1"])
    b3 = c << gf.c.bend_euler(cross_section=cross_section)
    b3.connect("o2", h.ports["o2"])
    sr = c << gf.c.straight(length=delta_length / 2, cross_section=cross_section)
    sr.connect("o1", b3.ports["o1"])
    b4 = c << gf.c.bend_euler(cross_section=cross_section)
    b4.connect("o1", sr.ports["o2"])

    cp = c << gf.c.mmi2x2(cross_section=cross_section)
    cp.name = "combiner"
    cp.connect("o4", b4.ports["o2"])

    # Lower arm
    gf.routing.route_single(
        c, sp.ports["o3"], cp.ports["o3"], cross_section=cross_section
    )

    c.add_port("o1", port=sp.ports["o1"])
    c.add_port("o2", port=cp.ports["o1"])
    c.add_port("o3", port=cp.ports["o2"])
    c.add_port("l_e2", port=h.ports["l_e2"])
    c.add_port("r_e2", port=h.ports["r_e2"])
    return c


def _outputs_top_bottom(ref: gf.ComponentReference) -> tuple[gf.Port, gf.Port]:
    """Return (top, bottom) output ports of an MZI, robust to mirroring."""
    o2, o3 = ref.ports["o2"], ref.ports["o3"]
    return (o2, o3) if o2.dy >= o3.dy else (o3, o2)


def _place_tree(
    c: gf.Component,
    mzi: gf.Component,
    levels: int,
    x_spacing: float,
    y_spacing: float,
) -> list[list[gf.ComponentReference]]:
    """Place the binary tree; lower-half MZIs (y < 0) are mirrored so their heaters face the south pad row."""
    mzi_refs = []
    for level in range(levels):
        pitch = y_spacing * 2 ** (levels - 1 - level)
        y0 = -(2**level - 1) * pitch / 2
        refs = []
        for i in range(2**level):
            ref = c.add_ref(mzi, name=f"mzi_L{level}_N{i}")
            y = y0 + i * pitch
            if y < 0:
                ref.dmirror_y(0)
            ref.dmove((level * x_spacing, y))
            refs.append(ref)
        mzi_refs.append(refs)
    return mzi_refs


def _route_tree_optical(
    c: gf.Component,
    mzi_refs: list[list[gf.ComponentReference]],
    radius: float = 40.0,
) -> None:
    """Connect each parent's outputs to its children, top to top, so the waveguides never cross regardless of mirroring.

    Corners are drawn at ``radius`` (well above the cross-section
    minimum — there is plenty of clearance between tree columns, and
    small-radius corners read as kinks at layout scale).
    """
    _ensure_configured()
    xs = gf.get_cross_section("strip_cband").copy(radius=radius, radius_min=radius / 2)
    for level, parents in enumerate(mzi_refs[:-1]):
        children = mzi_refs[level + 1]
        for i, parent in enumerate(parents):
            top_out, bottom_out = _outputs_top_bottom(parent)
            pld.add_route_bundle(
                c,
                start_ports=[top_out, bottom_out],
                end_ports=[
                    children[2 * i + 1].ports["o1"],
                    children[2 * i].ports["o1"],
                ],
                cross_section=xs,
            )


def _heater_columns(
    mzi_refs: list[list[gf.ComponentReference]],
    mzi: gf.Component,
    x_spacing: float,
    lane_margin: float,
) -> tuple[list[Column], list[Column]]:
    """Group heater traces per side and per tree column.

    Within a column the deepest MZI (closest to the tree midline) comes
    first, l_e2 before r_e2. The lone root MZI is split across both
    sides for symmetry: l_e2 joins the north comb; r_e2 escapes east
    (west would cross l_e2's stub) and dives down the empty corridor
    between columns 0 and 1 as a south pseudo-column.
    """
    north_cols: list[Column] = []
    south_cols: list[Column] = []
    for level, level_refs in enumerate(mzi_refs):
        col_x_min = level * x_spacing + mzi.dbbox().left
        if level == 0:
            root = level_refs[0]
            l_e2, r_e2 = root.ports["l_e2"], root.ports["r_e2"]
            north_cols.append((col_x_min, [(l_e2, l_e2.dy + _SHELF_LO)]))
            pseudo_x = level * x_spacing + mzi.dbbox().right + 2 * lane_margin
            south_cols.append((pseudo_x, [(r_e2, r_e2.dy + _SHELF_HI)]))
            continue
        ntraces: list[tuple[gf.Port, float]] = []
        straces: list[tuple[gf.Port, float]] = []
        nrefs = sorted(
            (r for r in level_refs if r.ports["l_e2"].orientation == 90),
            key=lambda r: r.dy,
        )
        srefs = sorted(
            (r for r in level_refs if r.ports["l_e2"].orientation == 270),
            key=lambda r: -r.dy,
        )
        for ref in nrefs:
            ntraces.append((ref.ports["l_e2"], ref.ports["l_e2"].dy + _SHELF_LO))
            ntraces.append((ref.ports["r_e2"], ref.ports["r_e2"].dy + _SHELF_HI))
        for ref in srefs:
            straces.append((ref.ports["l_e2"], ref.ports["l_e2"].dy - _SHELF_LO))
            straces.append((ref.ports["r_e2"], ref.ports["r_e2"].dy - _SHELF_HI))
        north_cols.append((col_x_min, ntraces))
        south_cols.append((col_x_min, straces))
    return north_cols, south_cols


@gf.cell
def sample_opa_mzi_tree(
    levels: int = 4,
    delta_length: float = 10.0,
    x_spacing: float = 1000.0,
    y_spacing: float = 100.0,
    pad: ComponentSpec = "pad",
    pad_pitch: float = 150.0,
    fan_room: float = 300.0,
    wire_spacing: float = 20.0,
    grid_unit: int = 15000,
    bend_radius: float = 30.0,
) -> gf.Component:
    """1×2^levels binary MZI switch tree with bond pads.

    Args:
        levels: tree levels (4 → 1×16 tree, 15 MZIs).
        delta_length: MZI arm length difference.
        x_spacing: column pitch; must leave room for the widest lane
            block (2^(levels-1) · wire_spacing) beside the ~170 µm MZI.
        y_spacing: leaf-level MZI pitch.
        pad: bond pad component.
        pad_pitch: bond pad pitch.
        fan_room: minimum gap between staging lines and the pad rows.
        wire_spacing: metal trace pitch.
        grid_unit: A* grid in dbu (≤ bend_radius/2 in dbu).
        bend_radius: arm length of the sharp metal corners; keep
            2·bend_radius ≤ (pad_pitch − wire_spacing)/2 + wire_spacing.

    Ports:
        o1: tree input. o_upper_i / o_lower_i: leaf outputs.
    """
    c = gf.Component()
    mzi = sample_mzi_switch(delta_length=delta_length)
    lane_margin = 1.5 * wire_spacing

    mzi_refs = _place_tree(c, mzi, levels, x_spacing, y_spacing)
    _route_tree_optical(c, mzi_refs)
    north_cols, south_cols = _heater_columns(mzi_refs, mzi, x_spacing, lane_margin)

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

    c.add_port("o1", port=mzi_refs[0][0].ports["o1"])
    for i, leaf in enumerate(mzi_refs[-1]):
        top_out, bottom_out = _outputs_top_bottom(leaf)
        c.add_port(f"o_upper_{i}", port=top_out)
        c.add_port(f"o_lower_{i}", port=bottom_out)
    return c


@gf.cell
def sample_opa_mzi_tree_l5() -> gf.Component:
    """1×32 binary MZI switch tree — ``sample_opa_mzi_tree`` with ``levels=5``.

    Ports:
        o1: tree input. o_upper_i / o_lower_i: leaf outputs.
    """
    return sample_opa_mzi_tree(levels=5)


def _split_pair(
    c: gf.Component,
    top_port: gf.Port,
    bottom_port: gf.Port,
    sep: float,
    length: float = 75.0,
) -> list[gf.Port]:
    """S-bend a closely spaced output pair apart to ``sep``; returns the separated [top, bottom] ports."""
    center = (top_port.dy + bottom_port.dy) / 2
    out = []
    for port, sgn in ((top_port, +1), (bottom_port, -1)):
        b = c << gf.c.bend_s(
            size=(length, center + sgn * sep / 2 - port.dy),
            cross_section="strip_cband",
        )
        b.connect("o1", port)
        out.append(b.ports["o2"])
    return out


@gf.cell
def sample_opa_mzi_tree_die(
    levels: int = 4,
    x_spacing: float = 1800.0,
    edge_coupler: ComponentSpec = edge_coupler_strip,
    coupler_pitch: float = 127.0,
    fanout_length: float = 800.0,
    fanout_sep: float = 25.0,
    die_margin: float = 150.0,
    layer_floorplan: LayerSpec = "FLOORPLAN",
) -> gf.Component:
    """OPA MZI tree in an edge-coupled die frame.

    Design-for-test: all facet couplers sit on a uniform
    ``coupler_pitch`` grid; the east output array and the west input
    carry fiber-alignment loopbacks at their ends; a reference MZI
    (heaters unconnected) and a straight pass-through allow baseline
    measurements; metal crosses mark the corners. The die outline is
    centered on the mirror-symmetric pad rows and the spare west strip
    hosts the test structures.

    Args:
        levels: tree levels.
        x_spacing: tree column pitch (wider than the standalone default
            so the tree sits centered in the die).
        edge_coupler: edge coupler component.
        coupler_pitch: facet coupler pitch.
        fanout_length: horizontal room for the east output fan.
        fanout_sep: separation of each leaf's output pair after S-bends.
        die_margin: clearance between structures and the die outline.
        layer_floorplan: die outline layer.

    Ports:
        o_in, o_out_0..o_out_{2^levels-1}: facet ports (west / east).
        o_test_in, o_test_up, o_test_low: reference MZI facet ports.
        o_pass_w, o_pass_e: pass-through facet ports.
    """
    c = gf.Component()
    tree = c << sample_opa_mzi_tree(levels=levels, x_spacing=x_spacing)
    bbox = tree.dbbox()
    n_out = 2**levels
    ec_cell = gf.get_component(edge_coupler)
    facet_port = ec_cell.ports["o2"]  # facet-side (taper tip) port
    # Clearance between the tree's east edge and the start of the output fan.
    fan_clearance = 75.0

    # East facet: output array with a loopback pair at each end.
    fan_ports: list[gf.Port] = []
    for i in range(n_out // 2):
        fan_ports += _split_pair(
            c, tree.ports[f"o_upper_{i}"], tree.ports[f"o_lower_{i}"], fanout_sep
        )
    east = c.add_ref(
        gf.c.edge_coupler_array_with_loopback(
            edge_coupler=edge_coupler,
            n=n_out + 4,
            pitch=coupler_pitch,
            text=None,
            cross_section="strip_cband",
        ),
        name="ec_east",
    )
    east.dmove((bbox.right + fan_clearance + fanout_length - east.ports["o1"].dx, 0))
    sig = [east.ports[f"o{i + 1}"] for i in range(n_out)]
    east.dmove((0, -(min(p.dy for p in sig) + max(p.dy for p in sig)) / 2))
    # ports are positional snapshots: re-fetch after the move
    sig = [east.ports[f"o{i + 1}"] for i in range(n_out)]
    facet_e = east.dbbox().right

    out_ports = sorted(sig, key=lambda p: p.dy)
    fan_ports.sort(key=lambda p: p.dy)
    gf.routing.route_bundle(
        c,
        fan_ports,
        out_ports,
        cross_section="strip_cband",
        separation=5.0,
        sort_ports=False,
    )
    for k, p in enumerate(out_ports):
        c.add_port(
            f"o_out_{k}",
            center=(facet_e, p.dy),
            width=facet_port.width,
            orientation=0,
            layer=facet_port.layer,
            port_type="optical",
        )

    # Die outline centered on the pad rows; the leaf-column routing
    # constraint pins the rows west, so the spare west strip hosts the
    # test structures.
    pad_boxes = [
        i.dbbox() for i in tree.cell.insts if (i.name or "").startswith("pad_")
    ]
    row_center = (min(b.left for b in pad_boxes) + max(b.right for b in pad_boxes)) / 2
    facet_w = min(2 * row_center - facet_e, bbox.left - die_margin)

    # West facet: input with a loopback pair on each side (n=5 array).
    west = c.add_ref(
        gf.c.edge_coupler_array_with_loopback(
            edge_coupler=edge_coupler,
            n=5,
            pitch=coupler_pitch,
            text=None,
            cross_section="strip_cband",
        ),
        name="ec_west",
    )
    west.drotate(180)
    west.dmove((facet_w - west.dbbox().left, 0))
    west.dmove((0, -west.ports["o1"].dy))
    gf.routing.route_single(
        c, west.ports["o1"], tree.ports["o1"], cross_section="strip_cband"
    )
    c.add_port(
        "o_in",
        center=(facet_w, 0),
        width=facet_port.width,
        orientation=180,
        layer=facet_port.layer,
        port_type="optical",
    )

    # Reference MZI in the west strip with its own three couplers.
    test_y = -max(b.top for b in pad_boxes) / 2
    test = c.add_ref(
        gf.c.edge_coupler_array(
            edge_coupler=edge_coupler, n=3, pitch=coupler_pitch, text=None
        ),
        name="ec_test",
    )
    test.drotate(180)
    test.dmove((facet_w - test.dbbox().left, 0))
    test.dmove((0, test_y - test.ports["o2"].dy))
    mzi_test = c.add_ref(sample_mzi_switch(), name="mzi_test")
    mzi_test.dmove((facet_w + 700.0, test_y))
    gf.routing.route_single(
        c, test.ports["o2"], mzi_test.ports["o1"], cross_section="strip_cband"
    )
    t_low, t_mid, t_high = sorted(test.ports, key=lambda p: p.dy)
    up, low = _split_pair(c, mzi_test.ports["o2"], mzi_test.ports["o3"], fanout_sep)
    gf.routing.route_single(c, up, t_high, cross_section="strip_cband")
    gf.routing.route_single(c, low, t_low, cross_section="strip_cband")
    for name, p in (
        ("o_test_in", t_mid),
        ("o_test_up", t_high),
        ("o_test_low", t_low),
    ):
        c.add_port(
            name,
            center=(facet_w, p.dy),
            width=facet_port.width,
            orientation=180,
            layer=facet_port.layer,
            port_type="optical",
        )

    # Floorplan, pass-through, markers, label.
    half_h = (
        max(
            abs(bbox.top),
            abs(bbox.bottom),
            (n_out + 3) / 2 * coupler_pitch + coupler_pitch,
        )
        + 2 * die_margin
    )
    # Pass-through just below the coupler array but well inside the pad
    # rows — never in the die-edge margin under the pads.
    pass_y = -((n_out + 3) / 2 * coupler_pitch + coupler_pitch)
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
    label = c << gf.c.text(text="OPA_MZI_TREE", size=100.0, layer="PAD")
    label.dmove((facet_w + 3 * die_margin, half_h - 2.5 * die_margin))

    return c
