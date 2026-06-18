"""Reusable schematic factory for cspdk cells, linked to SAX models.

Mirrors gdsfactory's ``gpdk/_schematic.py`` port-pattern approach and
IHP's ``s.info["models"]`` SPICE-link pattern, but carries SAX model
references instead of SPICE.

Each entry in ``s.info["models"]`` has the shape::

    {
        "language": "sax",
        "name": "mmi1x2",                       # PDK.models key
        "module": "cspdk.si220.cband.models",   # python dotted path
        "qualname": "mmi1x2",                   # attribute in module
        "port_order": ["o1", "o2", "o3"],       # SAX SDict port key order
        "params": {"length": "length", ...},    # component-arg -> model-arg
    }

Some cspdk SAX models dispatch on a ``cross_section`` kwarg (e.g. ``mmi1x2``
routes to ``mmi1x2_strip`` or ``mmi1x2_rib``). Consumers that want the
dispatched variant should pass the component's ``settings["cross_section"]``
into the SAX call; the link here targets the dispatcher name.
"""

from __future__ import annotations

from kfactory.schematic import DSchematic

# ---------------------------------------------------------------------------
# Port patterns
# ---------------------------------------------------------------------------

# 2-port horizontal (straight, taper, transition)
_LEFT_RIGHT = [
    {"name": "o1", "side": "left", "type": "photonic"},
    {"name": "o2", "side": "right", "type": "photonic"},
]

# 2-port 90-degree bend (o1 left/180°, o2 top/90°) — matches bend_euler GDS
_LEFT_TOP = [
    {"name": "o1", "side": "left", "type": "photonic"},
    {"name": "o2", "side": "top", "type": "photonic"},
]

# 1x2 splitter (mmi1x2)
_1X2 = [
    {"name": "o1", "side": "left", "type": "photonic"},
    {"name": "o2", "side": "right", "type": "photonic"},
    {"name": "o3", "side": "right", "type": "photonic"},
]

# 2x2 coupler / mmi2x2 / mzi (with 2x2 splitter)
_2X2 = [
    {"name": "o1", "side": "left", "type": "photonic"},
    {"name": "o2", "side": "left", "type": "photonic"},
    {"name": "o3", "side": "right", "type": "photonic"},
    {"name": "o4", "side": "right", "type": "photonic"},
]

# ring_double: two parallel through-buses (o1-o2 and o3-o4), each left->right.
# Matches the GDS (o1/o3 at 180°/left, o2/o4 at 0°/right) and the Mosaic
# ring-double symbol, which draws two horizontal buses. Using _2X2 here would
# group o1,o2 on the left and o3,o4 on the right, twisting the buses.
_RING_DOUBLE = [
    {"name": "o1", "side": "left", "type": "photonic"},
    {"name": "o2", "side": "right", "type": "photonic"},
    {"name": "o3", "side": "left", "type": "photonic"},
    {"name": "o4", "side": "right", "type": "photonic"},
]

# ring coupler element: bus (o1 left, o4 right) with two ports up to the ring
# (o2, o3 on top). Matches the GDS (o2/o3 at 90°). No dedicated Mosaic symbol;
# renders as a generic box.
_RING_COUPLER = [
    {"name": "o1", "side": "left", "type": "photonic"},
    {"name": "o2", "side": "top", "type": "photonic"},
    {"name": "o3", "side": "top", "type": "photonic"},
    {"name": "o4", "side": "right", "type": "photonic"},
]

# 1x2 MZI (mmi1x2 splitter + mmi1x2 combiner), 3 ports
_MZI_1X2 = [
    {"name": "o1", "side": "left", "type": "photonic"},
    {"name": "o2", "side": "right", "type": "photonic"},
    {"name": "o3", "side": "right", "type": "photonic"},
]

# 4-port crossing (o1 left/180°, o2 top/90°, o3 right/0°, o4 bottom/270°) —
# matches the GDS port orientations and the symmetric Mosaic crossing symbol.
_CROSSING = [
    {"name": "o1", "side": "left", "type": "photonic"},
    {"name": "o2", "side": "top", "type": "photonic"},
    {"name": "o3", "side": "right", "type": "photonic"},
    {"name": "o4", "side": "bottom", "type": "photonic"},
]

# Grating coupler: bus o1 left/180°, fiber port o2 right/0° — matches the GDS
# (o2 at 0°) and the Mosaic grating-coupler symbol, whose fan radiates right.
_GRATING = [
    {"name": "o1", "side": "left", "type": "photonic"},
    {"name": "o2", "side": "right", "type": "photonic"},
]

# Pad with 4 electrical edges
_PAD = [
    {"name": "e1", "side": "left", "type": "electric"},
    {"name": "e2", "side": "top", "type": "electric"},
    {"name": "e3", "side": "right", "type": "electric"},
    {"name": "e4", "side": "bottom", "type": "electric"},
]

# Wire bend / corner (electrical 90° turn): e1 left/180°, e2 top/90° — matches
# the GDS (wire_corner/bend_metal e2 at 90°), mirroring the photonic bend.
_WIRE_BEND = [
    {"name": "e1", "side": "left", "type": "electric"},
    {"name": "e2", "side": "top", "type": "electric"},
]

# Wire straight (electrical)
_WIRE_STRAIGHT = [
    {"name": "e1", "side": "left", "type": "electric"},
    {"name": "e2", "side": "right", "type": "electric"},
]

# Heater with top-only electrical ports (cband straight_heater_metal)
_HEATER_TOP = [
    {"name": "o1", "side": "left", "type": "photonic"},
    {"name": "o2", "side": "right", "type": "photonic"},
    {"name": "l_e2", "side": "top", "type": "electric"},
    {"name": "r_e2", "side": "top", "type": "electric"},
]

# Heater with full via-stack electrical ports (oband straight_heater_metal)
_HEATER_FULL = [
    {"name": "o1", "side": "left", "type": "photonic"},
    {"name": "o2", "side": "right", "type": "photonic"},
    {"name": "l_e1", "side": "left", "type": "electric"},
    {"name": "l_e2", "side": "top", "type": "electric"},
    {"name": "l_e3", "side": "right", "type": "electric"},
    {"name": "l_e4", "side": "bottom", "type": "electric"},
    {"name": "r_e1", "side": "left", "type": "electric"},
    {"name": "r_e2", "side": "top", "type": "electric"},
    {"name": "r_e3", "side": "right", "type": "electric"},
    {"name": "r_e4", "side": "bottom", "type": "electric"},
]


# ---------------------------------------------------------------------------
# Schematic builder
# ---------------------------------------------------------------------------

_SIDE_XY = {
    "left": (-1, 0, 180),
    "right": (1, 0, 0),
    "top": (0, 1, 90),
    "bottom": (0, -1, 270),
}


def _make_schematic(
    symbol: str,
    tags: list[str],
    ports: list[dict],
    models: list[dict] | None,
) -> DSchematic:
    # Deep-copy ports and models: both are lists of dicts shared across all
    # calls via module-level constants (_LEFT_RIGHT, _1X2, ...) and closure
    # variables. A downstream consumer mutating s.info would otherwise
    # corrupt every future schematic built from the same pattern.
    s = DSchematic()
    s.info["symbol"] = symbol
    s.info["tags"] = list(tags)
    s.info["ports"] = [dict(p) for p in ports]
    s.info["models"] = [dict(m) for m in models or []]

    side_counts: dict[str, int] = {}
    for port in ports:
        side_counts[port["side"]] = side_counts.get(port["side"], 0) + 1

    seen_sides: dict[str, int] = {}
    spacing = 0.5
    for port in ports:
        side = port["side"]
        try:
            bx, by, orientation = _SIDE_XY[side]
        except KeyError as exc:
            raise ValueError(
                f"schematic {symbol!r} port {port['name']!r}: unknown side "
                f"{side!r} (expected one of {sorted(_SIDE_XY)})"
            ) from exc
        idx = seen_sides.get(side, 0)
        seen_sides[side] = idx + 1
        total = side_counts[side]
        offset = (idx - (total - 1) / 2) * spacing
        if side in ("left", "right"):
            x, y = bx, by + offset
        else:
            x, y = bx + offset, by

        xs = "metal_routing" if port["type"] == "electric" else "strip"
        s.create_port(
            name=port["name"],
            cross_section=xs,
            x=x,
            y=y,
            orientation=orientation,
        )

    return s


def schematic(
    symbol: str,
    tags: list[str],
    ports: list[dict],
    models: list[dict] | None = None,
):
    """Return a ``schematic_function`` closure for use with ``@gf.cell``."""

    def _schematic_fn(**kwargs) -> DSchematic:
        return _make_schematic(symbol, tags, ports, models)

    return _schematic_fn


def sax_model(
    name: str,
    module: str,
    port_order: list[str],
    qualname: str | None = None,
    params: dict[str, str] | None = None,
) -> dict:
    """Build a SAX model entry for ``s.info["models"]``."""
    return {
        "language": "sax",
        "name": name,
        "module": module,
        "qualname": qualname or name,
        "port_order": port_order,
        "params": params or {},
    }
