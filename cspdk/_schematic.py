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

# 2-port 90-degree bend
_LEFT_BOTTOM = [
    {"name": "o1", "side": "left", "type": "photonic"},
    {"name": "o2", "side": "bottom", "type": "photonic"},
]

# 1x2 splitter (mmi1x2)
_1X2 = [
    {"name": "o1", "side": "left", "type": "photonic"},
    {"name": "o2", "side": "right", "type": "photonic"},
    {"name": "o3", "side": "right", "type": "photonic"},
]

# 2x2 coupler / mmi2x2 / ring_double / mzi (with 2x2 splitter)
_2X2 = [
    {"name": "o1", "side": "left", "type": "photonic"},
    {"name": "o2", "side": "left", "type": "photonic"},
    {"name": "o3", "side": "right", "type": "photonic"},
    {"name": "o4", "side": "right", "type": "photonic"},
]

# 1x2 MZI (mmi1x2 splitter + mmi1x2 combiner), 3 ports
_MZI_1X2 = [
    {"name": "o1", "side": "left", "type": "photonic"},
    {"name": "o2", "side": "right", "type": "photonic"},
    {"name": "o3", "side": "right", "type": "photonic"},
]

# 4-port crossing
_CROSSING = [
    {"name": "o1", "side": "left", "type": "photonic"},
    {"name": "o2", "side": "bottom", "type": "photonic"},
    {"name": "o3", "side": "right", "type": "photonic"},
    {"name": "o4", "side": "top", "type": "photonic"},
]

# Grating coupler (bus left, fiber above)
_GRATING = [
    {"name": "o1", "side": "left", "type": "photonic"},
    {"name": "o2", "side": "top", "type": "photonic"},
]

# Pad with 4 electrical edges
_PAD = [
    {"name": "e1", "side": "left", "type": "electric"},
    {"name": "e2", "side": "top", "type": "electric"},
    {"name": "e3", "side": "right", "type": "electric"},
    {"name": "e4", "side": "bottom", "type": "electric"},
]

# Wire bend (electrical)
_WIRE_BEND = [
    {"name": "e1", "side": "left", "type": "electric"},
    {"name": "e2", "side": "bottom", "type": "electric"},
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
    {"name": "l_e3", "side": "left", "type": "electric"},
    {"name": "l_e4", "side": "bottom", "type": "electric"},
    {"name": "r_e1", "side": "right", "type": "electric"},
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
    s = DSchematic()
    s.info["symbol"] = symbol
    s.info["tags"] = tags
    s.info["ports"] = ports
    s.info["models"] = models or []

    side_counts: dict[str, int] = {}
    for port in ports:
        side_counts[port["side"]] = side_counts.get(port["side"], 0) + 1

    seen_sides: dict[str, int] = {}
    spacing = 0.5
    for port in ports:
        side = port["side"]
        idx = seen_sides.get(side, 0)
        seen_sides[side] = idx + 1
        total = side_counts[side]
        bx, by, orientation = _SIDE_XY[side]
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
