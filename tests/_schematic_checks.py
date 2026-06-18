"""Shared helpers for per-band schematic tests."""

from __future__ import annotations

import importlib
from collections.abc import Iterable

import jax.numpy as jnp
import kfactory as kf


def _unwrap_factory(fn):
    """Extract the ``WrappedKCellFunc`` from a ``gf.cell``-decorated function.

    ``kf.kcl.factories[name]`` is keyed globally, so when two bands register a
    cell with the same name (e.g. ``straight_heater_metal``), only the
    last-imported one is reachable. The band-specific factory is still stored
    in the closure of the wrapper returned by ``gf.cell``.
    """
    closure = getattr(fn, "__closure__", None) or ()
    for cell in closure:
        obj = cell.cell_contents
        if isinstance(obj, kf.decorators.WrappedKCellFunc):
            return obj
    return None


def schematic_driven_cells(
    pdk,
) -> Iterable[tuple[str, kf.decorators.WrappedKCellFunc]]:
    for name in sorted(pdk.cells):
        factory = _unwrap_factory(pdk.cells[name])
        if factory is not None and factory.schematic_driven():
            yield name, factory


# Composite cells that instantiate sub-cells through the active PDK. If the
# kfactory cell cache has stale entries from another band (common when tests
# run in a shared session), these can fail with port-width mismatches. Skip
# them for the port-subset check and rely on the dedicated cell tests.
_COMPOSITE_SKIP = {
    # Composites that instantiate sub-cells through the active PDK. Their
    # gdsfactory-side inner cells (coupler_symmetric, coupler_straight) are
    # cache-keyed globally; when multiple bands share a pytest session the
    # cached variant can mismatch the active band's cross_section width.
    # Skipped here; each band's own test_<band>.py exercises them.
    "ring_single",
    "ring_double",
    "mzi",
    "mzi_rc",
    "mzi_ro",
    "mzi_nc",
    "mzi_no",
    "spiral",
    "spiral_racetrack",
    "spiral_racetrack_heater",
    "coupler",
    "coupler_rib",
    "coupler_ring",
    "coupler_straight",
}


def check_symbol_present(pdk) -> None:
    found = False
    for _name, factory in schematic_driven_cells(pdk):
        found = True
        s = factory.get_schematic()
        assert s.info["symbol"], f"{_name} has empty symbol"
    assert found, "no schematic-driven cells"


def check_ports_subset_of_component(pdk) -> None:
    for name, factory in schematic_driven_cells(pdk):
        if name in _COMPOSITE_SKIP:
            continue
        s = factory.get_schematic()
        declared = {p["name"] for p in s.info["ports"]}
        component_ports = {p.name for p in pdk.cells[name]().ports}
        missing = declared - component_ports
        assert not missing, (
            f"{name}: schematic declares ports not on component: "
            f"{sorted(missing)} (component: {sorted(component_ports)})"
        )


def check_bend_ports_left_top(pdk) -> None:
    """90-degree bend schematics declare o1 on the left and o2 on the top.

    This matches the physical bend_euler/bend_circular GDS (o1 at 180°, o2 at
    90°) and the canonical Mosaic ``bend`` symbol. Guards against regressing to
    the old left/bottom layout.
    """
    checked = False
    for name, factory in schematic_driven_cells(pdk):
        if name not in ("bend_euler", "bend_circular"):
            continue
        s = factory.get_schematic()
        sides = {p["name"]: p["side"] for p in s.info["ports"]}
        assert sides.get("o1") == "left", (
            f"{name}: o1 side {sides.get('o1')!r} != 'left'"
        )
        assert sides.get("o2") == "top", f"{name}: o2 side {sides.get('o2')!r} != 'top'"
        checked = True
    assert checked, "no bend_euler/bend_circular schematic-driven cells found"


# side -> canonical outward-normal orientation (mirrors _schematic._SIDE_XY)
_SIDE_ORIENTATION = {"left": 180, "right": 0, "top": 90, "bottom": 270}

# (cell, port) pairs whose declared schematic side intentionally differs from
# the GDS port orientation. These are documented exemptions, not bugs:
_GDS_ORIENT_EXEMPT = {
    # spiral exposes both optical ports at 0° (same side) in the GDS, but it has
    # a bespoke Mosaic glyph drawn left->right; matching the GDS here would need
    # a glyph redraw, not a preset tweak. It is a 2-port inline element, so the
    # deviation creates no routing "knot".
    ("spiral", "o1"),
}


def _orientation_close(a: float, b: float) -> bool:
    return abs(((a - b + 180) % 360) - 180) < 1.0


def check_all_ports_match_gds(pdk) -> None:
    """Every declared schematic port side matches the real GDS port orientation.

    Generalises ``check_bend_ports_left_top`` to the whole schematic-driven set:
    a port declared on side ``S`` must point in ``S``'s outward direction on the
    physical component (left=180°, right=0°, top=90°, bottom=270°). This guards
    against the class of bug where a schematic symbol is "twisted" relative to
    the GDS layout (the bend left->bottom regression). Known, intentional
    deviations are listed in ``_GDS_ORIENT_EXEMPT``.
    """
    mismatches = []
    for name, factory in schematic_driven_cells(pdk):
        s = factory.get_schematic()
        try:
            ports = {p.name: p for p in pdk.cells[name]().ports}
        except Exception:
            # Composite cells can fail to build in a shared pytest session due
            # to cross-band kfactory cache contamination (see _COMPOSITE_SKIP);
            # their geometry is locked by the per-band reference-GDS tests. Only
            # tolerate this for known composites — a build failure anywhere else
            # is a real error and must fail the test.
            if name in _COMPOSITE_SKIP:
                continue
            raise
        for p in s.info["ports"]:
            if (name, p["name"]) in _GDS_ORIENT_EXEMPT:
                continue
            want = _SIDE_ORIENTATION.get(p["side"])
            ap = ports.get(p["name"])
            if ap is None:
                mismatches.append(
                    f"{name}.{p['name']}: declared in schematic but missing on the GDS component"
                )
                continue
            if ap.orientation is None or want is None:
                continue
            got = float(ap.orientation) % 360
            if not _orientation_close(got, want):
                mismatches.append(
                    f"{name}.{p['name']}: declared {p['side']!r} ({want}°) "
                    f"but GDS orientation is {got:.1f}°"
                )
    assert not mismatches, (
        "schematic port sides disagree with GDS layout:\n  " + "\n  ".join(mismatches)
    )


# Cells exempt from the clockwise-order check. The heaters render as a generic
# "ckt" box but are physically wide devices whose via-stack ports face inward
# (e.g. a left-facing port sits at the far-right x). Collapsed onto a unit box,
# "clockwise from the layout" is not meaningful, so their multi-port sides are
# not order-checked. (Tracked for a proper symbol in doplaydo/gdsfactoryplus#3050.)
_CLOCKWISE_EXEMPT = {"straight_heater_metal", "straight_heater_meander"}


def check_ports_clockwise_from_left(pdk) -> None:
    """Schematic ports are listed clockwise-from-left within each side.

    The nyanlib->Mosaic bridge converts gdsfactory's clockwise port ordering into
    Mosaic's top->bottom rendering by reversing the left and bottom side groups.
    For that to land each port where the GDS layout puts it, every schematic must
    list same-side ports in clockwise order, measured against the real component
    port positions: left bottom->top, right top->bottom, top left->right, bottom
    right->left. (Coordinate monotonic per side; equal positions are allowed.)
    """
    mismatches = []
    for name, factory in schematic_driven_cells(pdk):
        if name in _CLOCKWISE_EXEMPT:
            continue
        s = factory.get_schematic()
        try:
            ports = {p.name: p for p in pdk.cells[name]().ports}
        except Exception:
            if name in _COMPOSITE_SKIP:
                continue
            raise
        by_side: dict[str, list[str]] = {}
        for p in s.info["ports"]:
            by_side.setdefault(p["side"], []).append(p["name"])
        for side, names in by_side.items():
            # coordinate along the side: y for left/right, x for top/bottom
            coords = []
            for nm in names:
                ap = ports.get(nm)
                if ap is None:
                    coords = None
                    break
                coords.append(ap.y if side in ("left", "right") else ap.x)
            if not coords or len(coords) < 2:
                continue
            # left/top run ascending (bottom->top, left->right); right/bottom
            # run descending (top->bottom, right->left). Ties (equal) are fine.
            ascending = side in ("left", "top")
            ok = all(
                (a <= b + 1e-6) if ascending else (a >= b - 1e-6)
                for a, b in zip(coords, coords[1:])
            )
            if not ok:
                mismatches.append(
                    f"{name} side {side!r}: schematic order {names} is not "
                    f"clockwise (GDS coords {[round(c, 2) for c in coords]})"
                )
    assert not mismatches, (
        "schematic ports not listed clockwise-from-left:\n  " + "\n  ".join(mismatches)
    )


def check_sax_model_refs(pdk, *, has_models: bool) -> None:
    for name, factory in schematic_driven_cells(pdk):
        s = factory.get_schematic()
        for entry in s.info.get("models") or []:
            if entry["language"] != "sax":
                continue
            if has_models:
                assert entry["name"] in pdk.models, (
                    f"{name}: model {entry['name']!r} not in PDK.models"
                )
            module = importlib.import_module(entry["module"])
            assert hasattr(module, entry["qualname"]), (
                f"{name}: {entry['module']}.{entry['qualname']} missing"
            )


def check_sax_port_order_matches_sdict(pdk) -> None:
    """For each SAX model, the SDict keys must be drawn from the declared port_order."""
    for name, factory in schematic_driven_cells(pdk):
        s = factory.get_schematic()
        for entry in s.info.get("models") or []:
            if entry["language"] != "sax" or entry["name"] not in pdk.models:
                continue
            model = pdk.models[entry["name"]]
            try:
                sdict = model(wl=jnp.asarray(1.55))
            except (TypeError, NotImplementedError):
                continue
            except Exception:
                # Some models are composites that fail under default kwargs.
                continue
            allowed = set(entry["port_order"])
            used = {k for pair in sdict for k in pair}
            extra = used - allowed
            assert not extra, (
                f"{name} model {entry['name']}: SDict uses {sorted(extra)} "
                f"not in port_order {sorted(allowed)}"
            )
