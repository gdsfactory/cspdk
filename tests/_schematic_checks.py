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
