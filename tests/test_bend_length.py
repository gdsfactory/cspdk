"""Bend models carry the drawn bend's path length, with no ``info`` transport.

A layout netlist carries a cell's *settings*, never its ``info`` dictionary, and no engine
(sax, sox, socklu) merges ``info`` into ``settings``. A bend model that expects its path
length to arrive as info therefore simulates the drawn bend at whatever default it has:
before this migration the si220 ``bend_euler`` model was a 10.0 um straight, so a ring's
FSR came out 9.250 nm instead of 7.708 nm -- 16.7 % off, identically on every engine.

These models instead take the geometry (``radius``/``angle``/``p`` for Euler bends,
``size`` for S-bends) and derive the same path length the cell puts in ``info["length"]``.
The check is exact rather than approximate: the derivation and the cell's own length must
produce the same S-matrix.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

# The flavours this change migrates. The others (sin200, si340/500, si_sus, ge_on_si)
# already derive their path length -- from band tables or their own cross-section maps, and
# some no longer expose ``length`` at all -- so pinning them here would re-test
# ``tests/test_shared_models.py``'s ground under a different set of conventions.
MIGRATED_FLAVOURS = ("si220", "sin300")
# A bend radius a circuit may set instead of the cross-section default. It has to clear the
# cross-section's minimum bend radius, which the cell enforces: si220's strip radius is 10 um
# (min 5), sin300's xs_nc is 30 um (min 30).
NON_DEFAULT_RADIUS = {"si220": 20.0, "sin300": 60.0}
S_BEND_SIZE = (15.0, 2.0)
# The reference length comes from the cell, which reports it rounded to 3 decimals
# (``info["length"] = np.round(path.length(), 3)`` in gdsfactory's bend cells). A 5e-4 um
# rounding step is a phase step of 2*pi*neff/wl * 5e-4 ~ 5e-3 rad, so that bounds the
# S-matrix difference between the derived length and the cell's rounded one; the repo's own
# helper test uses the same 1e-3 um bound (tests/test_shared_models.py). A model that ignores
# the geometry is off by whole microns (10.0 vs 16.637 um), orders of magnitude above this.
TOLERANCE = 5e-3


def _cases():
    """(flavour, model name, geometry) for every shape worth checking.

    No PDK is activated here, and no cell or model object is captured: activating a PDK is a
    global side effect that should not happen at collection time, so the tests below do it.
    """
    for flavour in MIGRATED_FLAVOURS:
        yield flavour, "bend_euler", {}
        yield flavour, "bend_euler", {"radius": NON_DEFAULT_RADIUS[flavour], "p": 0.3}
        yield flavour, "bend_s", {}
        yield flavour, "bend_s", {"size": S_BEND_SIZE}


def _same_response(left: dict, right: dict) -> bool:
    if set(left) != set(right):
        return False
    return all(
        np.allclose(
            np.asarray(left[key]), np.asarray(right[key]), rtol=0, atol=TOLERANCE
        )
        for key in left
    )


def _pdk(flavour: str):
    """Import a flavour's PDK and make it the active one (a cell needs its own)."""
    pdk = importlib.import_module(f"cspdk.{flavour}").PDK
    pdk.activate()
    return pdk


def _case_id(case: tuple) -> str:
    flavour, name, geometry = case
    shape = "-".join(f"{key}{value}" for key, value in geometry.items()) or "default"
    return f"{flavour}-{name}-{shape}"


@pytest.mark.parametrize(
    "flavour, name, geometry",
    [pytest.param(*case, id=_case_id(case)) for case in _cases()],
)
def test_bend_model_derives_the_drawn_length(flavour, name, geometry) -> None:
    """The model's default length is the length of the bend the cell draws."""
    pdk = _pdk(flavour)
    drawn = float(pdk.cells[name](**geometry).info["length"])
    derived = pdk.models[name](**geometry)
    explicit = pdk.models[name](**geometry, length=drawn)
    worst = max(
        np.max(np.abs(np.asarray(derived[key]) - np.asarray(explicit[key])))
        for key in derived
    )
    assert _same_response(derived, explicit), (
        f"{flavour}.{name}(**{geometry}): derived length is not the drawn length "
        f"{drawn} um, max |dS| {worst:.3e}"
    )


def test_bend_model_still_honours_an_explicit_length() -> None:
    """A circuit that sets ``length`` (or receives it) overrides the geometry."""
    pdk = _pdk("si220")
    model = pdk.models["bend_euler"]
    drawn = float(pdk.cells["bend_euler"](radius=10.0, angle=90.0).info["length"])
    assert not _same_response(
        model(radius=10.0, angle=90.0), model(length=drawn + 50.0)
    )


def test_si220_rib_bend_defaults_to_the_rib_radius() -> None:
    """``radius=None`` is the cross-section radius, which for the rib is not the strip's."""
    pdk = _pdk("si220")
    rib = float(pdk.cells["bend_euler"](cross_section="rib_cband").info["length"])
    assert _same_response(
        pdk.models["bend_euler_rib"](), pdk.models["bend_euler_rib"](length=rib)
    )


def test_oband_bend_defaults_to_the_oband_geometry() -> None:
    """The O-band model derives the O-band cell's length, not the C-band one's."""
    pdk = _pdk("si220")
    oband = float(pdk.cells["bend_euler"](cross_section="strip_oband").info["length"])
    model = pdk.models["bend_euler"]
    assert _same_response(
        model(cross_section="strip_oband", wl=1.31),
        model(cross_section="strip_oband", wl=1.31, length=oband),
    )


def test_sin300_bend_defaults_to_the_cross_section_radius() -> None:
    """``xs_nc`` carries ``Tech.radius_nc``, and the model derives that radius' bend."""
    pdk = _pdk("sin300")
    drawn = float(pdk.cells["bend_euler"]().info["length"])
    model = pdk.models["bend_euler"]
    assert _same_response(model(), model(length=drawn))


def test_unknown_cross_section_is_an_error_not_a_silent_default() -> None:
    """A cross-section with no bend radius must not quietly derive some other bend."""
    pdk = _pdk("sin300")
    with pytest.raises(ValueError, match="No SiN300 bend radius"):
        pdk.models["bend_euler"](cross_section="not_a_cross_section")
