"""Contract every SAX model must meet so SAX and circulax (GDSFactory+) can use it.

For every SAX model of every flavour:

* it evaluates with no arguments (SAX fills every parameter from its defaults);
* it accepts an array of wavelengths and works under ``jax.jit``;
* a real circulax DC sweep runs through it: circulax bakes the model's defaults into
  a component, traces every numeric parameter, and ``circuit.dc(wl=...)`` broadcasts
  ``wl`` with ``jnp.full_like`` (this is the path GDSFactory+ uses for its sweeps);
* only parameters that are never broadcast default to ``None``;
* band aliases (``straight_so``, ``mmi1x2_n638``, ...) default ``wl`` to their band.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect

import jax
import jax.numpy as jnp
import pytest
import sax
from circulax import attach_testbench, compile_circuit
from circulax.components.electronic import Resistor
from circulax.components.photonic import OpticalSource
from circulax.s_transforms import sax_component

FLAVOURS = ["si220", "si340", "si500", "sin200", "sin300", "si_sus", "ge_on_si"]

# None means "take it from the cross-section or band" (bend radius, band loss and
# grating bandwidth, or a path length derived from the geometry); a single concrete
# value would be wrong for the other cross-sections, and none of these is broadcast.
NONE_DEFAULTS_ALLOWED = {"radius", "length", "loss", "bandwidth"}
# sax.models.mmi2x2 itself defaults these to None (derived from loss_dB).
NONE_DEFAULTS_ALLOWED |= {"loss_dB_cross", "loss_dB_thru"}
# si220 couplers derive their geometry from the cross-section.
NONE_DEFAULTS_ALLOWED_SI220 = NONE_DEFAULTS_ALLOWED | {"offset", "bend_radius"}

# Band-alias suffix -> band centre wavelength (um). si220 has its own band checks
# (tests/test_si220_unified.py); si_sus and ge_on_si have a single band, no aliases.
BAND_CENTRES = {
    "si340": {"sc": 1.55, "so": 1.31, "rc": 1.55},
    "si500": {"rc": 1.55, "ro": 1.31},
    "sin200": {"n780": 0.78, "n638": 0.638, "n520": 0.52},
    "sin300": {"nc": 1.55, "no": 1.31},
}


def _flavour_models():
    for flavour in FLAVOURS:
        pdk = importlib.import_module(f"cspdk.{flavour}").PDK
        for name, model in sorted(pdk.models.items()):
            if isinstance(model, type):  # circulax-native (active) models
                continue
            yield flavour, name, model


def _sax_models():
    for flavour, name, model in _flavour_models():
        yield pytest.param(model, id=f"{flavour}-{name}")


def _band_aliases():
    for flavour, name, model in _flavour_models():
        centres = BAND_CENTRES.get(flavour, {})
        band = name.rsplit("_", 1)[-1]
        if band in centres:
            yield pytest.param(model, centres[band], id=f"{flavour}-{name}")


def _wl0(model) -> float:
    return inspect.signature(model).parameters["wl"].default


@pytest.mark.parametrize("model", _sax_models())
def test_model_evaluates_with_defaults(model) -> None:
    """Every parameter has a usable default."""
    assert sax.get_ports(model())


@pytest.mark.parametrize("model", _sax_models())
def test_model_takes_wavelength_arrays_under_jit(model) -> None:
    """A wavelength array gives one value per wavelength, also when jitted."""
    wl0 = _wl0(model)
    wl = jnp.array([wl0 - 0.01, wl0, wl0 + 0.01])
    result = jax.jit(lambda wl: model(wl=wl))(wl)
    for value in result.values():
        assert jnp.shape(value) in ((), (3,))


@pytest.mark.parametrize("model", _sax_models())
def test_model_runs_in_a_circulax_dc_sweep(model) -> None:
    """Drive the first port, load the rest, and sweep wl like GDSFactory+ does."""
    first, *rest = sorted(sax.get_ports(model()))
    device = {
        "instances": {"dut": {"component": "dut"}},
        "ports": {port: f"dut,{port}" for port in (first, *rest)},
    }
    netlist = attach_testbench(
        device,
        sources={first: {"component": "source"}},
        loads={port: {"component": "load", "settings": {"R": 1.0}} for port in rest},
    )
    circuit = compile_circuit(
        netlist, {"dut": model, "source": OpticalSource, "load": Resistor}
    )
    wl0 = _wl0(model)
    solution = circuit.dc(wl=jnp.array([wl0 - 0.01, wl0, wl0 + 0.01]))
    assert bool(jnp.all(jnp.isfinite(solution)))


@pytest.mark.parametrize("model", _sax_models())
def test_model_defaults_work_in_circulax(model, request) -> None:
    """The model wraps as a circulax component; only unbroadcast params are None."""
    component = sax_component(model)
    defaults = {f.name: f.default for f in dataclasses.fields(component)}
    if "wl" in defaults:
        jnp.full_like(defaults["wl"], 1.31)
    flavour = request.node.callspec.id.split("-", 1)[0]
    allowed = (
        NONE_DEFAULTS_ALLOWED_SI220 if flavour == "si220" else NONE_DEFAULTS_ALLOWED
    )
    none_defaults = {k for k, v in defaults.items() if v is None}
    assert none_defaults <= allowed, none_defaults


@pytest.mark.parametrize(("model", "centre"), _band_aliases())
def test_band_alias_defaults_to_band_centre(model, centre) -> None:
    """A band alias evaluated with its defaults runs at its band's centre."""
    assert _wl0(model) == pytest.approx(centre)
