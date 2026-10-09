"""Contract every SAX model must meet so SAX and circulax (GDSFactory+) can use it.

For every SAX model of every flavour:

* it evaluates with no arguments (SAX fills every parameter from its defaults);
* it accepts an array of wavelengths and works under ``jax.jit``;
* a real circulax DC sweep runs through it: circulax bakes the model's defaults into
  a component, traces every numeric parameter, and ``circuit.dc(wl=...)`` broadcasts
  ``wl`` with ``jnp.full_like`` (this is the path GDSFactory+ uses for its sweeps).
"""

from __future__ import annotations

import importlib
import inspect

import jax
import jax.numpy as jnp
import pytest
import sax
from circulax import attach_testbench, compile_circuit
from circulax.components.electronic import Resistor
from circulax.components.photonic import OpticalSource

FLAVOURS = ["si220", "si340", "si500", "sin200", "sin300", "si_sus", "ge_on_si"]

# si220 couplers on main still build a nested SAX circuit with float() on their
# inputs, which cannot be traced; #372 rewrites them.
KNOWN_UNTRACEABLE = {
    "si220-coupler",
    "si220-coupler_rib",
    "si220-coupler_strip",
    "si220-directional_coupler",
}


def _sax_models():
    for flavour in FLAVOURS:
        pdk = importlib.import_module(f"cspdk.{flavour}").PDK
        for name, model in sorted(pdk.models.items()):
            if isinstance(model, type):  # circulax-native (active) models
                continue
            yield pytest.param(model, id=f"{flavour}-{name}")


def _wl0(model) -> float:
    return inspect.signature(model).parameters["wl"].default


def _xfail_if_untraceable(request) -> None:
    if request.node.callspec.id in KNOWN_UNTRACEABLE:
        request.applymarker(
            pytest.mark.xfail(reason="untraceable until #372", strict=True)
        )


@pytest.mark.parametrize("model", _sax_models())
def test_model_evaluates_with_defaults(model) -> None:
    """Every parameter has a usable default."""
    assert sax.get_ports(model())


@pytest.mark.parametrize("model", _sax_models())
def test_model_takes_wavelength_arrays_under_jit(model, request) -> None:
    """A wavelength array gives one value per wavelength, also when jitted."""
    _xfail_if_untraceable(request)
    wl0 = _wl0(model)
    wl = jnp.array([wl0 - 0.01, wl0, wl0 + 0.01])
    result = jax.jit(lambda wl: model(wl=wl))(wl)
    for value in result.values():
        assert jnp.shape(value) in ((), (3,))


@pytest.mark.parametrize("model", _sax_models())
def test_model_runs_in_a_circulax_dc_sweep(model, request) -> None:
    """Drive the first port, load the rest, and sweep wl like GDSFactory+ does."""
    _xfail_if_untraceable(request)
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
