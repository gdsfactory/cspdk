"""Si220 models: JIT and gradients, rib defaults, transition lengths, heater physics."""

from __future__ import annotations

import gdsfactory as gf
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sax

import cspdk.si220 as si220

MODELS = si220.PDK.models


@pytest.mark.parametrize("name", ["coupler", "coupler_ring"])
def test_coupler_models_jit_and_differentiate(name: str) -> None:
    """Couplers have no nested circuit, so they JIT over wl and differentiate in gap."""
    jitted = jax.jit(lambda wl: MODELS[name](wl=wl)["o1", "o3"])
    assert np.isfinite(complex(jitted(1.55)))

    grad = jax.grad(
        lambda gap: jnp.abs(MODELS[name](wl=1.55, gap=gap)["o1", "o3"]) ** 2
    )
    assert np.isfinite(float(grad(0.27)))


def _through(model, **settings) -> complex:
    return complex(np.asarray(model(**settings)["o1", "o2"]).ravel()[0])


def test_rib_models_default_to_rib() -> None:
    """``coupler_rib`` with no cross-section is the rib coupler, not the strip one."""
    default = MODELS["coupler_rib"](wl=1.55)["o1", "o3"]
    rib = MODELS["coupler"](wl=1.55, cross_section="rib_cband")["o1", "o3"]
    assert np.allclose(default, rib)


@pytest.mark.parametrize("name", ["bend_euler_rib", "taper_rib"])
def test_rib_models_stay_rib_for_any_cross_section(name: str) -> None:
    """A model named *_rib keeps rib geometry even if handed a strip cross-section."""
    strip = _through(MODELS[name], wl=1.31, cross_section="strip_oband")
    rib = _through(MODELS[name], wl=1.31, cross_section="rib_oband")
    assert np.isclose(strip, rib)


def _two_in_series(cell: str, **settings) -> gf.Component:
    c = gf.Component()
    first = c << gf.get_component(cell, **settings)
    second = c << gf.get_component(cell, **settings)
    second.connect("o1", first.ports["o2"])
    c.add_port("o1", port=first.ports["o1"])
    c.add_port("o2", port=second.ports["o2"])
    return c


@pytest.mark.parametrize(
    ("cell", "cross_section", "wl"),
    [
        ("trans_rib10", "strip_cband", 1.55),
        ("trans_rib50", "strip_cband", 1.55),
        ("trans_rib50", "strip_oband", 1.31),
    ],
)
def test_trans_rib_keeps_its_length_in_a_circuit(
    cell: str, cross_section: str, wl: float
) -> None:
    """Two fixed-length transitions in series act as one taper of twice the length."""
    si220.PDK.activate()
    circuit, _ = sax.circuit(
        _two_in_series(cell, cross_section=cross_section).get_netlist(), MODELS
    )
    result = jax.block_until_ready(circuit(wl=jnp.array([wl])))
    length = 2 * float(cell.removeprefix("trans_rib"))
    expected = MODELS["taper_strip_to_ridge"](
        wl=jnp.array([wl]), length=length, cross_section=cross_section
    )
    assert np.allclose(result["o1", "o2"], expected["o1", "o2"])


@pytest.mark.parametrize("name", ["straight_heater_metal", "straight_heater_meander"])
@pytest.mark.parametrize(
    ("cross_section", "wl"), [("strip_cband", 1.55), ("strip_oband", 1.31)]
)
def test_unpowered_heater_is_the_band_strip_waveguide(
    name: str, cross_section: str, wl: float
) -> None:
    """With no voltage, a heater propagates like the band's dispersive strip waveguide."""
    wls = jnp.linspace(wl - 0.02, wl + 0.02, 5)
    heater = MODELS[name](wl=wls, length=320, cross_section=cross_section)
    straight = MODELS["straight"](wl=wls, length=320, cross_section=cross_section)
    assert np.allclose(heater["o1", "o2"], straight["o1", "o2"])


@pytest.mark.parametrize(
    ("voltage", "phase"), [(1.0, np.pi), (1 / np.sqrt(2), np.pi / 2), (-1.0, np.pi)]
)
def test_heater_phase_scales_with_power(voltage: float, phase: float) -> None:
    """Heater phase grows with dissipated power (V squared), positive like extra length."""
    off = _through(MODELS["straight_heater_metal"], wl=1.55, voltage=0.0, vpi=1.0)
    on = _through(MODELS["straight_heater_metal"], wl=1.55, voltage=voltage, vpi=1.0)
    assert np.isclose(np.angle(on / off) % (2 * np.pi), phase)
