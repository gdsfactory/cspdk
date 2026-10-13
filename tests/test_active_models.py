"""Tests for circulax active models."""

from __future__ import annotations

import dataclasses

import pytest

circulax = pytest.importorskip("circulax")

from cspdk.si220.active_models import (  # noqa: E402
    ThermalPhaseShifter,
    get_active_models,
)


def test_get_active_models_keys() -> None:
    """Verify both heater cell names are registered."""
    models = get_active_models()
    assert "straight_heater_meander" in models
    assert "straight_heater_metal" in models


def test_thermal_phase_shifter_has_length_field() -> None:
    """Model 'length' parameter must match pcell parameter name."""
    fields = {f.name for f in dataclasses.fields(ThermalPhaseShifter)}
    assert "length" in fields
    assert "length_um" not in fields


def test_thermal_phase_shifter_has_ohms_per_um_field() -> None:
    """Resistance must scale with length via ohms_per_um."""
    fields = {f.name for f in dataclasses.fields(ThermalPhaseShifter)}
    assert "ohms_per_um" in fields
    assert "R_ohm" not in fields


def test_thermal_phase_shifter_defaults_match_pcell() -> None:
    """Default length must match pcell default (320 um)."""
    defaults = {f.name: f.default for f in dataclasses.fields(ThermalPhaseShifter)}
    assert defaults["length"] == 320.0


def test_resistance_scales_with_length() -> None:
    """R at default length must be backward-compatible (120 ohm)."""
    defaults = {f.name: f.default for f in dataclasses.fields(ThermalPhaseShifter)}
    ohms_per_um = defaults["ohms_per_um"]
    length = defaults["length"]
    assert ohms_per_um * length == pytest.approx(120.0)


@pytest.mark.parametrize("wl", [1.50, 1.55, 1.60])
@pytest.mark.parametrize("voltage", [0.0, 1.0, 2.2, 3.0])
def test_circulax_heater_matches_sax_heater(wl: float, voltage: float) -> None:
    """The circulax heater and the SAX heater give the same transmission (#374).

    Both use SAX's phase convention (``exp(+1j * phi)``), the same C-band strip
    ``neff``, ``ng``, dispersion and loss, so their defaults agree across the band.
    The SAX ``vpi`` is chosen so both dissipate to the same phase shift:
    ``pi * eta * V**2 / R == pi * (V / vpi)**2``.
    """
    import jax.numpy as jnp
    import numpy as np

    from cspdk.si220 import PDK

    heater = ThermalPhaseShifter(wl=wl)  # every other parameter at its default

    # Drive o1 = 1 with o2 = 0: the i_fwd residual (o2 - T * o1) is then -T.
    names = (*ThermalPhaseShifter.ports, *ThermalPhaseShifter.states)
    values = {"o1": 1.0, "l_e2": voltage}
    y = jnp.array([values.get(name, 0.0) for name in names], dtype=complex)
    residual, _ = ThermalPhaseShifter.solver_call(0.0, y, heater)
    t_circulax = -complex(residual[names.index("i_fwd")])

    # Only the drive and the length are shared; the SAX loss stays at its default,
    # so the test also catches the two models' defaults drifting apart.
    vpi = float(np.sqrt(heater.ohms_per_um * heater.length / heater.eta_pi_per_W))
    s = PDK.models["straight_heater_metal"](
        wl=wl, voltage=voltage, vpi=vpi, length=heater.length
    )
    t_sax = complex(np.asarray(s["o1", "o2"]))

    assert t_circulax == pytest.approx(t_sax, rel=1e-6, abs=1e-9)
