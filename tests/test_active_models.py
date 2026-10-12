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


@pytest.mark.parametrize("voltage", [0.0, 1.0, 2.2, 3.0])
def test_circulax_heater_matches_sax_heater_phase(voltage: float) -> None:
    """The circulax heater and the SAX heater give the same transmission (#374).

    Both turn the phase the same way as SAX propagation (``exp(+1j * phi)``), so in
    a mixed circuit the circulax heater's phase adds to the SAX parts' phase. The
    SAX ``vpi`` is chosen so both dissipate to the same phase shift:
    ``pi * eta * V**2 / R == pi * (V / vpi)**2``.
    """
    import jax.numpy as jnp
    import numpy as np

    from cspdk.si220 import PDK

    length, ohms_per_um, eta = 320.0, 0.375, 12.5
    heater = ThermalPhaseShifter(
        ohms_per_um=ohms_per_um,
        eta_pi_per_W=eta,
        length=length,
        loss_dBcm=3.0,
        neff0=2.38,  # the C-band strip neff the SAX model uses at wl0 = 1.55 um
        wl=1.55,
    )
    # Ports (o1, o2, l_e2, r_e2) then the i_fwd state. With o1 = 1 and o2 = 0 the
    # i_fwd residual o2 - T * o1 is -T.
    y = jnp.array([1.0, 0.0, voltage, 0.0, 0.0], dtype=complex)
    residual, _ = ThermalPhaseShifter.solver_call(0.0, y, heater)
    t_circulax = -complex(residual[4])

    vpi = float(np.sqrt(ohms_per_um * length / eta))
    s = PDK.models["straight_heater_metal"](
        wl=1.55, voltage=voltage, vpi=vpi, length=length, loss_dB_cm=3.0
    )
    t_sax = complex(np.asarray(s["o1", "o2"]))

    assert t_circulax == pytest.approx(t_sax, rel=1e-6, abs=1e-9)
