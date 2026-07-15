"""Tests for circulax active models."""

from __future__ import annotations

import dataclasses

import pytest

circulax = pytest.importorskip("circulax")

from cspdk.si220.cband.active_models import (  # noqa: E402
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
