"""Contract tests for the unified Cornerstone Si220 PDK."""

from pathlib import Path

import gdsfactory as gf
import numpy as np
import pytest

import cspdk.si220 as si220

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_unified_pdk_is_exported() -> None:
    """The si220 package exposes the single public PDK entry point."""
    assert si220.PDK.name == "cspdk.si220"
    assert si220.PATH.repo == REPO_ROOT


@pytest.mark.parametrize(
    ("name", "width"),
    [
        ("strip_cband", 0.45),
        ("rib_cband", 0.45),
        ("strip_oband", 0.40),
        ("rib_oband", 0.40),
    ],
)
def test_band_cross_section_width(name: str, width: float) -> None:
    """Each named band cross-section resolves to its documented width."""
    si220.PDK.activate()
    assert gf.get_cross_section(name).width == width


def test_band_detection_accepts_registered_names_factories_and_objects() -> None:
    """Band-sensitive defaults work with every supported cross-section spec form."""
    assert si220.tech.get_band("strip_oband") == "oband"
    assert si220.tech.get_band(si220.tech.strip_oband) == "oband"
    assert si220.tech.get_band(si220.tech.strip_oband()) == "oband"
    assert si220.tech.get_band(si220.tech.strip_cband()) == "cband"
    assert si220.tech.is_rib(si220.tech.rib_oband)
    assert si220.tech.is_rib(si220.tech.rib_cband())


def test_band_variants_coexist_without_cell_cache_collision() -> None:
    """Shared factories create distinct geometry for each band in one process."""
    si220.PDK.activate()
    cband = si220.cells.straight(cross_section="strip_cband")
    oband = si220.cells.straight(cross_section="strip_oband")
    assert cband.name != oband.name
    assert cband.ports["o1"].width == 0.45
    assert oband.ports["o1"].width == 0.40


@pytest.mark.parametrize(
    ("cross_section", "coupler_right", "mmi1x2_right", "mmi2x2_right"),
    [
        ("strip_cband", 24.5, 51.0, 62.5),
        ("strip_oband", 30.0, 60.0, 73.5),
    ],
)
def test_band_sensitive_cell_defaults(
    cross_section: str,
    coupler_right: float,
    mmi1x2_right: float,
    mmi2x2_right: float,
) -> None:
    """Shared cells reproduce the established geometry for each band."""
    si220.PDK.activate()
    assert si220.cells.coupler(
        cross_section=cross_section
    ).dbbox().right == pytest.approx(coupler_right)
    assert si220.cells.mmi1x2(
        cross_section=cross_section
    ).dbbox().right == pytest.approx(mmi1x2_right)
    assert si220.cells.mmi2x2(
        cross_section=cross_section
    ).dbbox().right == pytest.approx(mmi2x2_right)


def test_straight_model_dispatches_by_band() -> None:
    """One model name dispatches to the selected band's optical response."""
    model = si220.PDK.models["straight"]
    cband = model(wl=1.55, length=10.0, cross_section="strip_cband")
    oband = model(wl=1.31, length=10.0, cross_section="strip_oband")
    expected_ports = {("o1", "o2"), ("o2", "o1")}
    assert set(cband) == expected_ports
    assert set(oband) == expected_ports
    assert not np.allclose(cband["o1", "o2"], oband["o1", "o2"])
