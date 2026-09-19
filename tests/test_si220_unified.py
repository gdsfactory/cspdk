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


@pytest.mark.parametrize(
    "cross_section",
    [si220.tech.strip_oband, si220.tech.strip_oband()],
)
def test_model_dispatch_accepts_oband_factories_and_objects(cross_section) -> None:
    """Resolved cross-sections select the same O-band model as its name."""
    model = si220.PDK.models["straight"]
    expected = model(wl=1.31, length=10.0, cross_section="strip_oband")
    actual = model(wl=1.31, length=10.0, cross_section=cross_section)
    assert np.allclose(actual["o1", "o2"], expected["o1", "o2"])


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (si220.cells.mzi, {}),
        (si220.cells.mzi_lattice, {}),
        (si220.cells.straight_heater_metal, {}),
        (
            si220.cells.die_with_pads,
            {"ngratings": 2, "npads": 1, "with_loopback": False},
        ),
    ],
)
def test_composite_cells_propagate_oband_cross_section(factory, kwargs) -> None:
    """O-band selection reaches every nested optical component."""
    si220.PDK.activate()
    component = factory(cross_section="strip_oband", **kwargs)
    optical_ports = list(component.ports.filter(port_type="optical"))
    assert optical_ports
    assert all(port.width == pytest.approx(0.40) for port in optical_ports)


def test_fiber_container_propagates_oband_cross_section() -> None:
    """The routed device and grating inside a container use O-band ports."""
    si220.PDK.activate()
    component = si220.cells.add_fiber_single(
        cross_section="strip_oband", with_loopback=False
    )
    optical_ports = [
        port
        for instance in component.insts
        for port in instance.ports
        if port.port_type == "optical"
    ]
    assert optical_ports
    assert all(port.width == pytest.approx(0.40) for port in optical_ports)


@pytest.mark.parametrize(
    ("factory", "cross_section"),
    [
        (si220.cells.crossing, "strip_oband"),
        (si220.cells.crossing_rib, "rib_oband"),
    ],
)
def test_fixed_crossings_select_oband_geometry(factory, cross_section: str) -> None:
    """Fixed crossings load the 1310 nm geometry and expose O-band ports."""
    si220.PDK.activate()
    component = factory(cross_section=cross_section)
    assert all(port.width == pytest.approx(0.40) for port in component.ports)


def test_pdk_uses_dispatchable_sax_heater_model() -> None:
    """Optional active models do not replace the shared SAX model."""
    model = si220.PDK.models["straight_heater_metal"]
    result = model(wl=1.31, cross_section="strip_oband")
    assert ("o1", "o2") in result
