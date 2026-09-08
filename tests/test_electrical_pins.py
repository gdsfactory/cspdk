"""Tests that logical electrical pins are registered on cells with electrical ports."""

from __future__ import annotations

import pytest

from cspdk.si220.cband import PDK as si220_cband_PDK
from cspdk.si220.oband import PDK as si220_oband_PDK
from cspdk.si340 import PDK as si340_PDK
from cspdk.si500 import PDK as si500_PDK
from cspdk.sin200 import PDK as sin200_PDK
from cspdk.sin300 import PDK as sin300_PDK

SI220_OBAND_EXPECTED_PIN_NAMES: dict[str, set[str]] = {
    "bend_metal": {"e1", "e2"},
    "bend_s_metal": {"e1", "e2"},
    "pad": {"pad"},
    "spiral_racetrack_heater": {"top", "bot"},
    "straight_heater_meander": {"l", "r"},
    "straight_heater_metal": {"l", "r"},
    "straight_metal": {"e1", "e2"},
    "taper_metal": {"e1", "e2"},
    "via_stack_heater_mtop": {"pad"},
    "wire_corner": {"e1", "e2"},
    "wire_corner45": {"e1", "e2"},
    "wire_corner45_straight": {"e1", "e2"},
}

SI220_CBAND_EXPECTED_PIN_NAMES: dict[str, set[str]] = {
    "bend_metal": {"e1", "e2"},
    "bend_s_metal": {"e1", "e2"},
    "pad": {"pad"},
    "spiral_racetrack_heater": {"top", "bot"},
    "straight_heater_meander": {"l", "r"},
    "straight_heater_metal": {"l", "r"},
    "straight_metal": {"e1", "e2"},
    "taper_metal": {"e1", "e2"},
    "via_stack_heater_mtop": {"pad"},
    "wire_corner": {"e1", "e2"},
    "wire_corner45": {"e1", "e2"},
    "wire_corner45_straight": {"e1", "e2"},
}

SI340_EXPECTED_PIN_NAMES: dict[str, set[str]] = {
    "compass": {"pad"},
    "pad": {"pad"},
    "rectangle": {"pad"},
    "wire_corner": {"e1", "e2"},
}

SIN200_EXPECTED_PIN_NAMES: dict[str, set[str]] = {
    "compass": {"pad"},
    "pad": {"pad"},
    "rectangle": {"pad"},
    "wire_corner": {"e1", "e2"},
}

SIN300_EXPECTED_PIN_NAMES: dict[str, set[str]] = {
    "compass": {"pad"},
    "pad": {"pad"},
    "rectangle": {"pad"},
    "wire_corner": {"e1", "e2"},
}

SI500_EXPECTED_PIN_NAMES: dict[str, set[str]] = {
    "compass": {"pad"},
    "pad": {"pad"},
    "rectangle": {"pad"},
    "wire_corner": {"e1", "e2"},
}

SI220_OBAND_CELLS = [pytest.param(n, id=n) for n in SI220_OBAND_EXPECTED_PIN_NAMES]
SI220_CBAND_CELLS = [pytest.param(n, id=n) for n in SI220_CBAND_EXPECTED_PIN_NAMES]
SI340_CELLS = [pytest.param(n, id=n) for n in SI340_EXPECTED_PIN_NAMES]
SIN200_CELLS = [pytest.param(n, id=n) for n in SIN200_EXPECTED_PIN_NAMES]
SIN300_CELLS = [pytest.param(n, id=n) for n in SIN300_EXPECTED_PIN_NAMES]
SI500_CELLS = [pytest.param(n, id=n) for n in SI500_EXPECTED_PIN_NAMES]


# ── si220 oband ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cell_name", SI220_OBAND_CELLS)
def test_si220_oband_logical_pin_registered(cell_name: str) -> None:
    """Si220 O-band cell has logical pins."""
    si220_oband_PDK.activate()
    c = si220_oband_PDK.cells[cell_name]()
    assert c.pins, f"{cell_name} should have logical pins"


@pytest.mark.parametrize("cell_name", SI220_OBAND_CELLS)
def test_si220_oband_pin_type_is_dc(cell_name: str) -> None:
    """Si220 O-band pin type is DC."""
    si220_oband_PDK.activate()
    c = si220_oband_PDK.cells[cell_name]()
    for pin in c.pins:
        assert pin.pin_type == "DC", (
            f"{cell_name} pin {pin.name!r}: expected pin_type='DC', got {pin.pin_type!r}"
        )


@pytest.mark.parametrize("cell_name", SI220_OBAND_CELLS)
def test_si220_oband_expected_pin_names(cell_name: str) -> None:
    """Si220 O-band cell has expected pin names."""
    si220_oband_PDK.activate()
    c = si220_oband_PDK.cells[cell_name]()
    expected = SI220_OBAND_EXPECTED_PIN_NAMES[cell_name]
    actual = {pin.name for pin in c.pins}
    assert expected.issubset(actual), (
        f"{cell_name}: expected pins {expected} ⊄ actual {actual}"
    )


# ── si220 cband ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cell_name", SI220_CBAND_CELLS)
def test_si220_cband_logical_pin_registered(cell_name: str) -> None:
    """Si220 C-band cell has logical pins."""
    si220_cband_PDK.activate()
    c = si220_cband_PDK.cells[cell_name]()
    assert c.pins, f"{cell_name} should have logical pins"


@pytest.mark.parametrize("cell_name", SI220_CBAND_CELLS)
def test_si220_cband_pin_type_is_dc(cell_name: str) -> None:
    """Si220 C-band pin type is DC."""
    si220_cband_PDK.activate()
    c = si220_cband_PDK.cells[cell_name]()
    for pin in c.pins:
        assert pin.pin_type == "DC", (
            f"{cell_name} pin {pin.name!r}: expected pin_type='DC', got {pin.pin_type!r}"
        )


@pytest.mark.parametrize("cell_name", SI220_CBAND_CELLS)
def test_si220_cband_expected_pin_names(cell_name: str) -> None:
    """Si220 C-band cell has expected pin names."""
    si220_cband_PDK.activate()
    c = si220_cband_PDK.cells[cell_name]()
    expected = SI220_CBAND_EXPECTED_PIN_NAMES[cell_name]
    actual = {pin.name for pin in c.pins}
    assert expected.issubset(actual), (
        f"{cell_name}: expected pins {expected} ⊄ actual {actual}"
    )


# ── si340 ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cell_name", SI340_CELLS)
def test_si340_logical_pin_registered(cell_name: str) -> None:
    """Si340 cell has logical pins."""
    si340_PDK.activate()
    c = si340_PDK.cells[cell_name]()
    assert c.pins, f"{cell_name} should have logical pins"


@pytest.mark.parametrize("cell_name", SI340_CELLS)
def test_si340_pin_type_is_dc(cell_name: str) -> None:
    """Si340 pin type is DC."""
    si340_PDK.activate()
    c = si340_PDK.cells[cell_name]()
    for pin in c.pins:
        assert pin.pin_type == "DC", (
            f"{cell_name} pin {pin.name!r}: expected pin_type='DC', got {pin.pin_type!r}"
        )


@pytest.mark.parametrize("cell_name", SI340_CELLS)
def test_si340_expected_pin_names(cell_name: str) -> None:
    """Si340 cell has expected pin names."""
    si340_PDK.activate()
    c = si340_PDK.cells[cell_name]()
    expected = SI340_EXPECTED_PIN_NAMES[cell_name]
    actual = {pin.name for pin in c.pins}
    assert expected.issubset(actual), (
        f"{cell_name}: expected pins {expected} ⊄ actual {actual}"
    )


# ── sin200 ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cell_name", SIN200_CELLS)
def test_sin200_logical_pin_registered(cell_name: str) -> None:
    """SiN200 cell has logical pins."""
    sin200_PDK.activate()
    c = sin200_PDK.cells[cell_name]()
    assert c.pins, f"{cell_name} should have logical pins"


@pytest.mark.parametrize("cell_name", SIN200_CELLS)
def test_sin200_pin_type_is_dc(cell_name: str) -> None:
    """SiN200 pin type is DC."""
    sin200_PDK.activate()
    c = sin200_PDK.cells[cell_name]()
    for pin in c.pins:
        assert pin.pin_type == "DC", (
            f"{cell_name} pin {pin.name!r}: expected pin_type='DC', got {pin.pin_type!r}"
        )


@pytest.mark.parametrize("cell_name", SIN200_CELLS)
def test_sin200_expected_pin_names(cell_name: str) -> None:
    """SiN200 cell has expected pin names."""
    sin200_PDK.activate()
    c = sin200_PDK.cells[cell_name]()
    expected = SIN200_EXPECTED_PIN_NAMES[cell_name]
    actual = {pin.name for pin in c.pins}
    assert expected.issubset(actual), (
        f"{cell_name}: expected pins {expected} ⊄ actual {actual}"
    )


# ── sin300 ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cell_name", SIN300_CELLS)
def test_sin300_logical_pin_registered(cell_name: str) -> None:
    """SiN300 cell has logical pins."""
    sin300_PDK.activate()
    c = sin300_PDK.cells[cell_name]()
    assert c.pins, f"{cell_name} should have logical pins"


@pytest.mark.parametrize("cell_name", SIN300_CELLS)
def test_sin300_pin_type_is_dc(cell_name: str) -> None:
    """SiN300 pin type is DC."""
    sin300_PDK.activate()
    c = sin300_PDK.cells[cell_name]()
    for pin in c.pins:
        assert pin.pin_type == "DC", (
            f"{cell_name} pin {pin.name!r}: expected pin_type='DC', got {pin.pin_type!r}"
        )


@pytest.mark.parametrize("cell_name", SIN300_CELLS)
def test_sin300_expected_pin_names(cell_name: str) -> None:
    """SiN300 cell has expected pin names."""
    sin300_PDK.activate()
    c = sin300_PDK.cells[cell_name]()
    expected = SIN300_EXPECTED_PIN_NAMES[cell_name]
    actual = {pin.name for pin in c.pins}
    assert expected.issubset(actual), (
        f"{cell_name}: expected pins {expected} ⊄ actual {actual}"
    )


# ── si500 ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cell_name", SI500_CELLS)
def test_si500_logical_pin_registered(cell_name: str) -> None:
    """Si500 cell has logical pins."""
    si500_PDK.activate()
    c = si500_PDK.cells[cell_name]()
    assert c.pins, f"{cell_name} should have logical pins"


@pytest.mark.parametrize("cell_name", SI500_CELLS)
def test_si500_pin_type_is_dc(cell_name: str) -> None:
    """Si500 pin type is DC."""
    si500_PDK.activate()
    c = si500_PDK.cells[cell_name]()
    for pin in c.pins:
        assert pin.pin_type == "DC", (
            f"{cell_name} pin {pin.name!r}: expected pin_type='DC', got {pin.pin_type!r}"
        )


@pytest.mark.parametrize("cell_name", SI500_CELLS)
def test_si500_expected_pin_names(cell_name: str) -> None:
    """Si500 cell has expected pin names."""
    si500_PDK.activate()
    c = si500_PDK.cells[cell_name]()
    expected = SI500_EXPECTED_PIN_NAMES[cell_name]
    actual = {pin.name for pin in c.pins}
    assert expected.issubset(actual), (
        f"{cell_name}: expected pins {expected} ⊄ actual {actual}"
    )
