"""Schematic annotation tests for cspdk.si_sus (no SAX models registered)."""

from __future__ import annotations

import pytest
from _schematic_checks import (
    check_all_ports_match_gds,
    check_bend_ports_left_top,
    check_ports_clockwise_from_left,
    check_ports_subset_of_component,
    check_sax_model_refs,
    check_symbol_present,
)

from cspdk.si_sus import PDK


@pytest.fixture(autouse=True)
def _activate_pdk() -> None:
    PDK.activate()


def test_symbol_present() -> None:
    """Every schematic-driven cell exposes a non-empty symbol."""
    check_symbol_present(PDK)


def test_all_ports_match_gds() -> None:
    """Every declared schematic port side matches the GDS port orientation."""
    check_all_ports_match_gds(PDK)


def test_ports_clockwise_from_left() -> None:
    """Schematic ports are listed clockwise-from-left within each side."""
    check_ports_clockwise_from_left(PDK)


def test_bend_ports_left_top() -> None:
    """90-degree bend schematics declare o1 left, o2 top (matches GDS)."""
    check_bend_ports_left_top(PDK)


def test_ports_subset_of_component() -> None:
    """Schematic port names are a subset of each Component's ports."""
    check_ports_subset_of_component(PDK)


def test_sax_model_refs() -> None:
    """Only the python module resolution is checked (si500 has no SAX models)."""
    check_sax_model_refs(PDK, has_models=False)
