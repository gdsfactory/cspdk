"""Schematic annotation tests for cspdk.si500 (no SAX models registered)."""

from __future__ import annotations

import pytest
from _schematic_checks import (
    check_ports_subset_of_component,
    check_sax_model_refs,
    check_symbol_present,
)

from cspdk.si500 import PDK


@pytest.fixture(autouse=True)
def _activate_pdk() -> None:
    PDK.activate()


def test_symbol_present() -> None:
    """Every schematic-driven cell exposes a non-empty symbol."""
    check_symbol_present(PDK)


def test_ports_subset_of_component() -> None:
    """Schematic port names are a subset of each Component's ports."""
    check_ports_subset_of_component(PDK)


def test_sax_model_refs() -> None:
    """Only the python module resolution is checked (si500 has no SAX models)."""
    check_sax_model_refs(PDK, has_models=False)
