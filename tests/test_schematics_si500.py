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
    check_symbol_present(PDK)


def test_ports_subset_of_component() -> None:
    check_ports_subset_of_component(PDK)


def test_sax_model_refs() -> None:
    check_sax_model_refs(PDK, has_models=False)
