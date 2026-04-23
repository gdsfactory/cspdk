"""Schematic annotation tests for cspdk.sin300."""

from __future__ import annotations

import pytest
from _schematic_checks import (
    check_ports_subset_of_component,
    check_sax_model_refs,
    check_sax_port_order_matches_sdict,
    check_symbol_present,
)

from cspdk.sin300 import PDK


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
    """Every SAX model entry resolves to PDK.models and its python module."""
    check_sax_model_refs(PDK, has_models=True)


def test_sax_port_order_matches_sdict() -> None:
    """Each SAX model's SDict keys are drawn from the declared port_order."""
    check_sax_port_order_matches_sdict(PDK)
