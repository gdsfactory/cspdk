"""Building a flavour's cells never makes kfactory duplicate a cell under its own name.

A cspdk ``@gf.cell`` wrapper with the same name and arguments as the gdsfactory cell
it calls makes kfactory duplicate the result under the same name and log "When
duplicating a Cell, avoid giving the duplicate the same name" (#402). Duplicate
names can make the GDS/OASIS file unwritable, so any such log fails here.
"""

from __future__ import annotations

import importlib

import gdsfactory as gf
import pytest
from loguru import logger
from test_model_contract import FLAVOURS

DUPLICATE_NAME = "When duplicating a Cell, avoid giving the duplicate the same name"


@pytest.mark.parametrize("flavour", FLAVOURS)
def test_cells_build_without_duplicate_names(flavour: str) -> None:
    """Build every cell from an empty cache and collect duplicate-name errors."""
    pdk = importlib.import_module(f"cspdk.{flavour}").PDK
    pdk.activate()
    gf.clear_cache()
    duplicated: list[str] = []
    current = [""]

    def sink(message) -> None:
        if DUPLICATE_NAME in message:
            duplicated.append(current[0])

    handler = logger.add(sink, level="ERROR")
    try:
        for name in sorted(pdk.cells):
            current[0] = name
            pdk.get_component(name)
    finally:
        logger.remove(handler)
    assert duplicated == []
