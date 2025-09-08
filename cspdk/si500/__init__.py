"""Cornerstone Si500 PDK."""

from functools import lru_cache

from gdsfactory.config import CONF
from gdsfactory.cross_section import get_cross_sections
from gdsfactory.get_factories import get_cells
from gdsfactory.pdk import Pdk

from cspdk.si500 import cells, config, tech
from cspdk.si500.config import PATH
from cspdk.si500.tech import LAYER, LAYER_STACK, LAYER_VIEWS, routing_strategies

_cells = get_cells(cells)
_cross_sections = get_cross_sections(tech)

CONF.pdk = "cspdk.si500"


@lru_cache
def get_pdk() -> Pdk:
    """Return Cornerstone Si500 PDK."""
    return Pdk(
        name="cornerstone_si500",
        cells=_cells,
        cross_sections=_cross_sections,  # type: ignore
        layers=LAYER,
        layer_stack=LAYER_STACK,
        layer_views=LAYER_VIEWS,
        routing_strategies=routing_strategies,
    )


def activate_pdk() -> None:
    """Activate Cornerstone Si500 PDK."""
    pdk = get_pdk()
    pdk.activate()


PDK = get_pdk()

__all__ = [
    "LAYER",
    "LAYER_STACK",
    "LAYER_VIEWS",
    "PATH",
    "cells",
    "config",
    "tech",
]
