"""Cornerstone Si340 PDK."""

from functools import lru_cache

from gdsfactory.config import CONF
from gdsfactory.cross_section import get_cross_sections
from gdsfactory.get_factories import get_cells
from gdsfactory.pdk import Pdk

from cspdk.si340 import cells, config, tech
from cspdk.si340.config import PATH
from cspdk.si340.tech import LAYER, LAYER_STACK, LAYER_VIEWS, routing_strategies

_cells = get_cells(cells)
_cross_sections = get_cross_sections(tech)

CONF.pdk = "cspdk.si340"


layer_transitions = {
    LAYER.WG: cells.taper,
}


@lru_cache
def get_pdk() -> Pdk:
    """Return Cornerstone Si340 PDK."""
    return Pdk(
        name="cornerstone_si340",
        cells=_cells,
        cross_sections=_cross_sections,  # type: ignore
        layers=LAYER,
        layer_stack=LAYER_STACK,
        layer_views=LAYER_VIEWS,
        routing_strategies=routing_strategies,
        layer_transitions=layer_transitions,
    )


def activate_pdk() -> None:
    """Activate Cornerstone Si340 PDK."""
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
