"""Si220 pdk."""

from functools import lru_cache

from gdsfactory.config import CONF
from gdsfactory.cross_section import get_cross_sections
from gdsfactory.get_factories import get_cells
from gdsfactory.pdk import Pdk

from cspdk.si220.cband import cells, config, tech
from cspdk.si220.cband.config import PATH
from cspdk.si220.cband.models import get_models
from cspdk.si220.cband.tech import LAYER, LAYER_STACK, LAYER_VIEWS, routing_strategies

_models = get_models()
_cells = get_cells(cells)
_cross_sections = get_cross_sections(tech)

CONF.pdk = "cspdk.si220.cband"


@lru_cache
def get_pdk() -> Pdk:
    """Return Cornerstone PDK."""
    return Pdk(
        name="cspdk.si220.cband",
        cells=_cells,
        cross_sections=_cross_sections,  # type: ignore
        layers=LAYER,
        layer_stack=LAYER_STACK,
        layer_views=LAYER_VIEWS,
        models=_models,
        routing_strategies=routing_strategies,
    )


def activate_pdk() -> None:
    """Activate Cornerstone Si220 PDK."""
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
