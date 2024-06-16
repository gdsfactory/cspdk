from functools import lru_cache

from gdsfactory.cross_section import get_cross_sections
from gdsfactory.get_factories import get_cells
from gdsfactory.pdk import Pdk

from cspdk.sin300 import cells, config, tech
from cspdk.sin300.config import PATH

# from cspdk.sin300.models import get_models
from cspdk.sin300.tech import LAYER, LAYER_STACK, LAYER_VIEWS, routing_strategies

_models = {}  # get_models()
_cells = get_cells(cells)
_cross_sections = get_cross_sections(tech)


@lru_cache
def get_pdk():
    pdk = Pdk(
        name="cornerstone_sin300",
        cells=_cells,
        cross_sections=_cross_sections,  # type: ignore
        layers=LAYER,
        layer_stack=LAYER_STACK,
        layer_views=LAYER_VIEWS,
        models=_models,
        routing_strategies=routing_strategies,
    )
    return pdk


def activate_pdk():
    pdk = get_pdk()
    pdk.activate()


__all__ = [
    "LAYER",
    "LAYER_STACK",
    "LAYER_VIEWS",
    "PATH",
    "cells",
    "config",
    "tech",
]
