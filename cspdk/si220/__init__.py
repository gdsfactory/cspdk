from gdsfactory.cross_section import get_cross_sections
from gdsfactory.get_factories import get_cells
from gdsfactory.pdk import Pdk

from cspdk.si220 import cells, config, tech
from cspdk.si220.cells import _bend, _straight, _taper
from cspdk.si220.config import PATH
from cspdk.si220.models import get_models
from cspdk.si220.tech import LAYER, LAYER_STACK, LAYER_VIEWS, routing_strategies

_models = get_models()
_cells = get_cells(cells)
_cells.update(
    {
        "_straight": _straight,
        "_bend": _bend,
        "_taper": _taper,
    }
)
_cross_sections = get_cross_sections(tech)
PDK = Pdk(
    name="cornerstone_si220",
    cells=_cells,
    cross_sections=_cross_sections,
    layers=LAYER,
    layer_stack=LAYER_STACK,
    layer_views=LAYER_VIEWS,
    models=_models,
    routing_strategies=routing_strategies,
)
PDK.activate()

__all__ = [
    "LAYER",
    "LAYER_STACK",
    "LAYER_VIEWS",
    "PATH",
    "cells",
    "config",
    "tech",
]
