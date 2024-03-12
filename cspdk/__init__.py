from gdsfactory.cross_section import get_cross_sections
from gdsfactory.get_factories import get_cells
from gdsfactory.pdk import Pdk

from cspdk import cells, config, tech
from cspdk.cells import _bend, _straight, _taper
from cspdk.config import PATH
from cspdk.models import get_models
from cspdk.routing import routing_strategies
from cspdk.tech import LAYER, LAYER_STACK, LAYER_VIEWS

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
    name="cornerstone",
    cells=_cells,
    cross_sections=_cross_sections,
    layers=dict(LAYER),
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
__version__ = "0.6.0"
