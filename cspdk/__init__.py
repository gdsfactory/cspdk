from gdsfactory.cross_section import get_cross_sections
from gdsfactory.get_factories import get_cells
from gdsfactory.pdk import Pdk

from cspdk import cells, config, tech
from cspdk.config import PATH
from cspdk.models import models
from cspdk.tech import LAYER, LAYER_STACK, LAYER_VIEWS

_cells = get_cells(cells)
_cross_sections = get_cross_sections(tech)
PDK = Pdk(
    name="cornerstone",
    cells=_cells,
    cross_sections=_cross_sections,
    layers=dict(LAYER),
    layer_stack=LAYER_STACK,
    layer_views=LAYER_VIEWS,
    models=models,
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
__version__ = "0.4.0"
