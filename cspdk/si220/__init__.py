"""Unified Cornerstone 220 nm silicon photonics PDK."""

from functools import lru_cache

import gdsfactory as gf
from gdsfactory.config import CONF
from gdsfactory.cross_section import get_cross_sections
from gdsfactory.get_factories import get_cells
from gdsfactory.pdk import Pdk

from cspdk.si220 import cells, config, tech
from cspdk.si220.config import PATH
from cspdk.si220.models import get_models
from cspdk.si220.tech import LAYER, LAYER_STACK, LAYER_VIEWS, routing_strategies

_models = get_models()
_cells = get_cells(cells)
_cross_sections = get_cross_sections(tech)

CONF.pdk = "cspdk.si220"

layer_transitions = {
    LAYER.WG: cells.taper,
    LAYER.PAD: cells.taper_metal,
}

# Profiles of ports that gdsfactory creates from a layer and width. Only the
# heater strip of a heater cross section still needs one: an auxiliary strip
# can't name a cross section for its ports. It needs the same radii as this
# PDK's heater cross sections of the same geometry.
port_cross_sections = {
    (LAYER.HEATER.layer, LAYER.HEATER.datatype): gf.cross_section.heater_metal,
}


@lru_cache
def get_pdk() -> Pdk:
    """Return the unified Cornerstone Si220 PDK."""
    return Pdk(
        name="cspdk.si220",
        cells=_cells,
        cross_sections=_cross_sections,  # type: ignore[arg-type]
        layers=LAYER,
        connectivity=tech.CONNECTIVITY,
        layer_stack=LAYER_STACK,
        layer_views=LAYER_VIEWS,
        models=_models,
        routing_strategies=routing_strategies,
        layer_transitions=layer_transitions,
        port_cross_sections=port_cross_sections,
    )


def activate_pdk() -> None:
    """Activate the unified Cornerstone Si220 PDK."""
    get_pdk().activate()


PDK = get_pdk()

__all__ = [
    "LAYER",
    "LAYER_STACK",
    "LAYER_VIEWS",
    "PATH",
    "PDK",
    "activate_pdk",
    "cells",
    "config",
    "tech",
]
