"""CORNERSTONE e-beam workshop PDK -- EVENT-ONLY.

This flavour is for the CORNERSTONE e-beam workshop event only. It is NOT for
regular CORNERSTONE MPW submissions. Its rules are provisional (CORNERSTONE
spec of 17 Oct 2026) and may change; use the cspdk release tagged for the
event. See ``README.md`` in this package.

SOI 220 nm, one 120 nm e-beam etch: draw cores on WG (3/0), extra etch on
EXPOSURE (4/0) and grating teeth on GRA (6/0). ``post_process`` builds the
exposure the foundry writes; ``check`` runs the event design rules.
"""

from functools import lru_cache

from gdsfactory.config import CONF
from gdsfactory.cross_section import get_cross_sections
from gdsfactory.get_factories import get_cells
from gdsfactory.pdk import Pdk

from cspdk.si220_ebeam_event import cells, config, tech
from cspdk.si220_ebeam_event.config import PATH
from cspdk.si220_ebeam_event.drc import Violation, check
from cspdk.si220_ebeam_event.mask import exposure_region, post_process
from cspdk.si220_ebeam_event.models import get_models
from cspdk.si220_ebeam_event.tech import (
    LAYER,
    LAYER_STACK,
    LAYER_VIEWS,
    TECH,
    routing_strategies,
)

_models = get_models()
_cells = get_cells(cells)
_cross_sections = get_cross_sections(tech)

CONF.pdk = "cspdk.si220_ebeam_event"


@lru_cache
def get_pdk() -> Pdk:
    """Return the EVENT-ONLY CORNERSTONE e-beam PDK."""
    return Pdk(
        name="cornerstone_si220_ebeam_event",
        cells=_cells,
        cross_sections=_cross_sections,
        layers=LAYER,
        layer_stack=LAYER_STACK,
        layer_views=LAYER_VIEWS,
        models=_models,
        routing_strategies=routing_strategies,
    )


def activate_pdk() -> None:
    """Activate the EVENT-ONLY CORNERSTONE e-beam PDK."""
    pdk = get_pdk()
    pdk.activate()


PDK = get_pdk()

__all__ = [
    "LAYER",
    "LAYER_STACK",
    "LAYER_VIEWS",
    "PATH",
    "TECH",
    "Violation",
    "cells",
    "check",
    "config",
    "exposure_region",
    "post_process",
    "tech",
]
