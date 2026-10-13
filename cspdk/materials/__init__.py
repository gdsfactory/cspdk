"""Optical material cards for the layer-stack materials (pdk-schema MaterialCard JSON).

- ``Si``: Li 1980, the 293 K column of gsim's ``Si-Li-293K`` table (1.2-14 um).
- ``SiN``: Luke et al. 2015, gsim's ``SiN-Luke`` (0.31-5.5 um).
- ``Si`` and ``SiN`` are copied from gsim's built-in cards (Apache-2.0).
- ``Ge``: Icenogle et al. 1976, Sellmeier fit at 293 K (2.5-12 um).

Al and TiN (metal and heater layers) have no card yet.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any


def material_cards() -> dict[str, dict[str, Any]]:
    """Return the cards as JSON dicts, keyed by layer-stack material name."""
    return {
        path.name.removesuffix(".json"): json.loads(path.read_text())
        for path in resources.files(__name__).iterdir()
        if path.name.endswith(".json")
    }
