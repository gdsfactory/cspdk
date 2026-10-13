"""Cells reject gaps below their flavour's design-rule minimum (cspdk._checks)."""

from __future__ import annotations

import importlib

import gdsfactory as gf
import pytest

from cspdk._checks import check_min

# (flavour, cell, parameter); the minimum is TECH.gap_min, or TECH.grating_gap_min
# for grating_line_width.
CHECKED = [
    ("si220", "coupler", "gap"),
    ("si220", "coupler_rib", "gap"),
    ("si220", "coupler_ring", "gap"),
    ("si220", "ring_single", "gap"),
    ("si220", "ring_double", "gap"),
    ("si220", "ring_double", "gap_top"),
    ("si220", "ring_double", "gap_bot"),
    ("si220", "mmi1x2", "gap_mmi"),
    ("si220", "mmi2x2", "gap_mmi"),
    ("si220", "grating_coupler_elliptical", "grating_line_width"),
    *(
        (flavour, cell, parameter)
        for flavour in ("si340", "si500", "sin200", "sin300")
        for cell, parameter in (
            ("coupler", "gap"),
            ("coupler_straight", "gap"),
            ("mmi1x2", "gap_mmi"),
            ("mmi2x2", "gap_mmi"),
            ("grating_coupler_elliptical", "grating_line_width"),
        )
    ),
]


def _minimum(flavour: str, parameter: str) -> float:
    tech = importlib.import_module(f"cspdk.{flavour}.tech").TECH
    return tech.grating_gap_min if parameter == "grating_line_width" else tech.gap_min


@pytest.mark.parametrize(("flavour", "cell", "parameter"), CHECKED)
def test_below_minimum_is_rejected(flavour: str, cell: str, parameter: str) -> None:
    """A value 1 nm below the minimum raises, naming the rule and its source."""
    pdk = importlib.import_module(f"cspdk.{flavour}").PDK
    pdk.activate()
    minimum = _minimum(flavour, parameter)
    with pytest.raises(
        ValueError, match=f"{parameter}=.* below the {minimum} um minimum"
    ):
        gf.get_component(cell, **{parameter: round(minimum - 0.001, 3)})


def test_minimum_itself_passes() -> None:
    """Equality passes, as in the design-rule tables; None (cell default) is not checked."""
    check_min("gap", 0.2, 0.2, "table")
    check_min("gap", None, 0.2, "table")
    with pytest.raises(
        ValueError, match=r"gap=0\.199 um is below the 0\.2 um minimum \(table\)"
    ):
        check_min("gap", 0.199, 0.2, "table")
