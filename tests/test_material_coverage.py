"""Every layer-stack material of every flavour has a valid material card.

The cards are in ``cspdk/materials`` (pdk-schema ``MaterialCard`` JSON). Each must
validate, and each must cover the wavelengths its flavours use. Al and TiN (metal
and heater layers) have no card yet: those cases are strict expected failures, so
adding a card makes them fail until they are removed from ``NO_CARD_YET``.
"""

from __future__ import annotations

import importlib
import math

import pytest
from pdk_schema import MaterialCard

from cspdk.materials import material_cards

CARDS = material_cards()
NO_CARD_YET = {"Al", "TiN"}

O_BAND = (1.26, 1.31, 1.36)
C_BAND = (1.5, 1.55, 1.6)
MID_IR = (3.7, 3.8, 3.9)
# Wavelengths (um) each flavour is designed for: band edges and centres.
WAVELENGTHS = {
    "si220": O_BAND + C_BAND,
    "si340": O_BAND + C_BAND,
    "si500": O_BAND + C_BAND,
    "sin200": (0.52, 0.638, 0.78),
    "sin300": O_BAND + C_BAND,
    "si_sus": MID_IR,
    "ge_on_si": MID_IR,
}


def _stack_materials():
    for flavour in WAVELENGTHS:
        stack = importlib.import_module(f"cspdk.{flavour}.tech").LAYER_STACK
        for material in sorted({level.material for level in stack.layers.values()}):
            marks = (
                [pytest.mark.xfail(strict=True, reason="no material card yet")]
                if material in NO_CARD_YET
                else []
            )
            yield pytest.param(
                flavour, material, id=f"{flavour}-{material}", marks=marks
            )


STACK_MATERIALS = list(_stack_materials())


@pytest.mark.parametrize(("flavour", "material"), STACK_MATERIALS)
def test_material_has_a_card(flavour: str, material: str) -> None:
    """The layer stack names a material that has a card, spelled exactly."""
    assert material in CARDS


@pytest.mark.parametrize("name", sorted(CARDS))
def test_card_is_valid(name: str) -> None:
    """Each card validates against the schema and carries its own name."""
    assert MaterialCard.model_validate(CARDS[name]).name == name


@pytest.mark.parametrize(
    ("flavour", "material"),
    [p for p in STACK_MATERIALS if p.values[1] not in NO_CARD_YET],
)
def test_card_covers_flavour_wavelengths(flavour: str, material: str) -> None:
    """The card is valid at every wavelength the flavour is designed for."""
    card = MaterialCard.model_validate(CARDS[material])
    band = card.optical.permittivity.validity.over["wavelength"]
    assert band.unit == "um"
    for wl in WAVELENGTHS[flavour]:
        assert band.min <= wl <= band.max, wl


def test_germanium_matches_an_independent_measurement() -> None:
    """Ge at 3.8 um agrees with Li 1980 (n = 4.0270 at 293 K) to 1e-3.

    Li's table is an independent source, so this catches a mistyped coefficient.
    The Sellmeier form is n^2 = 1 + offset + sum(b wl^2 / (wl^2 - c^2)).
    """
    sellmeier = MaterialCard.model_validate(CARDS["Ge"]).optical.permittivity
    wl2 = 3.8**2
    n2 = 1 + sellmeier.offset
    n2 += sum(t.b * wl2 / (wl2 - t.c_um**2) for t in sellmeier.terms)
    assert math.sqrt(n2) == pytest.approx(4.0270, abs=1e-3)
