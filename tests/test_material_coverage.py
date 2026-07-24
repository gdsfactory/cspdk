"""Pilot coverage test: cspdk material names resolve against pdk-spec.

pdk-spec is a local/unpublished package, so this test is guarded by
``pytest.importorskip`` and skips cleanly wherever it is not installed
(e.g. in CI). It runs once someone installs pdk-spec editable into the
cspdk venv.
"""

import pytest

from cspdk.ge_on_si.tech import LAYER_STACK as LAYER_STACK_GE_ON_SI
from cspdk.si220.cband.tech import LAYER_STACK as LAYER_STACK_SI220_CBAND
from cspdk.si220.oband.tech import LAYER_STACK as LAYER_STACK_SI220_OBAND
from cspdk.si340.tech import LAYER_STACK as LAYER_STACK_SI340
from cspdk.si500.tech import LAYER_STACK as LAYER_STACK_SI500
from cspdk.si_sus.tech import LAYER_STACK as LAYER_STACK_SI_SUS
from cspdk.sin200.tech import LAYER_STACK as LAYER_STACK_SIN200
from cspdk.sin300.tech import LAYER_STACK as LAYER_STACK_SIN300

# local-only for now; skips in CI (and anywhere pdk-spec is not installed)
pdk_spec = pytest.importorskip("pdk_spec")

LAYER_STACKS = {
    "si220.cband": LAYER_STACK_SI220_CBAND,
    "si220.oband": LAYER_STACK_SI220_OBAND,
    "si340": LAYER_STACK_SI340,
    "si500": LAYER_STACK_SI500,
    "si_sus": LAYER_STACK_SI_SUS,
    "sin200": LAYER_STACK_SIN200,
    "sin300": LAYER_STACK_SIN300,
    "ge_on_si": LAYER_STACK_GE_ON_SI,
}


def test_all_materials_resolve():
    """Assert every sub-PDK's LAYER_STACK has no dangling material names."""
    from pdk_spec import check_material_coverage

    material_cards = {**pdk_spec.material_cards}
    dangling = {}
    for name, stack in LAYER_STACKS.items():
        missing = check_material_coverage(stack, material_cards)
        if missing:
            dangling[name] = missing
    assert dangling == {}, f"unresolved material names: {dangling}"
