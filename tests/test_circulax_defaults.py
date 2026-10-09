"""Every flavour's SAX models must work inside circulax (GDSFactory+ DC sweeps).

circulax bakes a model's signature defaults into its components, and a sweep such as
``circuit.dc(wl=...)`` broadcasts ``wl`` with ``jnp.full_like(<default>, wl)`` to every
component that has it, so ``wl`` needs a concrete default.
"""

from __future__ import annotations

import dataclasses
import importlib

import jax.numpy as jnp
import pytest
from circulax.s_transforms import sax_component

# si220 has its own version of this check in tests/test_si220_unified.py.
FLAVOURS = ["si340", "si500", "sin200", "sin300", "si_sus", "ge_on_si"]

# None means "take it from the cross-section or band" (bend radius, band loss and
# grating bandwidth); a single concrete value would be wrong for the other
# cross-sections, and none of these is broadcast.
NONE_DEFAULTS_ALLOWED = {"radius", "length", "loss", "bandwidth"}
# sax.models.mmi2x2 itself defaults these to None (derived from loss_dB).
NONE_DEFAULTS_ALLOWED |= {"loss_dB_cross", "loss_dB_thru"}


def _models():
    for flavour in FLAVOURS:
        pdk = importlib.import_module(f"cspdk.{flavour}").PDK
        for name in sorted(pdk.models):
            yield pytest.param(pdk.models[name], id=f"{flavour}-{name}")


@pytest.mark.parametrize("model", _models())
def test_model_defaults_work_in_circulax(model) -> None:
    """The model wraps as a circulax component and ``wl`` can be broadcast."""
    component = sax_component(model)
    defaults = {f.name: f.default for f in dataclasses.fields(component)}
    if "wl" in defaults:
        jnp.full_like(defaults["wl"], 1.31)
    none_defaults = {k for k, v in defaults.items() if v is None}
    assert none_defaults <= NONE_DEFAULTS_ALLOWED, none_defaults
