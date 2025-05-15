"""SAX models for Sparameter circuit simulations."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
import sax
from numpy.typing import NDArray

import cspdk.si220.models as csm

nm = 1e-3

FloatArray = NDArray[jnp.floating]
Float = float | FloatArray

# TODO: we probably need specific si500 models...

################
# Straights
################

straight = partial(csm.straight, cross_section="xs_rc")
straight_rc = csm.straight_rc

################
# Bends
################

wire_corner = csm.wire_corner
bend_s = partial(csm.bend_s, cross_section="xs_rc")
bend_euler = partial(csm.bend_euler, cross_section="xs_rc")
bend_euler_rc = csm.bend_euler_rc


################
# Transitions
################

taper = partial(csm.taper, cross_section="xs_rc")
taper_rc = csm.taper_rc

################
# MMIs
################

mmi1x2 = partial(csm.mmi1x2, cross_section="xs_rc")
mmi1x2_rc = csm.mmi1x2_rc

mmi2x2 = partial(csm.mmi2x2, cross_section="xs_rc")
mmi2x2_rc = csm.mmi2x2_rc


##############################
# Evanescent couplers
##############################

coupler_straight = csm.coupler_straight
coupler_symmetric = csm.coupler_symmetric

coupler = partial(csm.coupler, cross_section="xs_rc")
coupler_rc = csm.coupler_rc


##############################
# grating couplers Rectangular
##############################

grating_coupler_rectangular = partial(
    csm.grating_coupler_rectangular, cross_section="xs_rc"
)
grating_coupler_rectangular_rc = csm.grating_coupler_rectangular_rc


##############################
# grating couplers Elliptical
##############################

grating_coupler_elliptical = partial(
    csm.grating_coupler_rectangular, cross_section="xs_rc"
)
grating_coupler_elliptical_rc = csm.grating_coupler_rectangular_rc


################
# MZI
################

# MZIs don't need models. They're composite components.

################
# Packaging
################

# No packaging models

################
# Imported
################

heater = csm.heater


crossing_rc = csm.crossing_rc


################
# Models Dict
################


def get_models() -> dict[str, Callable[..., sax.SDict]]:
    """Return a dictionary of all models in this module."""
    models = {}
    for name, func in list(globals().items()):
        if not callable(func):
            continue
        _func = func
        while isinstance(_func, partial):
            _func = _func.func
        try:
            sig = inspect.signature(_func)
        except ValueError:
            continue
        if str(sig.return_annotation).lower().split(".")[-1] == "sdict":
            models[name] = func
    return models


if __name__ == "__main__":
    print(list(get_models()))
    for name, model in get_models().items():
        try:
            print(name, model())
        except NotImplementedError:
            continue
