from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import partial

import gplugins.sax.models as sm
import jax.numpy as jnp
import sax
from gplugins.sax.models import bend as __bend
from gplugins.sax.models import straight as __straight
from numpy.typing import NDArray

nm = 1e-3

FloatArray = NDArray[jnp.floating]
Float = float | FloatArray

################
# Straights
################


def _straight(
    *,
    wl: Float = 1.55,
    length: Float = 10.0,
    loss: Float = 0.0,
    cross_section: str = "xs_sc",
) -> sax.SDict:
    if cross_section.endswith("nc"):
        return __straight(
            wl=wl,  # type: ignore
            length=length,  # type: ignore
            loss=loss,  # type: ignore
            wl0=1.55,
            neff=1.6,
            ng=1.94,
        )
    elif cross_section.endswith("no"):
        return __straight(
            wl=wl,  # type: ignore
            length=length,  # type: ignore
            loss=loss,  # type: ignore
            wl0=1.31,
            neff=1.63,
            ng=2.00,
        )
    else:
        return __straight(
            wl=wl,  # type: ignore
            length=length,  # type: ignore
            loss=loss,  # type: ignore
            wl0=1.55,
            neff=2.38,
            ng=4.30,
        )


straight_nc = partial(_straight, cross_section="xs_nc")
straight_no = partial(_straight, cross_section="xs_no")

################
# Bends
################
bend_s = _straight


def _bend(wl: Float = 1.5, length: Float = 20.0, loss: Float = 0.03) -> sax.SDict:
    return __bend(
        wl=wl,  # type: ignore
        length=length,  # type: ignore
        loss=loss,  # type: ignore
    )


bend_nc = partial(_bend, loss=0.03)
bend_no = partial(_bend, loss=0.03)


################
# MMIs
################
_mmi1x2 = sm.mmi1x2
_mmi2x2 = sm.mmi2x2

_mmi1x2_o = partial(_mmi1x2, wl0=1.31)
_mmi1x2_c = partial(_mmi1x2, wl0=1.55)
_mmi2x2_o = partial(_mmi2x2, wl0=1.31)
_mmi2x2_c = partial(_mmi2x2, wl0=1.55)

mmi1x2_nc = _mmi1x2_c
mmi1x2_no = _mmi1x2_o

mmi2x2_nc = _mmi2x2_c
mmi2x2_no = _mmi2x2_o

##############################
# Evanescent couplers
##############################

_coupler = _mmi2x2
_coupler_o = partial(_coupler, wl0=1.31)
_coupler_c = partial(_coupler, wl0=1.55)
coupler_nc = _coupler_c
coupler_no = _coupler_o

##############################
# grating couplers Rectangular
##############################
_gc_rectangular = sm.grating_coupler
_gcro = partial(_gc_rectangular, loss=6, bandwidth=35 * nm, wavelength=1.31)
_gcrc = partial(_gc_rectangular, loss=6, bandwidth=35 * nm, wavelength=1.55)
gc_rectangular_nc = _gcrc
gc_rectangular_no = _gcro

##############################
# grating couplers Elliptical
##############################
_gc_elliptical = _gc_rectangular
_gceo = partial(_gc_elliptical, loss=6, bandwidth=35 * nm, wavelength=1.31)
_gcec = partial(_gc_elliptical, loss=6, bandwidth=35 * nm, wavelength=1.55)
gc_elliptical_nc = _gcec
gc_elliptical_no = _gceo


################
# Models Dict
################
def get_models() -> dict[str, Callable[..., sax.SDict]]:
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
        if str(sig.return_annotation) == "sax.SDict":
            models[name] = func
    return models
