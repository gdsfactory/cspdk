from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import sax
from gdsfactory.pdk import get_cross_section_name
from gdsfactory.typings import CrossSectionSpec
from gplugins.sax.models import bend as __bend
from gplugins.sax.models import grating_coupler
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
    cross_section: CrossSectionSpec = "xs_sc",
) -> sax.SDict:
    if get_cross_section_name(cross_section).endswith("o"):
        return __straight(
            wl=wl,  # type: ignore
            length=length,  # type: ignore
            loss=loss,  # type: ignore
            wl0=1.31,
            neff=2.4,
            ng=4.2,
        )
    else:
        return __straight(
            wl=wl,  # type: ignore
            length=length,  # type: ignore
            loss=loss,  # type: ignore
            wl0=1.55,
            neff=2.4,
            ng=4.2,
        )


straight_sc = partial(_straight, cross_section="xs_sc")
straight_so = partial(_straight, cross_section="xs_so")
straight_rc = partial(_straight, cross_section="xs_rc")
straight_ro = partial(_straight, cross_section="xs_ro")
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


bend_sc = partial(_bend, loss=0.03)
bend_so = partial(_bend, loss=0.03)
bend_rc = partial(_bend, loss=0.03)
bend_ro = partial(_bend, loss=0.03)
bend_nc = partial(_bend, loss=0.03)
bend_no = partial(_bend, loss=0.03)

################
# Transitions
################

_taper = _straight
_taper_cross_section = _taper
trans_sc_rc10 = partial(_taper_cross_section, length=10.0)
trans_sc_rc20 = partial(_taper_cross_section, length=20.0)
trans_sc_rc50 = partial(_taper_cross_section, length=50.0)

################
# MMIs
################


def _mmi_amp(
    wl: Float = 1.55, wl0: Float = 1.55, fwhm: Float = 0.2, loss_dB: Float = 0.3
):
    max_power = 10 ** (-abs(loss_dB) / 10)
    f = 1 / wl
    f0 = 1 / wl0
    f1 = 1 / (wl0 + fwhm / 2)
    f2 = 1 / (wl0 - fwhm / 2)
    _fwhm = f2 - f1

    sigma = _fwhm / (2 * jnp.sqrt(2 * jnp.log(2)))
    power = jnp.exp(-((f - f0) ** 2) / (2 * sigma**2))
    power = max_power * power / power.max() / 2
    amp = jnp.sqrt(power)
    return amp


def _mmi1x2(
    wl: Float = 1.55, wl0: Float = 1.55, fwhm: Float = 0.2, loss_dB: Float = 0.3
) -> sax.SDict:
    thru = _mmi_amp(wl=wl, wl0=wl0, fwhm=fwhm, loss_dB=loss_dB)
    return sax.reciprocal(
        {
            ("o1", "o2"): thru,
            ("o1", "o3"): thru,
        }
    )


def _mmi2x2(
    wl: Float = 1.55,
    wl0: Float = 1.55,
    fwhm: Float = 0.2,
    loss_dB: Float = 0.3,
    shift: Float = 0.005,
) -> sax.SDict:
    thru = _mmi_amp(wl=wl, wl0=wl0, fwhm=fwhm, loss_dB=loss_dB)
    cross = 1j * _mmi_amp(wl=wl, wl0=wl0 + shift, fwhm=fwhm, loss_dB=loss_dB)
    return sax.reciprocal(
        {
            ("o1", "o3"): thru,
            ("o1", "o4"): cross,
            ("o2", "o3"): cross,
            ("o2", "o4"): thru,
        }
    )


_mmi1x2_o = partial(_mmi1x2, wl0=1.31)
_mmi1x2_c = partial(_mmi1x2, wl0=1.55)
_mmi2x2_o = partial(_mmi2x2, wl0=1.31)
_mmi2x2_c = partial(_mmi2x2, wl0=1.55)

mmi1x2_rc = _mmi1x2_c
mmi1x2_sc = _mmi1x2_c
mmi1x2_nc = _mmi1x2_c
mmi1x2_ro = _mmi1x2_o
mmi1x2_so = _mmi1x2_o
mmi1x2_no = _mmi1x2_o

mmi2x2_rc = _mmi2x2_c
mmi2x2_sc = _mmi2x2_c
mmi2x2_nc = _mmi2x2_c
mmi2x2_ro = _mmi2x2_o
mmi2x2_so = _mmi2x2_o
mmi2x2_no = _mmi2x2_o

##############################
# Evanescent couplers
##############################

_coupler = _mmi2x2
_coupler_o = partial(_coupler, wl0=1.31)
_coupler_c = partial(_coupler, wl0=1.55)
coupler_sc = _coupler_c
coupler_rc = _coupler_c
coupler_nc = _coupler_c
coupler_so = _coupler_o
coupler_ro = _coupler_o
coupler_no = _coupler_o

##############################
# grating couplers Rectangular
##############################


def _gc_rectangular(
    *,
    wl: Float = 1.55,
    reflection: Float = 0.0,
    reflection_fiber: Float = 0.0,
    loss=0.0,
    wavelength=1.55,
    bandwidth: Float = 40 * nm,
) -> sax.SDict:
    return grating_coupler(
        wl=wl,  # type: ignore
        wl0=wavelength,
        reflection=reflection,  # type: ignore
        reflection_fiber=reflection_fiber,  # type: ignore
        loss=loss,
        bandwidth=bandwidth,  # type: ignore
    )


_gcro = partial(_gc_rectangular, loss=6, bandwidth=35 * nm, wavelength=1.31)
_gcrc = partial(_gc_rectangular, loss=6, bandwidth=35 * nm, wavelength=1.55)
gc_rectangular_sc = _gcrc
gc_rectangular_so = _gcro
gc_rectangular_rc = _gcrc
gc_rectangular_ro = _gcro
gc_rectangular_nc = _gcrc
gc_rectangular_no = _gcro

##############################
# grating couplers Elliptical
##############################

_gc_elliptical = _gc_rectangular
_gceo = partial(_gc_elliptical, loss=6, bandwidth=35 * nm, wavelength=1.31)
_gcec = partial(_gc_elliptical, loss=6, bandwidth=35 * nm, wavelength=1.55)
gc_elliptical_sc = _gcec
gc_elliptical_so = _gceo
gc_elliptical_rc = _gcec
gc_elliptical_ro = _gceo
gc_elliptical_nc = _gcec
gc_elliptical_no = _gceo


################
# Crossings
################


@jax.jit
def _crossing(wl: Float = 1.5) -> sax.SDict:
    one = jnp.ones_like(jnp.asarray(wl))
    return sax.reciprocal(
        {
            ("o1", "o3"): one,
            ("o2", "o4"): one,
        }
    )


crossing_so = _crossing
crossing_rc = _crossing
crossing_sc = _crossing


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
