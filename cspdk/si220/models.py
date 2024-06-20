from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import partial

import gplugins.sax.models as sm
import jax.numpy as jnp
import sax
from numpy.typing import NDArray

nm = 1e-3

FloatArray = NDArray[jnp.floating]
Float = float | FloatArray

################
# Straights
################

straight_sc = partial(
    sm.straight,
    length=10.0,
    loss=0.0,
    wl0=1.55,
    neff=2.38,
    ng=4.30,
)

straight_so = partial(
    sm.straight,
    length=10.0,
    loss=0.0,
    wl0=1.31,
    neff=2.52,
    ng=4.33,
)


straight_rc = partial(
    sm.straight,
    length=10.0,
    loss=0.0,
    wl0=1.55,
    neff=2.38,
    ng=4.30,
)

straight_ro = partial(
    sm.straight,
    length=10.0,
    loss=0.0,
    wl0=1.31,
    neff=2.72,
    ng=3.98,
)


def straight(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss: float = 0.0,
    cross_section: str = "xs_sc",
) -> sax.SDict:
    wl = jnp.asarray(wl)  # type: ignore
    fs = {
        "xs_sc": straight_sc,
        "xs_so": straight_so,
        "xs_rc": straight_rc,
        "xs_ro": straight_ro,
    }
    f = fs[cross_section]
    return f(
        wl=wl,  # type: ignore
        length=length,
        loss=loss,
    )


################
# Bends
################


def wire_corner(*, wl: Float = 1.55) -> sax.SDict:
    wl = jnp.asarray(wl)  # type: ignore
    zero = jnp.zeros_like(wl)
    return {"e1": zero, "e2": zero}  # type: ignore


def bend_s(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss: float = 0.03,
    cross_section="xs_sc",
) -> sax.SDict:
    # NOTE: it is assumed that `bend_s` exposes it's length in its info dictionary!
    return straight(
        wl=wl,
        length=length,
        loss=loss,
        cross_section=cross_section,
    )


def bend_euler(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss: float = 0.03,
    cross_section="xs_sc",
) -> sax.SDict:
    # NOTE: it is assumed that `bend_euler` exposes it's length in its info dictionary!
    return straight(
        wl=wl,
        length=length,
        loss=loss,
        cross_section=cross_section,
    )


bend_sc = partial(bend_euler, cross_section="xs_sc")
bend_so = partial(bend_euler, cross_section="xs_so")
bend_rc = partial(bend_euler, cross_section="xs_rc")
bend_ro = partial(bend_euler, cross_section="xs_ro")


################
# Transitions
################


def taper(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss: float = 0.0,
    cross_section="xs_sc",
) -> sax.SDict:
    # NOTE: it is assumed that `taper` exposes it's length in its info dictionary!
    # TODO: take width1 and width2 into account.
    return straight(
        wl=wl,
        length=length,
        loss=loss,
        cross_section=cross_section,
    )


taper_sc = partial(taper, cross_section="xs_so", length=10.0)
taper_so = partial(taper, cross_section="xs_so", length=10.0)
taper_rc = partial(taper, cross_section="xs_rc", length=10.0)
taper_ro = partial(taper, cross_section="xs_ro", length=10.0)


def taper_strip_to_ridge(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss: float = 0.0,
    cross_section="xs_sc",
) -> sax.SDict:
    # NOTE: it is assumed that `taper_strip_to_ridge` exposes it's length in its info dictionary!
    # TODO: take w_slab1 and w_slab2 into account.
    return straight(
        wl=wl,
        length=length,
        loss=loss,
        cross_section=cross_section,
    )


trans_sc_rc10 = partial(taper_strip_to_ridge, length=10.0)
trans_sc_rc20 = partial(taper_strip_to_ridge, length=20.0)
trans_sc_rc50 = partial(taper_strip_to_ridge, length=50.0)

################
# MMIs
################

mmi1x2_sc = partial(sm.mmi1x2, wl0=1.55, fwhm=0.2)
mmi1x2_rc = mmi1x2_sc
mmi1x2_so = partial(sm.mmi1x2, wl0=1.31, fwhm=0.2)
mmi1x2_ro = mmi1x2_so


def mmi1x2(
    wl: Float = 1.55,
    loss_dB: Float = 0.3,
    cross_section="xs_sc",
) -> sax.SDict:
    wl = jnp.asarray(wl)  # type: ignore
    fs = {
        "xs_sc": mmi1x2_sc,
        "xs_so": mmi1x2_so,
        "xs_rc": mmi1x2_rc,
        "xs_ro": mmi1x2_ro,
    }
    f = fs[cross_section]
    return f(
        wl=wl,
        loss_dB=loss_dB,
    )


mmi2x2_sc = partial(sm.mmi2x2, wl0=1.55, fwhm=0.2)
mmi2x2_rc = mmi2x2_sc
mmi2x2_so = partial(sm.mmi2x2, wl0=1.31, fwhm=0.2)
mmi2x2_ro = mmi2x2_so


def mmi2x2(
    wl: Float = 1.55,
    loss_dB: Float = 0.3,
    cross_section="xs_sc",
) -> sax.SDict:
    wl = jnp.asarray(wl)  # type: ignore
    fs = {
        "xs_sc": mmi2x2_sc,
        "xs_so": mmi2x2_so,
        "xs_rc": mmi2x2_rc,
        "xs_ro": mmi2x2_ro,
    }
    f = fs[cross_section]
    return f(
        wl=wl,
        loss_dB=loss_dB,
    )


##############################
# Evanescent couplers
##############################


def coupler_straight() -> sax.SDict:
    # we should not need this model...
    raise NotImplementedError("No model for 'coupler_straight'")


def coupler_symmetric() -> sax.SDict:
    # we should not need this model...
    raise NotImplementedError("No model for 'coupler_symmetric'")


coupler_sc = partial(sm.mmi2x2, wl0=1.55, fwhm=0.2)
coupler_rc = coupler_sc
coupler_so = partial(sm.mmi2x2, wl0=1.31, fwhm=0.2)
coupler_ro = coupler_so


def coupler(
    wl: Float = 1.55,
    loss_dB: Float = 0.3,
    cross_section="xs_sc",
) -> sax.SDict:
    # TODO: take more coupler arguments into account
    wl = jnp.asarray(wl)  # type: ignore
    fs = {
        "xs_sc": coupler_sc,
        "xs_so": coupler_so,
        "xs_rc": coupler_rc,
        "xs_ro": coupler_ro,
    }
    f = fs[cross_section]
    return f(
        wl=wl,
        loss_dB=loss_dB,
    )


##############################
# grating couplers Rectangular
##############################

grating_coupler_rectangular_so = partial(
    sm.grating_coupler, loss=6, bandwidth=35 * nm, wl=1.31
)
grating_coupler_rectangular_ro = grating_coupler_rectangular_so

grating_coupler_rectangular_sc = partial(
    sm.grating_coupler, loss=6, bandwidth=35 * nm, wl=1.55
)
grating_coupler_rectangular_rc = grating_coupler_rectangular_sc


def grating_coupler_rectangular(
    wl: Float = 1.55,
    cross_section="xs_sc",
) -> sax.SDict:
    # TODO: take more grating_coupler_rectangular arguments into account
    wl = jnp.asarray(wl)  # type: ignore
    fs = {
        "xs_sc": grating_coupler_rectangular_sc,
        "xs_so": grating_coupler_rectangular_so,
        "xs_rc": grating_coupler_rectangular_rc,
        "xs_ro": grating_coupler_rectangular_ro,
    }
    f = fs[cross_section]
    return f(wl=wl)  # type: ignore


##############################
# grating couplers Elliptical
##############################

grating_coupler_elliptical_so = partial(
    sm.grating_coupler, loss=6, bandwidth=35 * nm, wl=1.31
)

grating_coupler_elliptical_sc = partial(
    sm.grating_coupler, loss=6, bandwidth=35 * nm, wl=1.55
)


def grating_coupler_elliptical(
    wl: Float = 1.55,
    bandwidth: float = 35e-3,
    cross_section="xs_sc",
) -> sax.SDict:
    # TODO: take more grating_coupler_elliptical arguments into account
    wl = jnp.asarray(wl)  # type: ignore
    fs = {
        "xs_sc": grating_coupler_elliptical_sc,
        "xs_so": grating_coupler_elliptical_so,
    }
    f = fs[cross_section]
    return f(
        wl=wl,  # type: ignore
        bandwidth=bandwidth,
    )


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


def heater() -> sax.SDict:
    raise NotImplementedError("No model for 'heater'")


crossing_so = sm.crossing
crossing_rc = sm.crossing
crossing_sc = sm.crossing


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
