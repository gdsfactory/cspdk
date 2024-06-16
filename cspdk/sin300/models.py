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

straight_nc = partial(
    sm.straight,
    length=10.0,
    loss=0.0,
    wl0=1.55,
    neff=1.60,
    ng=1.95,
)

straight_no = partial(
    sm.straight,
    length=10.0,
    loss=0.0,
    wl0=1.31,
    neff=1.63,
    ng=2.00,
)


def straight(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss: float = 0.0,
    cross_section: str = "xs_nc",
) -> sax.SDict:
    wl = jnp.asarray(wl)  # type: ignore
    fs = {
        "xs_nc": straight_nc,
        "xs_no": straight_no,
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
    cross_section="xs_nc",
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
    cross_section="xs_nc",
) -> sax.SDict:
    # NOTE: it is assumed that `bend_euler` exposes it's length in its info dictionary!
    return straight(
        wl=wl,
        length=length,
        loss=loss,
        cross_section=cross_section,
    )


bend_nc = partial(bend_euler, cross_section="xs_nc")
bend_no = partial(bend_euler, cross_section="xs_no")


################
# Transitions
################


def taper(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss: float = 0.0,
    cross_section="xs_nc",
) -> sax.SDict:
    # NOTE: it is assumed that `taper` exposes it's length in its info dictionary!
    # TODO: take width1 and width2 into account.
    return straight(
        wl=wl,
        length=length,
        loss=loss,
        cross_section=cross_section,
    )


taper_nc = partial(taper, cross_section="xs_no", length=10.0)
taper_no = partial(taper, cross_section="xs_no", length=10.0)


################
# MMIs
################

mmi1x2_nc = partial(sm.mmi1x2, wl0=1.55, fwhm=0.2)
mmi1x2_no = partial(sm.mmi1x2, wl0=1.31, fwhm=0.2)


def mmi1x2(
    wl: Float = 1.55,
    loss_dB: Float = 0.3,
    cross_section="xs_nc",
) -> sax.SDict:
    wl = jnp.asarray(wl)  # type: ignore
    fs = {
        "xs_nc": mmi1x2_nc,
        "xs_no": mmi1x2_no,
    }
    f = fs[cross_section]
    return f(
        wl=wl,
        loss_dB=loss_dB,
    )


mmi2x2_nc = partial(sm.mmi2x2, wl0=1.55, fwhm=0.2)
mmi2x2_no = partial(sm.mmi2x2, wl0=1.31, fwhm=0.2)


def mmi2x2(
    wl: Float = 1.55,
    loss_dB: Float = 0.3,
    cross_section="xs_nc",
) -> sax.SDict:
    wl = jnp.asarray(wl)  # type: ignore
    fs = {
        "xs_nc": mmi2x2_nc,
        "xs_no": mmi2x2_no,
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


coupler_nc = partial(sm.mmi2x2, wl0=1.55, fwhm=0.2)
coupler_no = partial(sm.mmi2x2, wl0=1.31, fwhm=0.2)


def coupler(
    wl: Float = 1.55,
    loss_dB: Float = 0.3,
    cross_section="xs_nc",
) -> sax.SDict:
    # TODO: take more coupler arguments into account
    wl = jnp.asarray(wl)  # type: ignore
    fs = {
        "xs_nc": coupler_nc,
        "xs_no": coupler_no,
    }
    f = fs[cross_section]
    return f(
        wl=wl,
        loss_dB=loss_dB,
    )


##############################
# grating couplers Rectangular
##############################

grating_coupler_rectangular_no = partial(
    sm.grating_coupler, loss=6, bandwidth=35 * nm, wl=1.31
)

grating_coupler_rectangular_nc = partial(
    sm.grating_coupler, loss=6, bandwidth=35 * nm, wl=1.55
)


def grating_coupler_rectangular(
    wl: Float = 1.55,
    cross_section="xs_nc",
) -> sax.SDict:
    # TODO: take more grating_coupler_rectangular arguments into account
    wl = jnp.asarray(wl)  # type: ignore
    fs = {
        "xs_nc": grating_coupler_rectangular_nc,
        "xs_no": grating_coupler_rectangular_no,
    }
    f = fs[cross_section]
    return f(wl=wl)  # type: ignore


##############################
# grating couplers Elliptical
##############################

grating_coupler_elliptical_no = partial(
    sm.grating_coupler, loss=6, bandwidth=35 * nm, wl=1.31
)

grating_coupler_elliptical_nc = partial(
    sm.grating_coupler, loss=6, bandwidth=35 * nm, wl=1.55
)


def grating_coupler_elliptical(
    wl: Float = 1.55,
    bandwidth: float = 35e-3,
    cross_section="xs_nc",
) -> sax.SDict:
    # TODO: take more grating_coupler_elliptical arguments into account
    wl = jnp.asarray(wl)  # type: ignore
    fs = {
        "xs_nc": grating_coupler_elliptical_nc,
        "xs_no": grating_coupler_elliptical_no,
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


crossing_no = sm.crossing


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
