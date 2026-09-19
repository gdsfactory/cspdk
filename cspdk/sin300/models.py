"""SAX models for Sparameter circuit simulations."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
import sax
import sax.models as sm
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
    """Returns the S-matrix of a straight waveguide.

    Args:
        wl: Wavelength of the simulation.
        length: Length of the waveguide.
        loss: Loss of the waveguide.
        cross_section: Cross section of the waveguide.
    """
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
    """Returns the S-matrix of a wire corner."""
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
    """Returns the S-matrix of a bend with a spline curve.

    NOTE: it is assumed that `bend_s` exposes it's length in its info dictionary!

    Args:
        wl: Wavelength of the simulation.
        length: Length of the bend.
        loss: Loss of the bend.
        cross_section: Cross section of the bend.

    """
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
    """Returns the S-matrix of a bend with an Euler curve.

     NOTE: it is assumed that `bend_euler` exposes it's length in its info dictionary!

    Args:
        wl: Wavelength of the simulation.
        length: Length of the bend.
        loss: Loss of the bend.
        cross_section: Cross section of the bend.
    """
    return straight(
        wl=wl,
        length=length,
        loss=loss,
        cross_section=cross_section,
    )


bend_euler_nc = partial(bend_euler, cross_section="xs_nc")
bend_euler_no = partial(bend_euler, cross_section="xs_no")


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
    """Returns the S-matrix of a taper.

    # NOTE: it is assumed that `taper` exposes it's length in its info dictionary!
    # TODO: take width1 and width2 into account.

    Args:
        wl: Wavelength of the simulation.
        length: Length of the taper.
        loss: Loss of the taper.
        cross_section: Cross section of the taper.
    """
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
    """Returns the S-matrix of a 1x2 MMI.

    Args:
        wl: Wavelength of the simulation.
        loss_dB: Loss of the MMI.
        cross_section: Cross section of the MMI.
    """
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
    """Returns the S-matrix of a 2x2 MMI.

    Args:
        wl: Wavelength of the simulation.
        loss_dB: Loss of the MMI.
        cross_section: Cross section of the MMI.
    """
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
    """Returns the S-matrix of a straight coupler."""
    # we should not need this model...
    raise NotImplementedError("No model for 'coupler_straight'")


def coupler_symmetric() -> sax.SDict:
    """Returns the S-matrix of a symmetric coupler."""
    # we should not need this model...
    raise NotImplementedError("No model for 'coupler_symmetric'")


coupler_nc = partial(sm.mmi2x2, wl0=1.55, fwhm=0.2)
coupler_no = partial(sm.mmi2x2, wl0=1.31, fwhm=0.2)


def coupler(
    wl: Float = 1.55,
    loss_dB: Float = 0.3,
    cross_section="xs_nc",
) -> sax.SDict:
    """Returns the S-matrix of a coupler.

    # TODO: take more coupler arguments into account

    Args:
        wl: Wavelength of the simulation.
        loss_dB: Loss of the coupler.
        cross_section: Cross section of the coupler.
    """
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
    """Returns the S-matrix of a rectangular grating coupler."""
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
    """Returns the S-matrix of an elliptical grating coupler."""
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
    """Returns the S-matrix of a heater."""
    raise NotImplementedError("No model for 'heater'")


crossing_no = sm.crossing_ideal


################
# Models Dict
################


def get_models() -> dict[str, Callable[..., sax.SDict]]:
    """Returns a dictionary of all models in this module."""
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
