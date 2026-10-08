"""SAX models for Sparameter circuit simulations."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import partial, wraps

import jax.numpy as jnp
import sax
import sax.models as sm
from numpy.typing import NDArray

nm = 1e-3

FloatArray = NDArray[jnp.floating]
Float = float | FloatArray


def _optical_model(model, inputs: int, outputs: int):
    """Normalize SAX ports without changing its process-wide naming strategy.

    Translate input/output and zero-based optical keys, preserving one-based
    optical keys. Inspect the returned keys because jitted models may retain
    a naming convention cached before the current strategy was selected.
    """
    port_map = {f"in{i}": f"o{i + 1}" for i in range(inputs)}
    port_map.update({f"out{i}": f"o{inputs + outputs - i}" for i in range(outputs)})

    @wraps(model)
    def optical(*args, **kwargs) -> sax.SDict:
        result = model(*args, **kwargs)
        mapping = port_map
        if any("o0" in pair for pair in result):
            mapping = {
                **port_map,
                **{f"o{i}": f"o{i + 1}" for i in range(inputs + outputs)},
            }
        return {
            (mapping.get(p, p), mapping.get(q, q)): value
            for (p, q), value in result.items()
        }

    return optical


_straight_model = _optical_model(sm.straight, 1, 1)
_mmi1x2_model = _optical_model(sm.mmi1x2, 1, 2)
_mmi2x2_model = _optical_model(sm.mmi2x2, 2, 2)
_grating_model = _optical_model(sm.grating_coupler, 1, 1)


def _straight(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss: float = 0.0,
    wl0: float = 1.55,
    neff: float = 1.60,
    ng: float = 1.95,
) -> sax.SDict:
    """Adapt the legacy SiN loss argument (dB/cm) to the SAX API."""
    return _straight_model(
        wl=wl, length=length, loss_dB_cm=loss, wl0=wl0, neff=neff, ng=ng
    )


################
# Straights
################

straight_nc = partial(
    _straight,
    length=10.0,
    loss=0.0,
    wl0=1.55,
    neff=1.60,
    ng=1.95,
)

straight_no = partial(
    _straight,
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
        loss: Propagation loss in dB/cm.
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
    return sax.reciprocal({("e1", "e2"): zero})


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
        loss: Propagation loss in dB/cm.
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
        loss: Propagation loss in dB/cm.
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
        loss: Propagation loss in dB/cm.
        cross_section: Cross section of the taper.
    """
    return straight(
        wl=wl,
        length=length,
        loss=loss,
        cross_section=cross_section,
    )


taper_nc = partial(taper, cross_section="xs_nc", length=10.0)
taper_no = partial(taper, cross_section="xs_no", length=10.0)


################
# MMIs
################

mmi1x2_nc = partial(_mmi1x2_model, wl0=1.55, fwhm=0.2)
mmi1x2_no = partial(_mmi1x2_model, wl0=1.31, fwhm=0.2)


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


mmi2x2_nc = partial(_mmi2x2_model, wl0=1.55, fwhm=0.2)
mmi2x2_no = partial(_mmi2x2_model, wl0=1.31, fwhm=0.2)


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


coupler_nc = partial(_mmi2x2_model, wl0=1.55, fwhm=0.2)
coupler_no = partial(_mmi2x2_model, wl0=1.31, fwhm=0.2)


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
    _grating_model, loss=6, bandwidth=35 * nm, wl=1.31, wl0=1.31
)

grating_coupler_rectangular_nc = partial(
    _grating_model, loss=6, bandwidth=35 * nm, wl=1.55, wl0=1.55
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
    _grating_model, loss=6, bandwidth=35 * nm, wl=1.31, wl0=1.31
)

grating_coupler_elliptical_nc = partial(
    _grating_model, loss=6, bandwidth=35 * nm, wl=1.55, wl0=1.55
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


crossing_no = _optical_model(sm.crossing_ideal, 2, 2)


################
# Models Dict
################


def get_models() -> dict[str, Callable[..., sax.SDict]]:
    """Returns a dictionary of all models in this module."""
    models = {}
    for name, func in list(globals().items()):
        if name.startswith("_") or name in {
            "heater",
            "coupler_straight",
            "coupler_symmetric",
        }:
            continue
        if not callable(func):
            continue
        _func = func
        while isinstance(_func, partial):
            _func = _func.func
        try:
            sig = inspect.signature(_func)
        except (ValueError, TypeError):
            continue
        if (
            sig.return_annotation == sax.SDict
            or str(sig.return_annotation).lower().split(".")[-1] == "sdict"
        ):
            models[name] = func
    return models


if __name__ == "__main__":
    print(list(get_models()))
    for name, model in get_models().items():
        try:
            print(name, model())
        except NotImplementedError:
            continue
