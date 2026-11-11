"""SAX models for Sparameter circuit simulations."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
import sax
import sax.models as sm
from numpy.typing import NDArray

from cspdk._models import _bend_s_length, _euler_length, _optical_model, _sdict_models
from cspdk.sin300.tech import Tech

nm = 1e-3

FloatArray = NDArray[jnp.floating]
Float = float | FloatArray

# Bend radius the cell uses when ``radius`` is not given: ``bend_euler`` passes
# ``radius=None`` through to the cross-section, whose radius is this.
_BEND_RADIUS = {"xs_nc": Tech.radius_nc, "xs_no": Tech.radius_no}


def _bend_radius(cross_section: str) -> float:
    """Radius of ``cross_section``, or a clear error if this flavour has no bend for it."""
    try:
        return _BEND_RADIUS[cross_section]
    except KeyError:
        msg = (
            f"No SiN300 bend radius for cross_section={cross_section!r}; "
            f"expected one of {sorted(_BEND_RADIUS)}."
        )
        raise ValueError(msg) from None


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
    wl=1.31,
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
    size: tuple[float, float] = (15.0, 1.8),
    length: float | None = None,
    loss: float = 0.03,
    cross_section="xs_nc",
) -> sax.SDict:
    """Returns the S-matrix of a bend with a spline curve.

    Layout netlists pass ``size`` but not the path length, so the length defaults to
    that of the drawn Bezier S-bend (``gf.components.bend_s``'s 99-point polyline,
    reproduced by ``_bend_s_length``). Pass ``length`` to override.

    Args:
        wl: Wavelength of the simulation.
        size: S-bend (dx, dy) in um.
        length: Path length in um (overrides ``size``).
        loss: Propagation loss in dB/cm.
        cross_section: Cross section of the bend.
    """
    path_length = length if length is not None else _bend_s_length(size[0], size[1])
    return straight(
        wl=wl,
        length=path_length,
        loss=loss,
        cross_section=cross_section,
    )


def bend_euler(
    *,
    wl: Float = 1.55,
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.5,
    length: float | None = None,
    loss: float = 0.03,
    cross_section="xs_nc",
) -> sax.SDict:
    """Returns the S-matrix of a bend with an Euler curve.

    Layout netlists pass ``radius``, ``angle`` and ``p`` but not the path length, so the
    length defaults to that of the drawn Euler bend
    (``gf.path.euler(radius, angle, p, use_eff=True)``, reproduced by ``_euler_length``).
    Pass ``length`` to override.

    Args:
        wl: Wavelength of the simulation.
        radius: Effective bend radius in um (``None`` is the cross-section radius).
        angle: Bend angle in degrees.
        p: Fraction of the bend that is an euler curve.
        length: Path length in um (overrides ``radius``, ``angle`` and ``p``).
        loss: Propagation loss in dB/cm.
        cross_section: Cross section of the bend.
    """
    r = _bend_radius(cross_section) if radius is None else radius
    path_length = length if length is not None else _euler_length(r, angle, p)
    return straight(
        wl=wl,
        length=path_length,
        loss=loss,
        cross_section=cross_section,
    )


bend_euler_nc = partial(bend_euler, cross_section="xs_nc")
bend_euler_no = partial(bend_euler, cross_section="xs_no", wl=1.31)


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

    # ``length`` is a cell setting (``taper(length=...)``), so a layout netlist carries it
    # and this default is only the fallback for a circuit that sets nothing.
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
taper_no = partial(taper, cross_section="xs_no", length=10.0, wl=1.31)


################
# MMIs
################

mmi1x2_nc = partial(_mmi1x2_model, wl0=1.55, fwhm=0.2)
mmi1x2_no = partial(_mmi1x2_model, wl=1.31, wl0=1.31, fwhm=0.2)


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
mmi2x2_no = partial(_mmi2x2_model, wl=1.31, wl0=1.31, fwhm=0.2)


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
coupler_no = partial(_mmi2x2_model, wl=1.31, wl0=1.31, fwhm=0.2)


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


crossing_no = partial(_optical_model(sm.crossing_ideal, 2, 2), wl=1.31)


################
# Models Dict
################


def get_models() -> dict[str, Callable[..., sax.SDict]]:
    """Returns a dictionary of all models in this module."""
    return _sdict_models(
        globals(), exclude={"heater", "coupler_straight", "coupler_symmetric"}
    )


if __name__ == "__main__":
    print(list(get_models()))
    for name, model in get_models().items():
        try:
            print(name, model())
        except NotImplementedError:
            continue
