"""SAX models for Sparameter circuit simulations."""

from __future__ import annotations

import inspect
from collections.abc import Callable

import jax.numpy as jnp
import sax
import sax.models as sm
from numpy.typing import NDArray

sax.set_port_naming_strategy("optical")

nm = 1e-3

FloatArray = NDArray[jnp.floating]
Float = float | FloatArray

################
# Straights
################


def straight_strip(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss_dB_cm: float = 3.0,
) -> sax.SDict:
    """Straight strip waveguide model."""
    return sm.straight(
        wl=wl,
        length=length,
        loss_dB_cm=loss_dB_cm,
        wl0=1.55,
        neff=2.38,
        ng=4.30,
    )


def straight_rib(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss_dB_cm: float = 3.0,
) -> sax.SDict:
    """Straight rib waveguide model."""
    return sm.straight(
        wl=wl,
        length=length,
        loss_dB_cm=loss_dB_cm,
        wl0=1.55,
        neff=2.38,
        ng=4.30,
    )


def straight(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss_dB_cm: float = 3.0,
    cross_section: str = "strip",
) -> sax.SDict:
    """Straight waveguide model."""
    wl = jnp.asarray(wl)  # type: ignore
    fs = {
        "strip": straight_strip,
        "rib": straight_rib,
    }
    f = fs[cross_section]
    return f(
        wl=wl,  # type: ignore
        length=length,
        loss_dB_cm=loss_dB_cm,
    )


################
# Bends
################


def wire_corner(*, wl: Float = 1.55) -> sax.SDict:
    """Wire corner model."""
    wl = jnp.asarray(wl)  # type: ignore
    zero = jnp.zeros_like(wl)
    return {"e1": zero, "e2": zero}  # type: ignore


def bend_s(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss_dB_cm=3.0,
    cross_section="strip",
) -> sax.SDict:
    """Bend S model."""
    # NOTE: it is assumed that `bend_s` exposes it's length in its info dictionary!
    return straight(
        wl=wl,
        length=length,
        loss_dB_cm=loss_dB_cm,
        cross_section=cross_section,
    )


def bend_euler(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss_dB_cm: float = 3,
    cross_section="strip",
) -> sax.SDict:
    """Euler bend model."""
    # NOTE: it is assumed that `bend_euler` exposes it's length in its info dictionary!
    return straight(
        wl=wl,
        length=length,
        loss_dB_cm=loss_dB_cm,
        cross_section=cross_section,
    )


def bend_euler_strip(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss_dB_cm: float = 3,
) -> sax.SDict:
    """Euler bend strip model."""
    return bend_euler(
        wl=wl,
        length=length,
        loss_dB_cm=loss_dB_cm,
        cross_section="strip",
    )


def bend_euler_rib(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss_dB_cm: float = 3,
) -> sax.SDict:
    """Euler bend rib model."""
    return bend_euler(
        wl=wl,
        length=length,
        loss_dB_cm=loss_dB_cm,
        cross_section="rib",
    )


################
# Transitions
################


def taper(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss_dB_cm: float = 0.0,
    cross_section="strip",
) -> sax.SDict:
    """Taper model."""
    # NOTE: it is assumed that `taper` exposes it's length in its info dictionary!
    # TODO: take width1 and width2 into account.
    return straight(
        wl=wl,
        length=length,
        loss_dB_cm=loss_dB_cm,
        cross_section=cross_section,
    )


def taper_rib(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss_dB_cm: float = 0.0,
) -> sax.SDict:
    """Taper rib model."""
    return taper(
        wl=wl,
        length=length,
        loss_dB_cm=loss_dB_cm,
        cross_section="rib",
    )


def taper_strip_to_ridge(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss_dB_cm: float = 0.0,
    cross_section="strip",
) -> sax.SDict:
    """Taper strip to ridge model."""
    # NOTE: it is assumed that `taper_strip_to_ridge` exposes it's length in its info dictionary!
    # TODO: take w_slab1 and w_slab2 into account.
    return straight(
        wl=wl,
        length=length,
        loss_dB_cm=loss_dB_cm,
        cross_section=cross_section,
    )


def trans_rib10(
    *,
    wl: Float = 1.55,
    loss_dB_cm: float = 0.0,
    cross_section="strip",
) -> sax.SDict:
    """Taper strip to ridge 10um model."""
    return taper_strip_to_ridge(
        wl=wl,
        length=10.0,
        loss_dB_cm=loss_dB_cm,
        cross_section=cross_section,
    )


def trans_rib20(
    *,
    wl: Float = 1.55,
    loss_dB_cm: float = 0.0,
    cross_section="strip",
) -> sax.SDict:
    """Taper strip to ridge 20um model."""
    return taper_strip_to_ridge(
        wl=wl,
        length=20.0,
        loss_dB_cm=loss_dB_cm,
        cross_section=cross_section,
    )


def trans_rib50(
    *,
    wl: Float = 1.55,
    loss_dB_cm: float = 0.0,
    cross_section="strip",
) -> sax.SDict:
    """Taper strip to ridge 50um model."""
    return taper_strip_to_ridge(
        wl=wl,
        length=50.0,
        loss_dB_cm=loss_dB_cm,
        cross_section=cross_section,
    )


################
# MMIs
################


def mmi1x2_strip(
    *,
    wl: Float = 1.55,
    wl0: float = 1.55,
    loss_dB: Float = 0.3,
    fwhm: Float = 0.2,
) -> sax.SDict:
    """MMI 1x2 strip model."""
    return sm.mmi1x2(
        wl=wl,
        wl0=wl0,
        fwhm=fwhm,
        loss_dB=loss_dB,
    )


def mmi1x2_rib(
    *,
    wl: Float = 1.55,
    wl0: float = 1.55,
    loss_dB: Float = 0.3,
    fwhm: Float = 0.2,
) -> sax.SDict:
    """MMI 1x2 rib model."""
    return sm.mmi1x2(
        wl=wl,
        wl0=wl0,
        fwhm=fwhm,
        loss_dB=loss_dB,
    )


def mmi1x2(
    wl: Float = 1.55,
    loss_dB: Float = 0.3,
    cross_section="strip",
) -> sax.SDict:
    """MMI 1x2 model."""
    wl = jnp.asarray(wl)  # type: ignore
    fs = {
        "strip": mmi1x2_strip,
        "rib": mmi1x2_rib,
    }
    f = fs[cross_section]
    return f(
        wl=wl,
        loss_dB=loss_dB,
    )


def mmi2x2_strip(
    *,
    wl: Float = 1.55,
    wl0: float = 1.55,
    loss_dB: Float = 0.3,
    fwhm: Float = 0.2,
) -> sax.SDict:
    """MMI 2x2 strip model."""
    return sm.mmi2x2(
        wl=wl,
        wl0=wl0,
        fwhm=fwhm,
        loss_dB=loss_dB,
    )


def mmi2x2_rib(
    *,
    wl: Float = 1.55,
    wl0: float = 1.55,
    loss_dB: Float = 0.3,
    fwhm: Float = 0.2,
) -> sax.SDict:
    """MMI 2x2 rib model."""
    return sm.mmi2x2(
        wl=wl,
        wl0=wl0,
        fwhm=fwhm,
        loss_dB=loss_dB,
    )


def mmi2x2(
    wl: Float = 1.55,
    loss_dB: Float = 0.3,
    cross_section="strip",
) -> sax.SDict:
    """MMI 2x2 model."""
    wl = jnp.asarray(wl)  # type: ignore
    fs = {
        "strip": mmi2x2_strip,
        "rib": mmi2x2_rib,
    }
    f = fs[cross_section]
    return f(
        wl=wl,
        loss_dB=loss_dB,
    )


##############################
# Evanescent couplers
##############################


def coupler_strip(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    coupling0: sax.FloatArrayLike = 0.2,
    dk1: sax.FloatArrayLike = 1.2435,
    dk2: sax.FloatArrayLike = 5.3022,
    dn: sax.FloatArrayLike = 0.02,
    dn1: sax.FloatArrayLike = 0.1169,
    dn2: sax.FloatArrayLike = 0.4821,
) -> sax.SDict:
    """Evanescent coupler strip model."""
    return sm.coupler(
        wl=wl,
        wl0=1.55,
        length=length,
        coupling0=coupling0,
        dk1=dk1,
        dk2=dk2,
        dn=dn,
        dn1=dn1,
        dn2=dn2,
    )


def coupler_rib(
    *,
    wl: Float = 1.55,
    wl0: float = 1.55,
    length: float = 10.0,
    coupling0: sax.FloatArrayLike = 0.2,
    dk1: sax.FloatArrayLike = 1.2435,
    dk2: sax.FloatArrayLike = 5.3022,
    dn: sax.FloatArrayLike = 0.02,
    dn1: sax.FloatArrayLike = 0.1169,
    dn2: sax.FloatArrayLike = 0.4821,
) -> sax.SDict:
    """Evanescent coupler rib model."""
    return sm.coupler(
        wl=wl,
        wl0=wl0,
        length=length,
        coupling0=coupling0,
        dk1=dk1,
        dk2=dk2,
        dn=dn,
        dn1=dn1,
        dn2=dn2,
    )


def coupler_ring(
    *,
    wl: Float = 1.55,
    wl0: float = 1.55,
    length: float = 10.0,
    coupling0: sax.FloatArrayLike = 0.2,
    dk1: sax.FloatArrayLike = 1.2435,
    dk2: sax.FloatArrayLike = 5.3022,
    dn: sax.FloatArrayLike = 0.02,
    dn1: sax.FloatArrayLike = 0.1169,
    dn2: sax.FloatArrayLike = 0.4821,
) -> sax.SDict:
    """Ring coupler model."""
    return sm.coupler(
        wl=wl,
        wl0=wl0,
        length=length,
        coupling0=coupling0,
        dk1=dk1,
        dk2=dk2,
        dn=dn,
        dn1=dn1,
        dn2=dn2,
    )


def coupler(
    wl: Float = 1.55,
    length: float = 10.0,
    cross_section="strip",
) -> sax.SDict:
    """Evanescent coupler model."""
    # TODO: take more coupler arguments into account
    wl = jnp.asarray(wl)  # type: ignore
    fs = {
        "strip": coupler_strip,
        "rib": coupler_rib,
    }
    f = fs[cross_section]
    return f(
        wl=wl,
        length=length,
    )


##############################
# grating couplers Rectangular
##############################


def grating_coupler_rectangular_strip(
    *,
    wl: Float = 1.55,
) -> sax.SDict:
    """Grating coupler rectangular strip model."""
    return sm.grating_coupler(
        wl=wl,
        loss=6,
        bandwidth=35 * nm,
    )


def grating_coupler_rectangular_rib(
    *,
    wl: Float = 1.55,
) -> sax.SDict:
    """Grating coupler rectangular rib model."""
    return sm.grating_coupler(
        wl=wl,
        loss=6,
        bandwidth=35 * nm,
    )


def grating_coupler_rectangular(
    wl: Float = 1.55,
    cross_section="strip",
) -> sax.SDict:
    """Grating coupler rectangular model."""
    # TODO: take more grating_coupler_rectangular arguments into account
    wl = jnp.asarray(wl)  # type: ignore
    fs = {
        "strip": grating_coupler_rectangular_strip,
        "rib": grating_coupler_rectangular_rib,
    }
    f = fs[cross_section]
    return f(wl=wl)  # type: ignore


##############################
# grating couplers Elliptical
##############################


def grating_coupler_elliptical(
    *,
    wl: Float = 1.55,
) -> sax.SDict:
    """Grating coupler elliptical model."""
    return sm.grating_coupler(
        wl=wl,
        loss=6,
        bandwidth=35 * nm,
    )


################
# Imported
################


def heater() -> sax.SDict:
    """Heater model."""
    raise NotImplementedError("No model for 'heater'")


def straight_heater_metal(
    wl: float = 1.55,
    neff: float = 2.34,
    voltage: float = 0,
    vpi: float = 1.0,  # Voltage required for π-phase shift
    length: float = 10,
    loss: float = 0.0,
) -> sax.SDict:
    """Returns simple phase shifter model.

    Args:
        wl: wavelength.
        neff: effective index.
        voltage: applied voltage.
        vpi: voltage required for a π-phase shift.
        length: length.
        loss: loss.
    """
    # Calculate additional phase shift due to applied voltage.
    deltaphi = (voltage / vpi) * jnp.pi
    phase = 2 * jnp.pi * neff * length / wl + deltaphi
    amplitude = jnp.asarray(10 ** (-loss * length / 20), dtype=complex)
    transmission = amplitude * jnp.exp(1j * phase)
    return sax.reciprocal(
        {
            ("o1", "o2"): transmission,
            ("l_e1", "r_e1"): 0.0,
            ("l_e2", "r_e2"): 0.0,
            ("l_e3", "r_e3"): 0.0,
            ("l_e4", "r_e4"): 0.0,
        }
    )


def crossing_rib(
    *,
    wl: Float = 1.55,
) -> sax.SDict:
    """Crossing rib model."""
    return sm.crossing_ideal(wl=wl)


def crossing(
    *,
    wl: Float = 1.55,
) -> sax.SDict:
    """Crossing model."""
    return sm.crossing_ideal(wl=wl)


################
# Models Dict
################


def get_models() -> dict[str, Callable[..., sax.SDict]]:
    """Return a dictionary of all models in this module."""
    models = {}
    for name, func in list(globals().items()):
        # Skip get_models itself and private functions
        if name == "get_models" or name.startswith("_"):
            continue
        if not callable(func):
            continue
        try:
            sig = inspect.signature(func)
        except (ValueError, TypeError):
            continue
        # Check for sax.SDict return type (case-insensitive)
        return_anno = str(sig.return_annotation)
        if "sdict" in return_anno.lower():
            models[name] = func
    return models
