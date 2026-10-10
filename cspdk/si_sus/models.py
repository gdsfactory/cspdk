"""SAX models for the suspended-Si 3.8um TE platform.

Waveguide index: femwell mode solve of the 1.5um x 450nm suspended Si core
with the slotted side cladding as a zeroth-order effective medium (see
cspdk/si_sus/samples/mode_solver.py for the set-up and approximations).

Loss: the MPW #7 quality target is < 5 dB/cm for the straight single-mode
waveguide at 3.8um; 5 dB/cm is used for straights, bends and tapers. The
foundry library has no measured data for any component ("No data at the
moment"), so bends carry no extra loss and the grating-coupler model is a
placeholder (6 dB peak loss, 150nm 3-dB bandwidth at 3.8um).
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import sax
import sax.models as sm
from numpy.typing import NDArray

from cspdk._models import _euler_length, _optical_model
from cspdk.si_sus.tech import TECH

FloatArray = NDArray[jnp.floating]
Float = float | FloatArray

WL0 = 3.8
# samples/mode_solver.py; the bare core in air gives neff 2.2826, ng 4.0141
NEFF = 2.3828
NG = 3.7328
LOSS_DB_CM = 5.0
GRATING_LOSS_DB = 6.0
GRATING_BANDWIDTH = 0.15


_straight_model = _optical_model(sm.straight, 1, 1)
_grating_model = _optical_model(sm.grating_coupler, 1, 1)


################
# Waveguides
################


def straight(
    *,
    wl: Float = WL0,
    length: float = 10.0,
    loss: float = LOSS_DB_CM,
    cross_section: str = "xs_sus",
) -> sax.SDict:
    """Returns the S-matrix of a straight suspended waveguide.

    Args:
        wl: wavelength in um.
        length: length in um.
        loss: propagation loss in dB/cm.
        cross_section: cross-section name (only xs_sus exists).
    """
    del cross_section
    return _straight_model(
        wl=jnp.asarray(wl), length=length, loss_dB_cm=loss, wl0=WL0, neff=NEFF, ng=NG
    )


def taper(
    *,
    wl: Float = WL0,
    length: float = 10.0,
    loss: float = LOSS_DB_CM,
    cross_section: str = "xs_sus",
) -> sax.SDict:
    """Returns the S-matrix of a taper (phase of the 1.5um core, no mode mismatch).

    Args:
        wl: wavelength in um.
        length: length in um.
        loss: propagation loss in dB/cm.
        cross_section: cross-section name (only xs_sus exists).
    """
    return straight(wl=wl, length=length, loss=loss, cross_section=cross_section)


################
# Bends
################


def bend_circular(
    *,
    wl: Float = WL0,
    radius: float | None = None,
    angle: float = 90.0,
    loss: float = LOSS_DB_CM,
    cross_section: str = "xs_sus",
) -> sax.SDict:
    """Returns the S-matrix of a circular bend (straight-waveguide propagation).

    Args:
        wl: wavelength in um.
        radius: center-line radius in um (defaults to the xs_sus radius).
        angle: bend angle in degrees.
        loss: propagation loss in dB/cm.
        cross_section: cross-section name (only xs_sus exists).
    """
    radius = TECH.radius_sus if radius is None else radius
    length = jnp.abs(jnp.deg2rad(angle)) * radius
    return straight(wl=wl, length=length, loss=loss, cross_section=cross_section)


def bend_euler(
    *,
    wl: Float = WL0,
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.5,
    loss: float = LOSS_DB_CM,
    cross_section: str = "xs_sus",
) -> sax.SDict:
    """Returns the S-matrix of an euler bend (straight-waveguide propagation).

    Args:
        wl: wavelength in um.
        radius: effective radius in um (defaults to the xs_sus radius).
        angle: bend angle in degrees.
        p: euler fraction of the bend.
        loss: propagation loss in dB/cm.
        cross_section: cross-section name (only xs_sus exists).
    """
    radius = TECH.radius_sus if radius is None else radius
    length = _euler_length(radius, angle, p)
    return straight(wl=wl, length=length, loss=loss, cross_section=cross_section)


def _cosine_sbend_length(dx: Float, dy: Float) -> Float:
    """Arc length of the raised-cosine centre line, in jax.numpy so it traces."""
    t = jnp.linspace(0.0, 1.0, 4001)
    slope = dy / 2 * jnp.pi / dx * jnp.sin(jnp.pi * t)
    return dx * jnp.trapezoid(jnp.hypot(1, slope), t)


def bend_s(
    *,
    wl: Float = WL0,
    size: tuple[float, float] = (40.0, 8.0),
    loss: float = LOSS_DB_CM,
    cross_section: str = "xs_sus",
) -> sax.SDict:
    """Returns the S-matrix of the cosine S-bend (straight-waveguide propagation).

    Args:
        wl: wavelength in um.
        size: S-bend length and height in um.
        loss: propagation loss in dB/cm.
        cross_section: cross-section name (only xs_sus exists).
    """
    length = _cosine_sbend_length(size[0], size[1])
    return straight(wl=wl, length=length, loss=loss, cross_section=cross_section)


################
# Grating couplers
################


def grating_coupler_rectangular(
    *,
    wl: Float = WL0,
    loss: float = GRATING_LOSS_DB,
    bandwidth: float = GRATING_BANDWIDTH,
    cross_section: str = "xs_sus",
) -> sax.SDict:
    """Returns the S-matrix of the foundry grating coupler (placeholder).

    o1 is the waveguide port and o2 the fiber port. The foundry library has
    no measured spectrum, so the peak loss and bandwidth are placeholders.

    Args:
        wl: wavelength in um.
        loss: peak insertion loss in dB.
        bandwidth: 3-dB bandwidth in um.
        cross_section: cross-section name (only xs_sus exists).
    """
    del cross_section
    return _grating_model(wl=jnp.asarray(wl), wl0=WL0, loss=loss, bandwidth=bandwidth)


################
# Models Dict
################


def get_models() -> dict[str, Callable[..., sax.SDict]]:
    """Returns the models of the suspended-Si cells, keyed by cell name."""
    return {
        "straight": straight,
        "taper": taper,
        "bend_circular": bend_circular,
        "bend_euler": bend_euler,
        "bend_s": bend_s,
        "grating_coupler_rectangular": grating_coupler_rectangular,
    }
