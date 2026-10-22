"""SAX models of cspdk.si220_ebeam_event (1550 nm TE).

* Rib n_g 3.85 and 2.5 dB/cm: CORNERSTONE's event spec.
* neff 2.54: placeholder, the cspdk si220 rib mode solve of the same geometry.
* Grating: the student-batch rib grating datasheet (5.5-6.5 dB, 1 dB bandwidth
  > 35 nm, centre 1550-1580 nm), as a Gaussian: 6 dB at 1.565 um, 61 nm FWHM.
* Couplers: placeholders. The power coupling is a parameter, with no gap
  dependence; phase and loss are the rib's over the coupler's path.

Bend models take the cell's geometry settings because SAX passes instance
settings, not ``info["length"]``. No bend radiation loss is modelled.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import sax
import sax.models as sm
from numpy.typing import NDArray

from cspdk._models import _bend_s_length, _euler_length, _optical_model, _sdict_models
from cspdk.si220_ebeam_event.tech import TECH, nm

FloatArray = NDArray[jnp.floating]
Float = float | FloatArray

WL0 = 1.55
NEFF = 2.54  # placeholder
NG = 3.85
LOSS_DB_CM = 2.5

# 1 dB full width of a Gaussian in units of its 3 dB full width.
_ONE_DB_PER_FWHM = float(np.sqrt(np.log(10**0.1) / np.log(2)))

GRATING_WL0 = 1.565
GRATING_LOSS_DB = 6.0
GRATING_BANDWIDTH = 35 * nm / _ONE_DB_PER_FWHM

# Placeholder power coupling ratios.
COUPLER_COUPLING = 0.5
COUPLER_RING_COUPLING = 0.1

_straight_model = _optical_model(sm.straight, 1, 1)
_grating_model = _optical_model(sm.grating_coupler, 1, 1)


def _transmission(wl: Float, length: Float, loss: float) -> Float:
    """Complex field transmission of the rib waveguide over ``length`` um."""
    return straight(wl=wl, length=length, loss=loss)["o1", "o2"]


def _coupler_sdict(through: Float, cross: Float) -> sax.SDict:
    """2x2 coupler with ports o1, o2 (west) and o3, o4 (east).

    o1 faces o4 and o2 faces o3 (the gdsfactory coupler port layout).
    """
    return sax.reciprocal(
        {
            ("o1", "o4"): through,
            ("o1", "o3"): cross,
            ("o2", "o3"): through,
            ("o2", "o4"): cross,
        }
    )


################
# Straights
################


def straight(
    *,
    wl: Float = WL0,
    length: float = 10.0,
    loss: float = LOSS_DB_CM,
    cross_section: str = "rib",
) -> sax.SDict:
    """Returns the S-matrix of a straight rib waveguide.

    Args:
        wl: Wavelength in um.
        length: Length of the waveguide in um.
        loss: Propagation loss in dB/cm.
        cross_section: Cross section of the waveguide (only rib exists).
    """
    del cross_section
    return _straight_model(
        wl=jnp.asarray(wl),
        length=length,
        loss_dB_cm=loss,
        wl0=WL0,
        neff=NEFF,
        ng=NG,
    )


################
# Bends
################


def bend_circular(
    *,
    wl: Float = WL0,
    radius: float | None = None,
    angle: float = 90.0,
    loss: float = LOSS_DB_CM,
    cross_section: str = "rib",
) -> sax.SDict:
    """Returns the S-matrix of a circular bend.

    Args:
        wl: Wavelength in um.
        radius: Centre-line radius in um (defaults to the 25 um rib radius).
        angle: Bend angle in degrees.
        loss: Propagation loss in dB/cm.
        cross_section: Cross section of the bend.
    """
    radius = TECH.radius_rib if radius is None else radius
    length = radius * jnp.deg2rad(jnp.abs(angle))
    return straight(wl=wl, length=length, loss=loss, cross_section=cross_section)


def bend_euler(
    *,
    wl: Float = WL0,
    radius: float = TECH.radius_euler,
    angle: float = 90.0,
    p: float = 0.5,
    loss: float = LOSS_DB_CM,
    cross_section: str = "rib",
) -> sax.SDict:
    """Returns the S-matrix of an Euler bend.

    Args:
        wl: Wavelength in um.
        radius: Effective radius in um (the bend_euler cell default, 36 um).
        angle: Bend angle in degrees.
        p: Fraction of the bend that is an Euler curve.
        loss: Propagation loss in dB/cm.
        cross_section: Cross section of the bend.
    """
    length = _euler_length(radius, angle, p)
    return straight(wl=wl, length=length, loss=loss, cross_section=cross_section)


def bend_s(
    *,
    wl: Float = WL0,
    size: tuple[float, float] = (50.0, 5.0),
    loss: float = LOSS_DB_CM,
    cross_section: str = "rib",
) -> sax.SDict:
    """Returns the S-matrix of a Bezier S-bend.

    Args:
        wl: Wavelength in um.
        size: S-bend (dx, dy) in um.
        loss: Propagation loss in dB/cm.
        cross_section: Cross section of the bend.
    """
    length = _bend_s_length(size[0], size[1])
    return straight(wl=wl, length=length, loss=loss, cross_section=cross_section)


################
# Couplers
################


def coupler(
    *,
    wl: Float = WL0,
    gap: float = TECH.gap,
    length: float = 20.0,
    dx: float = 16.0,
    dy: float = 3.5,
    coupling: float = COUPLER_COUPLING,
    loss: float = LOSS_DB_CM,
    cross_section: str = "rib",
) -> sax.SDict:
    """Returns the S-matrix of a directional coupler (placeholder coupling).

    Each arm is the coupling straight plus two S-bends of offset
    (dy - gap - width) / 2.

    Args:
        wl: Wavelength in um.
        gap: Gap of the coupling region in um (sets the S-bend offset only).
        length: Length of the coupling region in um.
        dx: Length of each S-bend in um.
        dy: Height of each S-bend in um.
        coupling: Power coupling ratio (0 to 1).
        loss: Propagation loss in dB/cm.
        cross_section: Cross section of the coupler.
    """
    del cross_section
    offset = (dy - gap - TECH.width_rib) / 2
    arm = _transmission(wl, length + 2 * _bend_s_length(dx, offset), loss)
    return _coupler_sdict(
        through=jnp.sqrt(1 - coupling) * arm, cross=1j * jnp.sqrt(coupling) * arm
    )


def coupler_ring(
    *,
    wl: Float = WL0,
    radius: float = TECH.radius_rib,
    length_x: float = 4.0,
    coupling: float = COUPLER_RING_COUPLING,
    loss: float = LOSS_DB_CM,
    cross_section: str = "rib",
) -> sax.SDict:
    """Returns the S-matrix of a ring coupler (placeholder coupling).

    Bus o1 to o4; ring ports o2 and o3 at the ends of the quarter-circle bends.
    The bus extensions are not modelled.

    Args:
        wl: Wavelength in um.
        radius: Ring bend radius in um.
        length_x: Length of the straight coupling section in um.
        coupling: Power coupling ratio between bus and ring (0 to 1).
        loss: Propagation loss in dB/cm.
        cross_section: Cross section of the coupler.
    """
    del cross_section
    t = _transmission(wl, length_x, loss)
    bend = _transmission(wl, jnp.pi * radius / 2, loss)
    tau = jnp.sqrt(1 - coupling) * t
    kappa = 1j * jnp.sqrt(coupling) * t
    return sax.reciprocal(
        {
            ("o1", "o4"): tau,
            ("o1", "o3"): kappa * bend,
            ("o2", "o4"): bend * kappa,
            ("o2", "o3"): bend * tau * bend,
        }
    )


################
# Grating couplers
################


def grating_coupler(
    *,
    wl: Float = WL0,
    loss: float = GRATING_LOSS_DB,
    bandwidth: float = GRATING_BANDWIDTH,
    cross_section: str = "rib",
) -> sax.SDict:
    """Returns the S-matrix of the grating coupler (student-batch datasheet).

    Args:
        wl: Wavelength in um.
        loss: Peak coupling loss in dB.
        bandwidth: 3 dB (full width at half maximum) bandwidth in um.
        cross_section: Cross section of the grating's waveguide port.
    """
    del cross_section
    return _grating_model(
        wl=jnp.asarray(wl), wl0=GRATING_WL0, loss=loss, bandwidth=bandwidth
    )


################
# Models Dict
################


def get_models() -> dict[str, Callable[..., sax.SDict]]:
    """Returns a dictionary of all models in this module."""
    return _sdict_models(globals())


if __name__ == "__main__":
    for name, model in get_models().items():
        print(name, model())
