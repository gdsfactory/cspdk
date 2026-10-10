"""SAX models for the Cornerstone Ge-on-Si rib waveguide at 3.8 um (TE).

Effective and group index come from a femwell mode solve (order-2 elements) of
the foundry rib: 3.2 um wide, 3.0 um Ge, 1.8 um etch leaving a 1.2 um Ge floor
in the 20 um foundry trenches, 3.0 um Ge field beyond, Si substrate, air
cladding, vertical sidewalls. Material indices at 3.8 um: Ge 4.0276
(Icenogle 1976), Si 3.4239 (Li 1993, 293 K), both from the tidy3d material
library. The fundamental TE mode (TE fraction 0.999, 97% of the power in the
rib) has neff 3.9529 and ng 4.144 (central difference over +-10 nm),
unchanged to 1e-4 when the rib mesh is refined from 0.15 to 0.08 um and the
3 um field Ge beyond the trenches is included. The field's own planar modes
(neff 3.96-3.98) carry no power in the rib and are discarded; tunnelling
into them across the 20 um, 1.2 um-thick trench floor is negligible. The
other rib-confined modes are slab-like (neff <= 3.867, <= 52% power in the
rib), so the selection is unambiguous. Ge is lossless at 3.8 um in this
data, so the solve predicts no material loss.

The only foundry loss figure is the MPW qualification limit, < 5 dB/cm for
the TE mode at 3.8 um (MPW-4/MPW-7 sec. 6), so 5 dB/cm is the default:
a worst-case bound, not a measured value. Bends use the same propagation
loss; no bend radiation loss is modelled.

Bend models take the cell's geometry settings (radius, angle, p, size)
because SAX only passes instance settings, not the cell's ``info["length"]``.
Lengths are computed with jnp so settings can be traced under ``jax.jit``.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import sax
import sax.models as sm
from numpy.typing import NDArray

from cspdk._models import _bend_s_length, _euler_length, _optical_model, _sdict_models

FloatArray = NDArray[jnp.floating]
Float = float | FloatArray

WL0 = 3.8
NEFF = 3.9529
NG = 4.144
LOSS_DB_CM = 5.0
RADIUS = 300.0  # xs_rib default radius (foundry bend centre-line radius)


_straight_model = _optical_model(sm.straight, 1, 1)


################
# Straights
################


def straight(
    *,
    wl: Float = WL0,
    length: float = 10.0,
    loss: float = LOSS_DB_CM,
    cross_section: str = "xs_rib",
) -> sax.SDict:
    """Returns the S-matrix of a straight Ge rib waveguide.

    Args:
        wl: Wavelength in um.
        length: Length of the waveguide in um.
        loss: Propagation loss in dB/cm.
        cross_section: Cross section of the waveguide (only xs_rib exists).
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
    cross_section: str = "xs_rib",
) -> sax.SDict:
    """Returns the S-matrix of a circular bend (the foundry 90 deg bend).

    Args:
        wl: Wavelength in um.
        radius: Centre-line radius in um (defaults to the 300 um xs_rib radius).
        angle: Bend angle in degrees.
        loss: Propagation loss in dB/cm.
        cross_section: Cross section of the bend.
    """
    radius = RADIUS if radius is None else radius
    length = radius * jnp.deg2rad(jnp.abs(angle))
    return straight(wl=wl, length=length, loss=loss, cross_section=cross_section)


def bend_euler(
    *,
    wl: Float = WL0,
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.5,
    loss: float = LOSS_DB_CM,
    cross_section: str = "xs_rib",
) -> sax.SDict:
    """Returns the S-matrix of an euler bend.

    Args:
        wl: Wavelength in um.
        radius: Effective radius in um (defaults to the 300 um xs_rib radius).
        angle: Bend angle in degrees.
        p: Fraction of the bend that is a circular arc.
        loss: Propagation loss in dB/cm.
        cross_section: Cross section of the bend.
    """
    radius = RADIUS if radius is None else radius
    length = _euler_length(radius, angle, p)
    return straight(wl=wl, length=length, loss=loss, cross_section=cross_section)


def bend_s(
    *,
    wl: Float = WL0,
    size: tuple[float, float] = (100.0, 5.0),
    loss: float = LOSS_DB_CM,
    cross_section: str = "xs_rib",
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
# Transitions
################


def taper(
    *,
    wl: Float = WL0,
    length: float = 10.0,
    loss: float = LOSS_DB_CM,
    cross_section: str = "xs_rib",
) -> sax.SDict:
    """Returns the S-matrix of a taper, modelled as a 3.2 um rib straight.

    Args:
        wl: Wavelength in um.
        length: Length of the taper in um.
        loss: Propagation loss in dB/cm.
        cross_section: Cross section of the taper.
    """
    return straight(wl=wl, length=length, loss=loss, cross_section=cross_section)


################
# Models Dict
################


def get_models() -> dict[str, Callable[..., sax.SDict]]:
    """Returns a dictionary of all models in this module."""
    return _sdict_models(globals())


if __name__ == "__main__":
    for name, model in get_models().items():
        print(name, model())
