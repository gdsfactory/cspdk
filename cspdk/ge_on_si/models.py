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

import inspect
from collections.abc import Callable
from functools import partial, wraps

import jax.numpy as jnp
import numpy as np
import sax
import sax.models as sm
from numpy.typing import NDArray

FloatArray = NDArray[jnp.floating]
Float = float | FloatArray

WL0 = 3.8
NEFF = 3.9529
NG = 4.144
LOSS_DB_CM = 5.0
RADIUS = 300.0  # xs_rib default radius (foundry bend centre-line radius)


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


_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(32)


def _euler_length(radius: Float, angle: Float, p: Float) -> Float:
    """Centre-line length of the PDK euler bend (``with_arc_floorplan=True``).

    Closed form of ``gf.path.euler(radius, angle, p, use_eff=True).length()``:
    the unit clothoid/arc/clothoid curve has length ``s0`` and is scaled so its
    endpoints match an arc of ``radius``. Uses jnp so circuit settings can be
    traced under ``jax.jit``.
    """
    alpha = jnp.deg2rad(jnp.abs(angle))
    p = jnp.clip(p, 1e-12, 1.0)  # p -> 0 is the circular-arc limit
    sp = jnp.sqrt(p * alpha)
    rp = 1 / sp
    # Clothoid end point: sqrt(2) * int_0^u (cos t^2, sin t^2) dt, u = sp/sqrt(2).
    u = sp / np.sqrt(2)
    t = u * (_GL_NODES + 1) / 2
    w = u * _GL_WEIGHTS / 2
    xp = np.sqrt(2) * jnp.sum(w * jnp.cos(t**2))
    yp = np.sqrt(2) * jnp.sum(w * jnp.sin(t**2))
    # Midpoint of the symmetric curve (end of the half arc at alpha/2).
    x1 = rp * (jnp.sin(alpha / 2) - jnp.sin(p * alpha / 2)) + xp
    y1 = rp * (jnp.cos(p * alpha / 2) - jnp.cos(alpha / 2)) + yp
    r_eff = (x1 * jnp.cos(alpha / 2) + y1 * jnp.sin(alpha / 2)) / jnp.sin(alpha / 2)
    s0 = 2 * sp + rp * alpha * (1 - p)
    return s0 * radius / r_eff


def _bend_s_length(dx: Float, dy: Float, npoints: int = 99) -> Float:
    """Centre-line length of gdsfactory's bend_s (99-point cubic Bezier polyline)."""
    t = np.linspace(0, 1, npoints)
    # Control points (0, 0), (dx/2, 0), (dx/2, dy), (dx, dy).
    x = (3 * (1 - t) ** 2 * t + 3 * (1 - t) * t**2) * dx / 2 + t**3 * dx
    y = (3 * (1 - t) * t**2 + t**3) * dy
    return jnp.sum(jnp.hypot(jnp.diff(x), jnp.diff(y)))


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
    models = {}
    for name, func in list(globals().items()):
        if name.startswith("_") or not callable(func):
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
    for name, model in get_models().items():
        print(name, model())
