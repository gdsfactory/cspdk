"""SAX models for Sparameter circuit simulations."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
import sax
import sax.models as sm
from numpy.typing import NDArray

from cspdk._models import _bend_s_length, _euler_length
from cspdk.si220.tech import TECH

from . import gratings, waveguides

FloatArray = NDArray[jnp.floating]
Float = float | FloatArray

################
# Straights
################

# loss_dB_cm: placeholder equal to C-band until measured O-band data is available
_straight_strip = partial(
    sm.straight,
    length=10.0,
    loss_dB_cm=3.0,
    wl0=1.31,
    neff=2.56,
    ng=4.34,
)

_straight_rib = partial(
    sm.straight,
    length=10.0,
    loss_dB_cm=3.0,
    wl0=1.31,
    neff=2.72,
    ng=3.98,
)


def _straight(
    *,
    wl: Float = 1.31,
    length: float = 10.0,
    loss_dB_cm: float = 3.0,
    cross_section: str = "strip",
) -> sax.SDict:
    """Straight waveguide model."""
    wl = jnp.asarray(wl)  # type: ignore
    fs = {
        "strip": _straight_strip,
        "rib": _straight_rib,
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


def _wire_corner(*, wl: Float = 1.31) -> sax.SDict:
    """Wire corner model."""
    wl = jnp.asarray(wl)  # type: ignore
    zero = jnp.zeros_like(wl)
    return sax.reciprocal({("e1", "e2"): zero})


def _bend_s(
    *,
    wl: Float = 1.31,
    size: tuple[float, float] = (11.0, 1.8),
    length: float | None = None,
    loss_dB_cm: float = 3.0,
    cross_section="strip",
) -> sax.SDict:
    """Bend S model.

    ``length=None`` is the drawn Bezier S-bend's path length (``_bend_s_length``, the
    99-point polyline ``gf.components.bend_s`` reports); pass ``length`` to override.
    """
    path_length = length if length is not None else _bend_s_length(size[0], size[1])
    return _straight(
        wl=wl,
        length=path_length,
        loss_dB_cm=loss_dB_cm,
        cross_section=cross_section,
    )


def _bend_euler(
    *,
    wl: Float = 1.31,
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.5,
    length: float | None = None,
    loss_dB_cm: float = 3.0,
    cross_section="strip",
) -> sax.SDict:
    """Euler bend model.

    Layout netlists pass ``radius``, ``angle`` and ``p`` but not the path length, so the
    length defaults to that of the drawn Euler bend (``_euler_length``); pass ``length``
    to override. ``radius=None`` is the cross-section radius, like the cell.
    """
    r = waveguides._BEND_RADIUS[cross_section] if radius is None else radius
    path_length = length if length is not None else _euler_length(r, angle, p)
    return _straight(
        wl=wl,
        length=path_length,
        loss_dB_cm=loss_dB_cm,
        cross_section=cross_section,
    )


_bend_euler_strip = partial(_bend_euler, cross_section="strip")
_bend_euler_rib = partial(_bend_euler, cross_section="rib")


################
# Transitions
################


def _taper(
    *,
    wl: Float = 1.31,
    length: float = 10.0,
    loss_dB_cm: float = 0.0,
    cross_section="strip",
) -> sax.SDict:
    """Taper model."""
    # ``length`` is a cell setting (``taper(length=...)``), so a layout netlist carries it
    # and this default is only the fallback for a circuit that sets nothing.
    # TODO: take width1 and width2 into account.
    return _straight(
        wl=wl,
        length=length,
        loss_dB_cm=loss_dB_cm,
        cross_section=cross_section,
    )


_taper_rib = partial(_taper, cross_section="rib")


def _taper_strip_to_ridge(
    *,
    wl: Float = 1.31,
    length: float = 10.0,
    loss_dB_cm: float = 0.0,
    cross_section="strip",
) -> sax.SDict:
    """Taper strip to ridge model."""
    # ``length`` is a cell setting (``taper_strip_to_ridge(length=...)``), so a layout
    # netlist carries it; this default is the fallback for a circuit that sets nothing.
    # TODO: take w_slab1 and w_slab2 into account.
    return _straight(
        wl=wl,
        length=length,
        loss_dB_cm=loss_dB_cm,
        cross_section=cross_section,
    )


_trans_rib10 = partial(_taper_strip_to_ridge, length=10.0)
_trans_rib20 = partial(_taper_strip_to_ridge, length=20.0)
_trans_rib50 = partial(_taper_strip_to_ridge, length=50.0)

# MMIs, crossings and the other Cornerstone components: see foundry.py.

##############################
# Evanescent couplers
##############################

# Shared with the C-band models (cspdk.si220.models.couplers) using O-band tables and
# waveguide models. Imported lazily: couplers imports this module.


def _directional_coupler(
    *,
    wl: Float = 1.31,
    length: float | None = None,
    gap: float = TECH.gap_strip,
    offset: float | None = None,
    bend_radius: float | None = None,
    cross_section: str = "strip",
) -> sax.SDict:
    """Directional coupler model (see couplers.directional_coupler)."""
    from cspdk.si220.models import couplers

    return couplers._directional_coupler(
        wl=wl,
        length=length,
        gap=gap,
        offset=offset,
        bend_radius=bend_radius,
        cross_section=cross_section,
        band="oband",
    )


_coupler = _directional_coupler
_coupler_strip = _directional_coupler
_coupler_rib = _directional_coupler


def _coupler_ring_coupling_area(
    *,
    wl: Float = 1.31,
    gap: float = 0.1,
    radius: float = 5.0,
    length_x: float = 1.0,
    loss_dB: float = 0.0,
    cross_section: str = "strip",
) -> sax.SDict:
    """Ring coupler coupling-region model (see couplers.coupler_ring_coupling_area)."""
    from cspdk.si220.models import couplers

    return couplers._coupler_ring_coupling_area(
        wl=wl,
        gap=gap,
        radius=radius,
        length_x=length_x,
        loss_dB=loss_dB,
        cross_section=cross_section,
        band="oband",
    )


def _coupler_ring(
    *,
    wl: Float = 1.31,
    gap: float = 0.1,
    radius: float = 40.0,
    length_x: float = 1.0,
    p: float = 0,
    loss_dB: float = 0.0,
    cross_section: str = "strip",
) -> sax.SDict:
    """Ring coupler model (see couplers.coupler_ring)."""
    from cspdk.si220.models import couplers

    return couplers._coupler_ring(
        wl=wl,
        gap=gap,
        radius=radius,
        length_x=length_x,
        p=p,
        loss_dB=loss_dB,
        cross_section=cross_section,
        band="oband",
    )


################
# Imported
################


# Same heater as the C-band model, with the O-band strip waveguide's indices.
_straight_heater_metal = partial(
    waveguides._straight_heater_metal, wl=1.31, wl0=1.31, neff=2.56, ng=4.34
)
_straight_heater_meander = _straight_heater_metal


################
# Models Dict
################


def get_models() -> dict[str, Callable[..., sax.SDict]]:
    """Return the O-band models keyed by model name."""
    return {
        "straight_strip": _straight_strip,
        "straight_rib": _straight_rib,
        "straight": _straight,
        "wire_corner": _wire_corner,
        "bend_s": _bend_s,
        "bend_euler": _bend_euler,
        "bend_euler_strip": _bend_euler_strip,
        "bend_euler_rib": _bend_euler_rib,
        "taper": _taper,
        "taper_rib": _taper_rib,
        "taper_strip_to_ridge": _taper_strip_to_ridge,
        "trans_rib10": _trans_rib10,
        "trans_rib20": _trans_rib20,
        "trans_rib50": _trans_rib50,
        "directional_coupler": _directional_coupler,
        "coupler": _coupler,
        "coupler_strip": _coupler_strip,
        "coupler_rib": _coupler_rib,
        "coupler_ring_coupling_area": _coupler_ring_coupling_area,
        "coupler_ring": _coupler_ring,
        "grating_coupler_rectangular_strip": gratings._grating_coupler_oband,
        "grating_coupler_rectangular_rib": gratings._grating_coupler_oband,
        "grating_coupler_rectangular": gratings._grating_coupler_oband,
        "grating_coupler_elliptical": gratings._grating_coupler_oband,
        "straight_heater_metal": _straight_heater_metal,
        "straight_heater_meander": _straight_heater_meander,
    }
