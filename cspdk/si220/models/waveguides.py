"""SAX models for Sparameter circuit simulations."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import sax
import sax.models as sm
from numpy.typing import NDArray

from cspdk._models import _bend_s_length, _euler_length
from cspdk.si220.tech import TECH

sax.set_port_naming_strategy("optical")

nm = 1e-3

FloatArray = NDArray[jnp.floating]
Float = float | FloatArray

# Bend radius the cell uses when ``radius`` is not given -- ``bend_euler`` passes
# ``radius=None`` through to the cross-section, whose radius is this
# (``strip_cband`` -> ``TECH.radius_strip``, ``rib_cband`` -> ``TECH.radius_rib``).
# Keyed by this module's cross-section vocabulary, which ``_dispatch`` settles.
_BEND_RADIUS = {"strip": TECH.radius_strip, "rib": TECH.radius_rib}

################
# Straights
################


def _straight_strip(
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


def _straight_rib(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss_dB_cm: float = 3.0,
) -> sax.SDict:
    """Straight rib waveguide model."""
    # 450 nm x 220 nm rib with 100 nm slab, from samples/mode_solver_r.py
    return sm.straight(
        wl=wl,
        length=length,
        loss_dB_cm=loss_dB_cm,
        wl0=1.55,
        neff=2.54,
        ng=3.87,
    )


def _straight(
    *,
    wl: Float = 1.55,
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


def _wire_corner(*, wl: Float = 1.55) -> sax.SDict:
    """Wire corner model."""
    wl = jnp.asarray(wl)  # type: ignore
    zero = jnp.zeros_like(wl)
    return sax.reciprocal({("e1", "e2"): zero})


def _bend_s(
    *,
    wl: Float = 1.55,
    size: tuple[float, float] = (11.0, 1.8),
    length: float | None = None,
    loss_dB_cm: float = 3.0,
    cross_section="strip",
) -> sax.SDict:
    """Bend S model.

    Layout netlists pass ``size`` but not the path length, so the length defaults to
    that of the drawn Bezier S-bend (``gf.components.bend_s``'s 99-point polyline,
    reproduced by ``_bend_s_length``). Pass ``length`` to override.

    Args:
        wl: wavelength in um.
        size: S-bend (dx, dy) in um.
        length: path length in um (overrides ``size``).
        loss_dB_cm: propagation loss in dB/cm.
        cross_section: this module's cross-section vocabulary (``strip`` or ``rib``).
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
    wl: Float = 1.55,
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.5,
    length: float | None = None,
    loss_dB_cm: float = 3,
    cross_section="strip",
) -> sax.SDict:
    """Euler bend model.

    Layout netlists pass ``radius``, ``angle`` and ``p`` but not the path length, so
    the length defaults to that of the drawn Euler bend
    (``gf.path.euler(radius, angle, p, use_eff=True)``, the ``with_arc_floorplan=True``
    curve the cell draws, reproduced by ``_euler_length``). Pass ``length`` to override.

    Args:
        wl: wavelength in um.
        radius: effective bend radius in um (defaults to the cross-section radius).
        angle: bend angle in degrees.
        p: fraction of the bend that is an euler curve.
        length: path length in um (overrides ``radius``, ``angle`` and ``p``).
        loss_dB_cm: propagation loss in dB/cm.
        cross_section: this module's cross-section vocabulary (``strip`` or ``rib``).
    """
    r = _BEND_RADIUS[cross_section] if radius is None else radius
    path_length = length if length is not None else _euler_length(r, angle, p)
    return _straight(
        wl=wl,
        length=path_length,
        loss_dB_cm=loss_dB_cm,
        cross_section=cross_section,
    )


def _bend_euler_strip(
    *,
    wl: Float = 1.55,
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.5,
    length: float | None = None,
    loss_dB_cm: float = 3,
) -> sax.SDict:
    """Euler bend strip model."""
    return _bend_euler(
        wl=wl,
        radius=radius,
        angle=angle,
        p=p,
        length=length,
        loss_dB_cm=loss_dB_cm,
        cross_section="strip",
    )


def _bend_euler_rib(
    *,
    wl: Float = 1.55,
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.5,
    length: float | None = None,
    loss_dB_cm: float = 3,
) -> sax.SDict:
    """Euler bend rib model."""
    return _bend_euler(
        wl=wl,
        radius=radius,
        angle=angle,
        p=p,
        length=length,
        loss_dB_cm=loss_dB_cm,
        cross_section="rib",
    )


################
# Transitions
################


def _taper(
    *,
    wl: Float = 1.55,
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


def _taper_rib(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss_dB_cm: float = 0.0,
) -> sax.SDict:
    """Taper rib model."""
    return _taper(
        wl=wl,
        length=length,
        loss_dB_cm=loss_dB_cm,
        cross_section="rib",
    )


def _taper_strip_to_ridge(
    *,
    wl: Float = 1.55,
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


def _trans_rib10(
    *,
    wl: Float = 1.55,
    loss_dB_cm: float = 0.0,
    cross_section="strip",
) -> sax.SDict:
    """Taper strip to ridge 10um model."""
    return _taper_strip_to_ridge(
        wl=wl,
        length=10.0,
        loss_dB_cm=loss_dB_cm,
        cross_section=cross_section,
    )


def _trans_rib20(
    *,
    wl: Float = 1.55,
    loss_dB_cm: float = 0.0,
    cross_section="strip",
) -> sax.SDict:
    """Taper strip to ridge 20um model."""
    return _taper_strip_to_ridge(
        wl=wl,
        length=20.0,
        loss_dB_cm=loss_dB_cm,
        cross_section=cross_section,
    )


def _trans_rib50(
    *,
    wl: Float = 1.55,
    loss_dB_cm: float = 0.0,
    cross_section="strip",
) -> sax.SDict:
    """Taper strip to ridge 50um model."""
    return _taper_strip_to_ridge(
        wl=wl,
        length=50.0,
        loss_dB_cm=loss_dB_cm,
        cross_section=cross_section,
    )


def _straight_heater_metal(
    *,
    wl: Float = 1.55,
    wl0: float = 1.55,
    neff: float = 2.38,
    ng: float = 4.30,
    voltage: float = 0.0,
    vpi: float = 1.0,
    length: float = 10.0,
    loss_dB_cm: float = 3.0,
) -> sax.SDict:
    """Thermo-optic phase shifter on a strip waveguide.

    Propagation is the strip waveguide's, dispersive through ``neff`` and ``ng`` at
    ``wl0``. The heater adds a phase proportional to its dissipated power, so to the
    voltage squared, with ``vpi`` giving a π shift. Heating raises the index, so the
    added phase has the same sign as extra waveguide length.

    Args:
        wl: wavelength [µm].
        wl0: wavelength at which ``neff`` and ``ng`` are given [µm].
        neff: effective index of the strip waveguide.
        ng: group index of the strip waveguide.
        voltage: heater voltage [V].
        vpi: heater voltage for a π phase shift [V].
        length: heated waveguide length [µm].
        loss_dB_cm: propagation loss [dB/cm].
    """
    straight = sm.straight(
        wl=wl, wl0=wl0, neff=neff, ng=ng, length=length, loss_dB_cm=loss_dB_cm
    )
    transmission = straight["o1", "o2"] * jnp.exp(1j * jnp.pi * (voltage / vpi) ** 2)
    return sax.reciprocal(
        {
            ("o1", "o2"): transmission,
            ("l_e1", "r_e1"): 0.0,
            ("l_e2", "r_e2"): 0.0,
            ("l_e3", "r_e3"): 0.0,
            ("l_e4", "r_e4"): 0.0,
        }
    )


# The meander's netlist ``length`` is its heated length; the extra optical path through
# the meander bends is not modelled.
_straight_heater_meander = _straight_heater_metal


def get_models() -> dict[str, Callable[..., sax.SDict]]:
    """Return the C-band waveguide models keyed by model name."""
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
        "straight_heater_metal": _straight_heater_metal,
        "straight_heater_meander": _straight_heater_meander,
    }
