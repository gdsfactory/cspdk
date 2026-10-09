"""SAX models for the CORNERSTONE SOI 340 nm (si340) PDK.

Waveguide indices
=================
``neff`` and ``ng`` come from a vectorial mode solve of the 49th-call 340 nm
SOI stack (``samples/mode_solver.py``): 340 nm Si on a 2 um SiO2 BOX with a
1 um SiO2 top cladding; strip waveguides are fully etched and rib waveguides
are etched 140 nm (200 nm slab). Vertical sidewalls, tidy3d material library
Si and SiO2, local tidy3d mode solver through ``gplugins.tidy3d`` (grid
resolution 40, group index from a +-10 nm finite difference), cross-checked
against femwell:

==========  =====  ==========  ======  ======  ======
xs          width  wavelength  neff    ng      radius
==========  =====  ==========  ======  ======  ======
xs_sc340    0.45   1.55        2.661   4.281   10
xs_so340    0.40   1.31        2.821   4.283   10
xs_rc340    0.80   1.55        3.022   3.821   100
==========  =====  ==========  ======  ======  ======

The default propagation loss is 0 dB/cm (no foundry value is given).

Component data (49th-call standard components)
==============================================
* C-band strip grating (SOI340nm_1550nm_TE_STRIP_Grating_Coupler): 5-7 dB
  coupling loss, 1 dB bandwidth > 35 nm, centre 1550-1570 nm. Modelled as a
  Gaussian with 6 dB loss at 1.56 um; the 1 dB bandwidth lower bound is
  converted to a 3 dB (FWHM) bandwidth of 35 nm / 0.5764 = 61 nm.
* O-band strip grating (SOI340nm_1310nm_TE_STRIP_Grating_Coupler): no
  performance data; placeholder 6 dB loss and 35 nm FWHM at 1.31 um.
* MMIs: dimensions only, so the generic SAX MMI models use placeholder values
  (0.3 dB excess loss, 0.2 um FWHM) at the band centre.
* Rib-to-strip transition: lossless, with the mean rib and strip indices.

There is no foundry rib MMI, rib grating, or coupler on any cross-section;
those models are placeholders (C-band strip values for the rib, a 50/50
splitter for couplers) and the elliptical grating reuses the rectangular
grating data.
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

nm = 1e-3

FloatArray = NDArray[jnp.floating]
Float = float | FloatArray

# Mode-solved waveguide parameters (see module docstring).
WAVEGUIDES: dict[str, dict[str, float]] = {
    "xs_sc340": {"wl0": 1.55, "neff": 2.661, "ng": 4.281, "radius": 10.0},
    "xs_so340": {"wl0": 1.31, "neff": 2.821, "ng": 4.283, "radius": 10.0},
    "xs_rc340": {"wl0": 1.55, "neff": 3.022, "ng": 3.821, "radius": 100.0},
}

# 1 dB full width of a Gaussian in units of its 3 dB full width.
_ONE_DB_PER_FWHM = float(np.sqrt(np.log(10**0.1) / np.log(2)))

GRATINGS: dict[str, dict[str, float]] = {
    "xs_sc340": {"wl0": 1.56, "loss": 6.0, "bandwidth": 35 * nm / _ONE_DB_PER_FWHM},
    "xs_so340": {"wl0": 1.31, "loss": 6.0, "bandwidth": 35 * nm},  # placeholder
    "xs_rc340": {"wl0": 1.56, "loss": 6.0, "bandwidth": 35 * nm / _ONE_DB_PER_FWHM},
}

# Placeholders: no numeric MMI data in the standard components document.
MMI_LOSS_DB = 0.3
MMI_FWHM = 0.2


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


def _lookup(table: dict, cross_section: str):
    try:
        return table[cross_section]
    except KeyError:
        raise ValueError(
            f"No model for cross_section={cross_section!r}; "
            f"expected one of {sorted(table)}."
        ) from None


################
# Straights
################


def straight(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss: float = 0.0,
    cross_section: str = "xs_sc340",
) -> sax.SDict:
    """Returns the S-matrix of a straight waveguide.

    Args:
        wl: Wavelength of the simulation in um.
        length: Length of the waveguide in um.
        loss: Propagation loss in dB/cm.
        cross_section: Cross section of the waveguide.
    """
    xs = _lookup(WAVEGUIDES, cross_section)
    return _straight_model(
        wl=jnp.asarray(wl),
        length=length,
        loss_dB_cm=loss,
        wl0=xs["wl0"],
        neff=xs["neff"],
        ng=xs["ng"],
    )


straight_sc = partial(straight, cross_section="xs_sc340")
straight_so = partial(straight, cross_section="xs_so340")
straight_rc = partial(straight, cross_section="xs_rc340")


################
# Bends
################


def wire_corner(*, wl: Float = 1.55) -> sax.SDict:
    """Returns the S-matrix of a wire corner."""
    zero = jnp.zeros_like(jnp.asarray(wl))
    return sax.reciprocal({("e1", "e2"): zero})


def bend_circular(
    *,
    wl: Float = 1.55,
    radius: float | None = None,
    angle: float = 90.0,
    loss: float = 0.0,
    cross_section: str = "xs_sc340",
) -> sax.SDict:
    """Returns the S-matrix of a circular bend (arc length radius * angle).

    Args:
        wl: Wavelength of the simulation in um.
        radius: Bend radius in um; defaults to the cross-section radius.
        angle: Bend angle in degrees.
        loss: Propagation loss in dB/cm.
        cross_section: Cross section of the bend.
    """
    radius = _lookup(WAVEGUIDES, cross_section)["radius"] if radius is None else radius
    length = radius * jnp.deg2rad(jnp.abs(angle))
    return straight(wl=wl, length=length, loss=loss, cross_section=cross_section)


bend_circular_sc = partial(bend_circular, cross_section="xs_sc340")
bend_circular_so = partial(bend_circular, cross_section="xs_so340")
bend_circular_rc = partial(bend_circular, cross_section="xs_rc340")


def _euler_length(radius: Float, angle: Float, p: Float) -> Float:
    """Length of ``gf.path.euler(radius, angle, p, use_eff=True)``.

    Closed form of the gdsfactory construction (Euler sections of minimum
    radius 1, an arc in between, scaled so the end points match an arc of
    ``radius``), written with ``jax.numpy`` so it also works on traced
    settings inside a jitted ``sax.circuit``.
    """
    alpha = jnp.deg2rad(jnp.abs(angle))
    p = jnp.clip(p, 1e-9, 1.0)
    sp = jnp.sqrt(p * alpha)
    rp = 1 / sp
    s = jnp.linspace(0.0, 1.0, 257) * sp
    xp = jnp.trapezoid(jnp.cos(s**2 / 2), s)
    yp = jnp.trapezoid(jnp.sin(s**2 / 2), s)
    a1, a2 = p * alpha / 2, alpha / 2
    xh = rp * (jnp.sin(a2) - jnp.sin(a1)) + xp
    yh = rp * (jnp.cos(a1) - jnp.cos(a2)) + yp
    ex = xh + jnp.cos(alpha) * xh + jnp.sin(alpha) * yh
    ey = yh + jnp.sin(alpha) * xh - jnp.cos(alpha) * yh
    reff = jnp.where(
        jnp.abs(jnp.rad2deg(alpha) - 180) < 1e-3,
        ey / 2,
        ey - jnp.tan(alpha - jnp.pi / 2) * ex,
    )
    return radius * (2 * sp + rp * (1 - p) * alpha) / reff


def bend_euler(
    *,
    wl: Float = 1.55,
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.5,
    loss: float = 0.0,
    cross_section: str = "xs_sc340",
) -> sax.SDict:
    """Returns the S-matrix of an Euler bend from its path length.

    Args:
        wl: Wavelength of the simulation in um.
        radius: Effective bend radius in um; defaults to the cross-section radius.
        angle: Bend angle in degrees.
        p: Fraction of the bend that is an Euler curve.
        loss: Propagation loss in dB/cm.
        cross_section: Cross section of the bend.
    """
    radius = _lookup(WAVEGUIDES, cross_section)["radius"] if radius is None else radius
    length = _euler_length(radius, angle, p)
    return straight(wl=wl, length=length, loss=loss, cross_section=cross_section)


bend_euler_sc = partial(bend_euler, cross_section="xs_sc340")
bend_euler_so = partial(bend_euler, cross_section="xs_so340")
bend_euler_rc = partial(bend_euler, cross_section="xs_rc340")


def _bend_s_length(dx: Float, dy: Float) -> Float:
    """Length of the cubic Bezier S-bend drawn by ``gf.components.bend_s``."""
    t = jnp.linspace(0.0, 1.0, 1001)
    # derivative of the Bezier with control points (0,0) (dx/2,0) (dx/2,dy) (dx,dy)
    vx = 3 * (1 - t) ** 2 * dx / 2 + 3 * t**2 * dx / 2
    vy = 6 * (1 - t) * t * dy
    return jnp.trapezoid(jnp.hypot(vx, vy), t)


def bend_s(
    *,
    wl: Float = 1.55,
    size: tuple[float, float] = (20.0, 1.8),
    loss: float = 0.0,
    cross_section: str = "xs_sc340",
) -> sax.SDict:
    """Returns the S-matrix of an S-bend from its Bezier path length.

    Args:
        wl: Wavelength of the simulation in um.
        size: Length and height of the S-bend in um.
        loss: Propagation loss in dB/cm.
        cross_section: Cross section of the bend.
    """
    dx, dy = size
    return straight(
        wl=wl, length=_bend_s_length(dx, dy), loss=loss, cross_section=cross_section
    )


################
# Transitions
################


def taper(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss: float = 0.0,
    cross_section: str = "xs_sc340",
) -> sax.SDict:
    """Returns the S-matrix of a taper, using the cross-section index.

    Args:
        wl: Wavelength of the simulation in um.
        length: Length of the taper in um.
        loss: Propagation loss in dB/cm.
        cross_section: Cross section of the taper.
    """
    return straight(wl=wl, length=length, loss=loss, cross_section=cross_section)


taper_sc = partial(taper, cross_section="xs_sc340", length=10.0)
taper_so = partial(taper, cross_section="xs_so340", length=10.0)
taper_rc = partial(taper, cross_section="xs_rc340", length=10.0)


def taper_rib_to_strip(
    *,
    wl: Float = 1.55,
    length: float = 200.0,
    loss: float = 0.0,
) -> sax.SDict:
    """Returns the S-matrix of the rib-to-strip transition.

    Lossless by default, with the mean of the rib and strip C-band indices.

    Args:
        wl: Wavelength of the simulation in um.
        length: Length of the transition in um.
        loss: Propagation loss in dB/cm.
    """
    rib, strip = WAVEGUIDES["xs_rc340"], WAVEGUIDES["xs_sc340"]
    return _straight_model(
        wl=jnp.asarray(wl),
        length=length,
        loss_dB_cm=loss,
        wl0=1.55,
        neff=(rib["neff"] + strip["neff"]) / 2,
        ng=(rib["ng"] + strip["ng"]) / 2,
    )


################
# MMIs
################

mmi1x2_sc = partial(_mmi1x2_model, wl0=1.55, fwhm=MMI_FWHM, loss_dB=MMI_LOSS_DB)
mmi1x2_so = partial(_mmi1x2_model, wl0=1.31, fwhm=MMI_FWHM, loss_dB=MMI_LOSS_DB)
mmi1x2_rc = partial(_mmi1x2_model, wl0=1.55, fwhm=MMI_FWHM, loss_dB=MMI_LOSS_DB)


def mmi1x2(
    wl: Float = 1.55,
    loss_dB: Float = MMI_LOSS_DB,
    cross_section: str = "xs_sc340",
) -> sax.SDict:
    """Returns the S-matrix of a 1x2 MMI (placeholder loss and bandwidth).

    Args:
        wl: Wavelength of the simulation in um.
        loss_dB: Excess loss of the MMI in dB.
        cross_section: Cross section of the MMI.
    """
    fs = {"xs_sc340": mmi1x2_sc, "xs_so340": mmi1x2_so, "xs_rc340": mmi1x2_rc}
    return _lookup(fs, cross_section)(wl=jnp.asarray(wl), loss_dB=loss_dB)


mmi2x2_sc = partial(_mmi2x2_model, wl0=1.55, fwhm=MMI_FWHM, loss_dB=MMI_LOSS_DB)
mmi2x2_so = partial(_mmi2x2_model, wl0=1.31, fwhm=MMI_FWHM, loss_dB=MMI_LOSS_DB)
mmi2x2_rc = partial(_mmi2x2_model, wl0=1.55, fwhm=MMI_FWHM, loss_dB=MMI_LOSS_DB)


def mmi2x2(
    wl: Float = 1.55,
    loss_dB: Float = MMI_LOSS_DB,
    cross_section: str = "xs_sc340",
) -> sax.SDict:
    """Returns the S-matrix of a 2x2 MMI (placeholder loss and bandwidth).

    Args:
        wl: Wavelength of the simulation in um.
        loss_dB: Excess loss of the MMI in dB.
        cross_section: Cross section of the MMI.
    """
    fs = {"xs_sc340": mmi2x2_sc, "xs_so340": mmi2x2_so, "xs_rc340": mmi2x2_rc}
    return _lookup(fs, cross_section)(wl=jnp.asarray(wl), loss_dB=loss_dB)


##############################
# Evanescent couplers
##############################

coupler_sc = partial(_mmi2x2_model, wl0=1.55, fwhm=MMI_FWHM, loss_dB=MMI_LOSS_DB)
coupler_so = partial(_mmi2x2_model, wl0=1.31, fwhm=MMI_FWHM, loss_dB=MMI_LOSS_DB)
coupler_rc = partial(_mmi2x2_model, wl0=1.55, fwhm=MMI_FWHM, loss_dB=MMI_LOSS_DB)


def coupler(
    wl: Float = 1.55,
    loss_dB: Float = MMI_LOSS_DB,
    cross_section: str = "xs_sc340",
) -> sax.SDict:
    """Returns the S-matrix of a coupler (placeholder 50/50 splitter).

    Args:
        wl: Wavelength of the simulation in um.
        loss_dB: Excess loss of the coupler in dB.
        cross_section: Cross section of the coupler.
    """
    fs = {"xs_sc340": coupler_sc, "xs_so340": coupler_so, "xs_rc340": coupler_rc}
    return _lookup(fs, cross_section)(wl=jnp.asarray(wl), loss_dB=loss_dB)


##############################
# Grating couplers
##############################

grating_coupler_rectangular_sc = partial(
    _grating_model, wl=1.55, **GRATINGS["xs_sc340"]
)
grating_coupler_rectangular_so = partial(
    _grating_model, wl=1.31, **GRATINGS["xs_so340"]
)
grating_coupler_rectangular_rc = partial(
    _grating_model, wl=1.55, **GRATINGS["xs_rc340"]
)


def grating_coupler_rectangular(
    wl: Float = 1.55,
    cross_section: str = "xs_sc340",
) -> sax.SDict:
    """Returns the S-matrix of a foundry rectangular grating coupler.

    Args:
        wl: Wavelength of the simulation in um.
        cross_section: Cross section of the grating's waveguide port.
    """
    return _grating_model(wl=jnp.asarray(wl), **_lookup(GRATINGS, cross_section))


grating_coupler_elliptical_sc = partial(_grating_model, wl=1.55, **GRATINGS["xs_sc340"])
grating_coupler_elliptical_so = partial(_grating_model, wl=1.31, **GRATINGS["xs_so340"])
grating_coupler_elliptical_rc = partial(_grating_model, wl=1.55, **GRATINGS["xs_rc340"])


def grating_coupler_elliptical(
    wl: Float = 1.55,
    cross_section: str = "xs_sc340",
) -> sax.SDict:
    """Returns the S-matrix of an elliptical grating (placeholder data).

    Args:
        wl: Wavelength of the simulation in um.
        cross_section: Cross section of the grating's waveguide port.
    """
    return _grating_model(wl=jnp.asarray(wl), **_lookup(GRATINGS, cross_section))


################
# Models Dict
################


def get_models() -> dict[str, Callable[..., sax.SDict]]:
    """Returns a dictionary of all models in this module."""
    models = {}
    for name, func in list(globals().items()):
        if name.startswith("_"):
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
