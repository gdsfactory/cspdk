"""SAX models for the CORNERSTONE SOI 500 nm (si500) PDK.

Waveguide indices
=================
``neff`` and ``ng`` come from a vectorial mode solve of the 42nd-call
500 nm SOI stack (``samples/mode_solver_r500.py``): 500 nm Si rib etched
300 nm (200 nm slab), 3 um SiO2 BOX and 2 um SiO2 top cladding, vertical
sidewalls, tidy3d material library Si and SiO2, solved with the local tidy3d
mode solver through ``gplugins.tidy3d`` (grid resolution 40, group index from
a +-10 nm finite difference) and cross-checked against femwell to within
0.01 in ``neff``:

==========  =====  ==========  ======  ======
xs          width  wavelength  neff    ng
==========  =====  ==========  ======  ======
xs_rc500    0.45   1.55        2.990   3.881
xs_ro500    0.40   1.31        3.099   3.933
==========  =====  ==========  ======  ======

The default propagation loss is 0 dB/cm; the 42nd-call design guidelines
only give a < 4 dB/cm qualification target.

Component data (42nd-call standard components)
==============================================
* Grating coupler (SOI500nm_1550nm_TE_RIB_Grating_Coupler): 5-6 dB coupling
  loss, 1 dB bandwidth > 30 nm, centre wavelength 1550-1570 nm. Modelled as a
  Gaussian with 5.5 dB loss at 1.56 um; the 1 dB bandwidth lower bound is
  converted to a 3 dB (FWHM) bandwidth of 30 nm / 0.5764 = 52 nm.
* MMIs (2x1 and 2x2): the standard components only show measured spectra,
  with no numbers, so the generic SAX MMI models use placeholder values
  (0.3 dB excess loss, 0.2 um FWHM) at 1.55 um.
* Directional coupler and elliptical grating: no foundry component; the
  coupler is a placeholder 50/50 splitter and the elliptical grating reuses
  the rectangular grating data.

O-band (``xs_ro500``)
=====================
The 500 nm platform is 1550 nm only: there is no O-band foundry data, and
whether the platform supports the O-band is an open question for
Cornerstone. The waveguide models (straight, bends, taper) use the
mode-solved ``xs_ro500`` indices above. The MMI, coupler and grating models
return O-band placeholders for ``xs_ro500`` so that the ``_ro`` layouts
(e.g. ``mzi_ro``) simulate: the same generic SAX responses and placeholder
losses as the C-band models (0.3 dB excess loss and 0.2 um FWHM for MMIs and
couplers; 5.5 dB loss and 52 nm FWHM for gratings), centred at 1.31 um. They
have no foundry basis.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
import numpy as np
import sax
import sax.models as sm
from numpy.typing import NDArray

from cspdk._models import _bend_s_length, _euler_length, _optical_model, _sdict_models

nm = 1e-3

FloatArray = NDArray[jnp.floating]
Float = float | FloatArray

# Mode-solved waveguide parameters (see module docstring).
WAVEGUIDES: dict[str, dict[str, float]] = {
    "xs_rc500": {"wl0": 1.55, "neff": 2.990, "ng": 3.881, "radius": 25.0},
    "xs_ro500": {"wl0": 1.31, "neff": 3.099, "ng": 3.933, "radius": 25.0},
}

# 1 dB full width of a Gaussian in units of its 3 dB full width.
_ONE_DB_PER_FWHM = float(np.sqrt(np.log(10**0.1) / np.log(2)))

GRATING_WL0 = 1.56
GRATING_LOSS_DB = 5.5
GRATING_BANDWIDTH = 30 * nm / _ONE_DB_PER_FWHM

# Placeholders: no numeric MMI data in the standard components document.
MMI_LOSS_DB = 0.3
MMI_FWHM = 0.2

# Centre wavelengths of the MMI, coupler and grating models per cross-section.
# The xs_ro500 entries are O-band placeholders (see module docstring).
MMI_WL0: dict[str, float] = {"xs_rc500": 1.55, "xs_ro500": 1.31}
GRATING_WL0S: dict[str, float] = {"xs_rc500": GRATING_WL0, "xs_ro500": 1.31}


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


def _waveguide(cross_section: str) -> dict[str, float]:
    return _lookup(WAVEGUIDES, cross_section)


################
# Straights
################


def straight(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss: float = 0.0,
    cross_section: str = "xs_rc500",
) -> sax.SDict:
    """Returns the S-matrix of a straight waveguide.

    Args:
        wl: Wavelength of the simulation in um.
        length: Length of the waveguide in um.
        loss: Propagation loss in dB/cm.
        cross_section: Cross section of the waveguide.
    """
    xs = _waveguide(cross_section)
    return _straight_model(
        wl=jnp.asarray(wl),
        length=length,
        loss_dB_cm=loss,
        wl0=xs["wl0"],
        neff=xs["neff"],
        ng=xs["ng"],
    )


straight_rc = partial(straight, cross_section="xs_rc500")
straight_ro = partial(straight, cross_section="xs_ro500", wl=1.31)


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
    cross_section: str = "xs_rc500",
) -> sax.SDict:
    """Returns the S-matrix of a circular bend (arc length radius * angle).

    Args:
        wl: Wavelength of the simulation in um.
        radius: Bend radius in um; defaults to the cross-section radius.
        angle: Bend angle in degrees.
        loss: Propagation loss in dB/cm.
        cross_section: Cross section of the bend.
    """
    radius = _waveguide(cross_section)["radius"] if radius is None else radius
    length = radius * jnp.deg2rad(jnp.abs(angle))
    return straight(wl=wl, length=length, loss=loss, cross_section=cross_section)


bend_circular_rc = partial(bend_circular, cross_section="xs_rc500")
bend_circular_ro = partial(bend_circular, cross_section="xs_ro500", wl=1.31)


def bend_euler(
    *,
    wl: Float = 1.55,
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.5,
    loss: float = 0.0,
    cross_section: str = "xs_rc500",
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
    radius = _waveguide(cross_section)["radius"] if radius is None else radius
    length = _euler_length(radius, angle, p)
    return straight(wl=wl, length=length, loss=loss, cross_section=cross_section)


bend_euler_rc = partial(bend_euler, cross_section="xs_rc500")
bend_euler_ro = partial(bend_euler, cross_section="xs_ro500", wl=1.31)


def bend_s(
    *,
    wl: Float = 1.55,
    size: tuple[float, float] = (20.0, 1.8),
    loss: float = 0.0,
    cross_section: str = "xs_rc500",
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
    cross_section: str = "xs_rc500",
) -> sax.SDict:
    """Returns the S-matrix of a taper, using the cross-section index.

    Args:
        wl: Wavelength of the simulation in um.
        length: Length of the taper in um.
        loss: Propagation loss in dB/cm.
        cross_section: Cross section of the taper.
    """
    return straight(wl=wl, length=length, loss=loss, cross_section=cross_section)


taper_rc = partial(taper, cross_section="xs_rc500", length=10.0)
taper_ro = partial(taper, cross_section="xs_ro500", length=10.0, wl=1.31)


################
# MMIs
################

mmi1x2_rc = partial(_mmi1x2_model, wl0=1.55, fwhm=MMI_FWHM, loss_dB=MMI_LOSS_DB)


def mmi1x2(
    wl: Float = 1.55,
    loss_dB: Float = MMI_LOSS_DB,
    cross_section: str = "xs_rc500",
) -> sax.SDict:
    """Returns the S-matrix of a 1x2 MMI (placeholder loss and bandwidth).

    ``xs_ro500`` is an O-band placeholder centred at 1.31 um (no foundry data).

    Args:
        wl: Wavelength of the simulation in um.
        loss_dB: Excess loss of the MMI in dB.
        cross_section: Cross section of the MMI.
    """
    return _mmi1x2_model(
        wl=jnp.asarray(wl),
        wl0=_lookup(MMI_WL0, cross_section),
        fwhm=MMI_FWHM,
        loss_dB=loss_dB,
    )


mmi2x2_rc = partial(_mmi2x2_model, wl0=1.55, fwhm=MMI_FWHM, loss_dB=MMI_LOSS_DB)


def mmi2x2(
    wl: Float = 1.55,
    loss_dB: Float = MMI_LOSS_DB,
    cross_section: str = "xs_rc500",
) -> sax.SDict:
    """Returns the S-matrix of a 2x2 MMI (placeholder loss and bandwidth).

    ``xs_ro500`` is an O-band placeholder centred at 1.31 um (no foundry data).

    Args:
        wl: Wavelength of the simulation in um.
        loss_dB: Excess loss of the MMI in dB.
        cross_section: Cross section of the MMI.
    """
    return _mmi2x2_model(
        wl=jnp.asarray(wl),
        wl0=_lookup(MMI_WL0, cross_section),
        fwhm=MMI_FWHM,
        loss_dB=loss_dB,
    )


##############################
# Evanescent couplers
##############################

coupler_rc = partial(_mmi2x2_model, wl0=1.55, fwhm=MMI_FWHM, loss_dB=MMI_LOSS_DB)


def coupler(
    wl: Float = 1.55,
    loss_dB: Float = MMI_LOSS_DB,
    cross_section: str = "xs_rc500",
) -> sax.SDict:
    """Returns the S-matrix of a coupler (placeholder 50/50 splitter).

    ``xs_ro500`` is an O-band placeholder centred at 1.31 um (no foundry data).

    Args:
        wl: Wavelength of the simulation in um.
        loss_dB: Excess loss of the coupler in dB.
        cross_section: Cross section of the coupler.
    """
    return _mmi2x2_model(
        wl=jnp.asarray(wl),
        wl0=_lookup(MMI_WL0, cross_section),
        fwhm=MMI_FWHM,
        loss_dB=loss_dB,
    )


##############################
# Grating couplers
##############################

grating_coupler_rectangular_rc = partial(
    _grating_model,
    loss=GRATING_LOSS_DB,
    bandwidth=GRATING_BANDWIDTH,
    wl=1.55,
    wl0=GRATING_WL0,
)


def grating_coupler_rectangular(
    wl: Float = 1.55,
    cross_section: str = "xs_rc500",
) -> sax.SDict:
    """Returns the S-matrix of the foundry rectangular grating coupler.

    ``xs_ro500`` is an O-band placeholder centred at 1.31 um with the C-band
    loss and bandwidth (no foundry data).

    Args:
        wl: Wavelength of the simulation in um.
        cross_section: Cross section of the grating's waveguide port.
    """
    return _grating_model(
        wl=jnp.asarray(wl),
        wl0=_lookup(GRATING_WL0S, cross_section),
        loss=GRATING_LOSS_DB,
        bandwidth=GRATING_BANDWIDTH,
    )


grating_coupler_elliptical_rc = partial(
    _grating_model,
    loss=GRATING_LOSS_DB,
    bandwidth=GRATING_BANDWIDTH,
    wl=1.55,
    wl0=GRATING_WL0,
)


def grating_coupler_elliptical(
    wl: Float = 1.55,
    cross_section: str = "xs_rc500",
) -> sax.SDict:
    """Returns the S-matrix of an elliptical grating (placeholder data).

    ``xs_ro500`` is an O-band placeholder centred at 1.31 um (no foundry data).

    Args:
        wl: Wavelength of the simulation in um.
        cross_section: Cross section of the grating's waveguide port.
    """
    return _grating_model(
        wl=jnp.asarray(wl),
        wl0=_lookup(GRATING_WL0S, cross_section),
        loss=GRATING_LOSS_DB,
        bandwidth=GRATING_BANDWIDTH,
    )


################
# Models Dict
################


def get_models() -> dict[str, Callable[..., sax.SDict]]:
    """Returns a dictionary of all models in this module."""
    return _sdict_models(globals())


if __name__ == "__main__":
    print(list(get_models()))
