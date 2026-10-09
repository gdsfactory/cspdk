"""SAX models for S-parameter circuit simulations of the SiN200 visible PDK.

Each model dispatches on ``cross_section`` (``xs_n780``, ``xs_n638`` or
``xs_n520``), which is how layout netlists reference the band: the MZI netlist
instantiates ``mmi1x2`` with ``cross_section="xs_n638"``. Band aliases such as
``straight_n638`` are also registered.

Waveguide indices come from a femwell finite-element TE0 mode solve of the
strip waveguides drawn by this PDK (200 nm SiN core, 0.50/0.36/0.27 um wide,
2 um SiO2 BOX below and SiO2 cladding above, per the design guidelines), with
Si3N4 dispersion from Luke et al., Opt. Lett. 40, 4823 (2015) and SiO2 from
Malitson (1965). The group index includes material dispersion
(ng = neff - wl * dneff/dwl, central difference over +/-5 nm). The measured
Cornerstone LPCVD index (design guidelines, figure 3) would refine these.

Grating couplers use the insertion loss and 3 dB bandwidth published in the
standard-components PDF. Propagation loss defaults to 5 dB/cm, the foundry's
780 nm quality-assessment bound (design guidelines, table 3); no value is
published for 638 nm or 520 nm, so the same bound is used there.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import jax.numpy as jnp
import sax
import sax.models as sm
from numpy.typing import NDArray

from cspdk._models import _bend_s_length, _euler_length, _optical_model, _sdict_models

nm = 1e-3

FloatArray = NDArray[jnp.floating]
Float = float | FloatArray


class Band(NamedTuple):
    """Per-band optical parameters."""

    wl0: float  # centre wavelength (um)
    neff: float  # strip TE0 effective index at wl0
    ng: float  # strip TE0 group index at wl0
    radius: float  # default bend radius (um), from tech.Tech
    loss: float  # propagation loss (dB/cm)
    gc_loss: float  # grating coupler insertion loss (dB)
    gc_bandwidth: float  # grating coupler 3 dB bandwidth (um)


BANDS: dict[str, Band] = {
    "xs_n780": Band(0.78, 1.6609, 2.1009, 60.0, 5.0, 9.0, 64 * nm),
    "xs_n638": Band(0.638, 1.6951, 2.1823, 40.0, 5.0, 15.06, 43 * nm),
    "xs_n520": Band(0.52, 1.7381, 2.2756, 30.0, 5.0, 13.25, 28 * nm),
}


def _band(cross_section: str) -> Band:
    try:
        return BANDS[cross_section]
    except (KeyError, TypeError):
        msg = (
            f"No SiN200 model for cross_section={cross_section!r}; "
            f"expected one of {sorted(BANDS)}."
        )
        raise ValueError(msg) from None


_straight_model = _optical_model(sm.straight, 1, 1)
_mmi1x2_model = _optical_model(sm.mmi1x2, 1, 2)
_mmi2x2_model = _optical_model(sm.mmi2x2, 2, 2)
_grating_model = _optical_model(sm.grating_coupler, 1, 1)


################
# Straights
################


def straight(
    *,
    wl: Float = 0.78,
    length: float = 10.0,
    loss: float | None = None,
    cross_section: str = "xs_n780",
) -> sax.SDict:
    """Returns the S-matrix of a straight waveguide.

    Args:
        wl: wavelength in um.
        length: length in um.
        loss: propagation loss in dB/cm (defaults to the band value).
        cross_section: band cross-section name.
    """
    band = _band(cross_section)
    return _straight_model(
        wl=jnp.asarray(wl),
        length=length,
        loss_dB_cm=band.loss if loss is None else loss,
        wl0=band.wl0,
        neff=band.neff,
        ng=band.ng,
    )


straight_n780 = partial(straight, cross_section="xs_n780")
straight_n638 = partial(straight, cross_section="xs_n638", wl=0.638)
straight_n520 = partial(straight, cross_section="xs_n520", wl=0.52)


################
# Bends
################


def wire_corner(*, wl: Float = 0.78) -> sax.SDict:
    """Returns the S-matrix of a wire corner (electrical, no optical path)."""
    zero = jnp.zeros_like(jnp.asarray(wl))
    return sax.reciprocal({("e1", "e2"): zero})


def bend_euler(
    *,
    wl: Float = 0.78,
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.5,
    length: float | None = None,
    loss: float | None = None,
    cross_section: str = "xs_n780",
) -> sax.SDict:
    """Returns the S-matrix of an euler bend, modelled as a straight.

    Layout netlists pass ``radius``, ``angle`` and ``p`` but not the path
    length, so the length defaults to that of the drawn euler bend
    (``gf.path.euler(radius, angle, p, use_eff=True)``). Pass ``length`` to
    override.

    Args:
        wl: wavelength in um.
        radius: effective bend radius in um (defaults to the band radius).
        angle: bend angle in degrees.
        p: fraction of the bend that is an euler curve.
        length: path length in um (overrides radius, angle and p).
        loss: propagation loss in dB/cm (defaults to the band value).
        cross_section: band cross-section name.
    """
    if length is None:
        r = _band(cross_section).radius if radius is None else radius
        length = _euler_length(r, angle, p)
    return straight(wl=wl, length=length, loss=loss, cross_section=cross_section)


bend_euler_n780 = partial(bend_euler, cross_section="xs_n780")
bend_euler_n638 = partial(bend_euler, cross_section="xs_n638", wl=0.638)
bend_euler_n520 = partial(bend_euler, cross_section="xs_n520", wl=0.52)


def bend_s(
    *,
    wl: Float = 0.78,
    size: tuple[float, float] = (15.0, 1.8),
    length: float | None = None,
    loss: float | None = None,
    cross_section: str = "xs_n780",
) -> sax.SDict:
    """Returns the S-matrix of an S-bend, modelled as a straight.

    Args:
        wl: wavelength in um.
        size: S-bend (dx, dy) in um; the default path length is that of the
            drawn Bezier S-bend.
        length: path length in um (overrides size).
        loss: propagation loss in dB/cm (defaults to the band value).
        cross_section: band cross-section name.
    """
    if length is None:
        length = _bend_s_length(size[0], size[1])
    return straight(wl=wl, length=length, loss=loss, cross_section=cross_section)


################
# Transitions
################


def taper(
    *,
    wl: Float = 0.78,
    length: float = 10.0,
    loss: float | None = None,
    cross_section: str = "xs_n780",
) -> sax.SDict:
    """Returns the S-matrix of a taper, modelled as a straight of the band.

    Args:
        wl: wavelength in um.
        length: taper length in um.
        loss: propagation loss in dB/cm (defaults to the band value).
        cross_section: band cross-section name.
    """
    return straight(wl=wl, length=length, loss=loss, cross_section=cross_section)


taper_n780 = partial(taper, cross_section="xs_n780")
taper_n638 = partial(taper, cross_section="xs_n638", wl=0.638)
taper_n520 = partial(taper, cross_section="xs_n520", wl=0.52)


################
# MMIs
################


def mmi1x2(
    *,
    wl: Float = 0.78,
    loss_dB: Float = 0.3,
    fwhm: float = 0.2,
    cross_section: str = "xs_n780",
) -> sax.SDict:
    """Returns the S-matrix of a 1x2 MMI centred on the band wavelength.

    Args:
        wl: wavelength in um.
        loss_dB: excess loss in dB (placeholder; not published by the foundry).
        fwhm: spectral width of the splitting response in um (placeholder).
        cross_section: band cross-section name.
    """
    return _mmi1x2_model(
        wl=jnp.asarray(wl),
        wl0=_band(cross_section).wl0,
        fwhm=fwhm,
        loss_dB=loss_dB,
    )


mmi1x2_n780 = partial(mmi1x2, cross_section="xs_n780")
mmi1x2_n638 = partial(mmi1x2, cross_section="xs_n638", wl=0.638)
mmi1x2_n520 = partial(mmi1x2, cross_section="xs_n520", wl=0.52)


def mmi2x2(
    *,
    wl: Float = 0.78,
    loss_dB: Float = 0.3,
    fwhm: float = 0.2,
    cross_section: str = "xs_n780",
) -> sax.SDict:
    """Returns the S-matrix of a 2x2 MMI centred on the band wavelength.

    Args:
        wl: wavelength in um.
        loss_dB: excess loss in dB (placeholder; not published by the foundry).
        fwhm: spectral width of the splitting response in um (placeholder).
        cross_section: band cross-section name.
    """
    return _mmi2x2_model(
        wl=jnp.asarray(wl),
        wl0=_band(cross_section).wl0,
        fwhm=fwhm,
        loss_dB=loss_dB,
    )


mmi2x2_n780 = partial(mmi2x2, cross_section="xs_n780")
mmi2x2_n638 = partial(mmi2x2, cross_section="xs_n638", wl=0.638)
mmi2x2_n520 = partial(mmi2x2, cross_section="xs_n520", wl=0.52)


##############################
# Evanescent couplers
##############################


def coupler(
    *,
    wl: Float = 0.78,
    loss_dB: Float = 0.3,
    fwhm: float = 0.2,
    cross_section: str = "xs_n780",
) -> sax.SDict:
    """Returns the S-matrix of a directional coupler as an ideal 50/50 splitter.

    The default coupler gap and length are not foundry components, so the
    coupler is modelled as a 3 dB 2x2 splitter at the band wavelength; it does
    not depend on gap or length.

    Args:
        wl: wavelength in um.
        loss_dB: excess loss in dB.
        fwhm: spectral width of the splitting response in um.
        cross_section: band cross-section name.
    """
    return mmi2x2(wl=wl, loss_dB=loss_dB, fwhm=fwhm, cross_section=cross_section)


coupler_n780 = partial(coupler, cross_section="xs_n780")
coupler_n638 = partial(coupler, cross_section="xs_n638", wl=0.638)
coupler_n520 = partial(coupler, cross_section="xs_n520", wl=0.52)


##############################
# Grating couplers
##############################


def grating_coupler_rectangular(
    *,
    wl: Float = 0.78,
    loss: float | None = None,
    bandwidth: float | None = None,
    cross_section: str = "xs_n780",
) -> sax.SDict:
    """Returns the S-matrix of a foundry rectangular grating coupler.

    Defaults are the standard-components PDF values: 9 / 15.06 / 13.25 dB
    insertion loss and 64 / 43 / 28 nm 3 dB bandwidth at 780 / 638 / 520 nm.

    Args:
        wl: wavelength in um.
        loss: peak insertion loss in dB (defaults to the band value).
        bandwidth: 3 dB bandwidth in um (defaults to the band value).
        cross_section: band cross-section name.
    """
    band = _band(cross_section)
    return _grating_model(
        wl=jnp.asarray(wl),
        wl0=band.wl0,
        loss=band.gc_loss if loss is None else loss,
        bandwidth=band.gc_bandwidth if bandwidth is None else bandwidth,
    )


grating_coupler_rectangular_n780 = partial(
    grating_coupler_rectangular, cross_section="xs_n780"
)
grating_coupler_rectangular_n638 = partial(
    grating_coupler_rectangular, cross_section="xs_n638", wl=0.638
)
grating_coupler_rectangular_n520 = partial(
    grating_coupler_rectangular, cross_section="xs_n520", wl=0.52
)


def grating_coupler_elliptical(
    *,
    wl: Float = 0.78,
    loss: float | None = None,
    bandwidth: float | None = None,
    cross_section: str = "xs_n780",
) -> sax.SDict:
    """Returns the S-matrix of an elliptical grating coupler.

    The elliptical cells share the foundry rectangular pitch and fibre angle;
    no measurement exists, so they reuse the rectangular grating values.

    Args:
        wl: wavelength in um.
        loss: peak insertion loss in dB (defaults to the band value).
        bandwidth: 3 dB bandwidth in um (defaults to the band value).
        cross_section: band cross-section name.
    """
    return grating_coupler_rectangular(
        wl=wl, loss=loss, bandwidth=bandwidth, cross_section=cross_section
    )


grating_coupler_elliptical_n780 = partial(
    grating_coupler_elliptical, cross_section="xs_n780"
)
grating_coupler_elliptical_n638 = partial(
    grating_coupler_elliptical, cross_section="xs_n638", wl=0.638
)
grating_coupler_elliptical_n520 = partial(
    grating_coupler_elliptical, cross_section="xs_n520", wl=0.52
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
