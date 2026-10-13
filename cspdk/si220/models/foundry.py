"""Band implementations of the models of the Cornerstone Run-50 standard components.

Losses are the measured values of the 50th-call Standard Components PDF
(references/october-2026/CORNERSTONE-50th-Call-Standard-Components.pdf). Its MMI
numbers are the insertion loss per MMI along one output, so the ideal 3 dB split
(10 log10 2 = 3.01 dB) is taken off to get the excess loss. Where the PDF has no
O-band data the C-band value stands in, as for the waveguide loss.

All functions here are private: GDSFactory+ registers every public model it finds
in the package, so only the dispatchers in ``cspdk.si220.models`` are public.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import jax.numpy as jnp
import sax
import sax.models as sm

from cspdk.si220.tech import TECH

from . import oband, waveguides

_SPLIT_DB = 10 * math.log10(2)

# Excess loss (dB) of the foundry MMIs at the band centre: measured insertion loss
# per MMI minus the 3 dB split. C strip 1x2 3.3056 dB (p. 15), C rib 1x2 3.0621 dB
# (p. 8), O 1x2 3.2885 dB (p. 43; p. 39 repeats it as "old data" for rib).
MMI1X2_EXCESS_DB = {
    ("cband", "strip"): 3.3056 - _SPLIT_DB,
    ("cband", "rib"): 3.0621 - _SPLIT_DB,
    ("oband", "strip"): 3.2885 - _SPLIT_DB,
    ("oband", "rib"): 3.2885 - _SPLIT_DB,
}
# (thru, cross) excess loss (dB) of the 2x2 MMIs. C strip: bar 3.3673 dB, cross
# 3.3073 dB (p. 16). The C rib numbers (bar 1.7851 dB, cross 2.2493 dB, p. 9) are
# below the 3 dB split, so they cannot be per-MMI losses; the O-band pages (40, 44)
# have no data. Those keep the generic 0.3 dB.
MMI2X2_EXCESS_DB = {
    ("cband", "strip"): (3.3673 - _SPLIT_DB, 3.3073 - _SPLIT_DB),
    ("cband", "rib"): (0.3, 0.3),
    ("oband", "strip"): (0.3, 0.3),
    ("oband", "rib"): (0.3, 0.3),
}
# Crossing insertion loss (dB) at the 450 nm (C) / 400 nm (O) width: C rib p. 11,
# C strip p. 19, O rib p. 41, O strip p. 47.
CROSSING_LOSS_DB = {
    ("cband", "strip"): 0.1195,
    ("cband", "rib"): 0.12467,
    ("oband", "strip"): 0.24359,
    ("oband", "rib"): 0.2123,
}
# Loss per 90 degree bend (dB) against radius (um): C rib W 450 nm p. 10, C strip
# W 450 nm p. 18, O strip W 400 nm p. 46. No O rib data.
BEND_LOSS_DB = {
    ("cband", "strip"): ((5.0, 0.013603), (10.0, 0.011593)),
    ("cband", "rib"): ((10.0, 0.10059), (25.0, 0.030998)),
    ("oband", "strip"): ((5.0, 0.010597), (10.0, 0.016604)),
    ("oband", "rib"): ((10.0, 0.10059), (25.0, 0.030998)),
}
# Multimode (10 um) to single-mode taper loss (dB) against taper length (um):
# C rib p. 12, C strip p. 20. No O-band data.
MM2SM_LOSS_DB = {
    "strip": ((300.0, 0.13649), (450.0, 0.11019), (600.0, 0.14237)),
    "rib": ((300.0, 0.19143), (450.0, 0.13498), (600.0, 0.15717)),
}
# Rib-to-strip taper loss (dB) against slab taper length (um), p. 21.
RIB_TO_STRIP_LOSS_DB = ((10.0, 0.11916), (20.0, 0.09023), (50.0, 0.068069))

# Y-splitter excess loss (dB), by cut-back (p. 17 C band, p. 45 O band): through
# one output of a cascade the transmission falls by 4.0123 dB per splitter in the
# C band for 450 nm waveguides (3.7101 dB for 400 nm) and 4.5006 dB in the O band.
# That includes the 3 dB of an ideal 50/50 split, so the excess loss is 1.0020 dB
# (C band, the 450 nm template cell) and 1.4903 dB (O band).
YSPLITTER_EXCESS_DB = {"cband": 1.002, "oband": 1.4903}

# Offset (um) of the 1x2 MMI outputs minus that of the 2x2 MMI ports: the extra
# length of each MZI arm's combiner-side vertical (see cells.mzis.mzi_heater_loop).
_MZI_PORT_OFFSET = {"cband": 0.555, "oband": 0.51}


def _interp(x: float, table: tuple[tuple[float, float], ...]) -> jnp.ndarray:
    """Linear interpolation in a measured table, constant beyond its ends."""
    xs, ys = zip(*table)
    return jnp.interp(x, jnp.asarray(xs), jnp.asarray(ys))


def _amplitude(loss_dB) -> jnp.ndarray:
    return 10 ** (-jnp.asarray(loss_dB) / 20)


def _mmi2x2_asymmetric(*, wl, wl0, fwhm, loss_dB_thru, loss_dB_cross) -> sax.SDict:
    """``sax.models.mmi2x2`` with separate bar and cross losses.

    ``sax.models.mmi2x2`` cannot take them itself under ``jax.jit``.
    """
    s = sm.mmi2x2(wl=wl, wl0=wl0, fwhm=fwhm, loss_dB=0.0)
    thru = _amplitude(loss_dB_thru)
    cross = _amplitude(loss_dB_cross)
    return {
        (a, b): value * (thru if {a, b} in ({"o1", "o4"}, {"o2", "o3"}) else cross)
        for (a, b), value in s.items()
    }


def _band_models(band: str, wl0: float) -> dict[str, Callable[..., sax.SDict]]:
    """Models of the foundry components for ``band``, whose centre is ``wl0``."""
    straight = (oband if band == "oband" else waveguides)._straight

    def _phase(wl, length, cross_section):
        """Lossless propagation over ``length`` of the band's waveguide."""
        return straight(
            wl=wl, length=length, loss_dB_cm=0.0, cross_section=cross_section
        )

    def _two_port(s21, loss_dB) -> sax.SDict:
        return sax.reciprocal({("o1", "o2"): s21 * _amplitude(loss_dB)})

    def _mmi1x2(
        *,
        wl=wl0,
        extra_loss_dB=0.0,
        fwhm=0.2,
        cross_section: str = "strip",
    ) -> sax.SDict:
        """1x2 MMI with the foundry's measured excess loss plus ``extra_loss_dB``."""
        loss_dB = MMI1X2_EXCESS_DB[band, cross_section] + extra_loss_dB
        return sm.mmi1x2(wl=wl, wl0=wl0, fwhm=fwhm, loss_dB=loss_dB)

    def _mmi2x2(
        *,
        wl=wl0,
        extra_loss_dB=0.0,
        fwhm=0.2,
        cross_section: str = "strip",
    ) -> sax.SDict:
        """2x2 MMI with the foundry's measured excess losses plus ``extra_loss_dB``."""
        thru, cross = MMI2X2_EXCESS_DB[band, cross_section]
        return _mmi2x2_asymmetric(
            wl=wl,
            wl0=wl0,
            fwhm=fwhm,
            loss_dB_thru=thru + extra_loss_dB,
            loss_dB_cross=cross + extra_loss_dB,
        )

    def _crossing(*, wl=wl0, cross_section: str = "strip") -> sax.SDict:
        """Crossing with the foundry's measured insertion loss and no crosstalk."""
        t = jnp.asarray(wl) * 0 + _amplitude(CROSSING_LOSS_DB[band, cross_section])
        return sax.reciprocal({("o1", "o3"): t, ("o2", "o4"): t})

    def _bend_circular(
        *,
        wl=wl0,
        radius=None,
        angle=90.0,
        cross_section: str = "strip",
    ) -> sax.SDict:
        """Circular bend: arc-length propagation plus the measured loss per 90 deg."""
        if radius is None:
            radius = TECH.radius_rib if cross_section == "rib" else TECH.radius
        length = abs(angle) * math.pi / 180 * radius
        loss = _interp(radius, BEND_LOSS_DB[band, cross_section]) * abs(angle) / 90
        return _two_port(_phase(wl, length, cross_section)["o1", "o2"], loss)

    def _taper_mm2sm(
        *,
        wl=wl0,
        length=450.0,
        cross_section: str = "strip",
    ) -> sax.SDict:
        """Multimode-to-single-mode taper with the measured loss for its length."""
        loss = _interp(length, MM2SM_LOSS_DB[cross_section])
        return _two_port(_phase(wl, length, cross_section)["o1", "o2"], loss)

    def _taper_rib_to_strip(
        *,
        wl=wl0,
        length=50.0,
        length_rib=5.0,
        length_strip=5.0,
    ) -> sax.SDict:
        """Rib-to-strip taper with the measured loss for its slab taper length."""
        total = length + length_rib + length_strip
        loss = _interp(length, RIB_TO_STRIP_LOSS_DB)
        return _two_port(_phase(wl, total, "rib")["o1", "o2"], loss)

    def _heater(*, wl=wl0) -> sax.SDict:
        """Heater pads: an ideal electrical connection (the filament is not modelled)."""
        one = jnp.ones_like(jnp.asarray(wl, dtype=float))
        return sax.reciprocal({("e1", "e2"): one})

    def _mzi_heater_loop(
        *,
        wl=wl0,
        delta_length=160.0,
        length_x=616.3,
        length_y=35.0,
        radius=30.0,
        voltage=0.0,
        vpi=1.0,
        loss_dB_cm=3.0,
        cross_section: str = "strip",
    ) -> sax.SDict:
        """MZI of ``cells.mzi_heater_loop``: 1x2 MMI, two looped arms, 2x2 MMI.

        The arm straights lose ``loss_dB_cm`` like the straight model; the bends
        their measured loss. The heater on the upper arm adds a phase proportional
        to the voltage squared, with ``vpi`` giving a pi shift. Ports: o1 input,
        o2/o3 the 2x2 MMI's lower/upper ports.
        """
        splitter = _mmi1x2(wl=wl, cross_section=cross_section)
        combiner = _mmi2x2(wl=wl, cross_section=cross_section)
        bends = _bend_circular(wl=wl, radius=radius, cross_section=cross_section)
        bends4 = bends["o1", "o2"] ** 4
        common = 2 * length_y + length_x + _MZI_PORT_OFFSET[band]
        top = straight(
            wl=wl, length=common, loss_dB_cm=loss_dB_cm, cross_section=cross_section
        )["o1", "o2"]
        top = top * bends4 * jnp.exp(1j * jnp.pi * (voltage / vpi) ** 2)
        bottom = straight(
            wl=wl,
            length=common + delta_length,
            loss_dB_cm=loss_dB_cm,
            cross_section=cross_section,
        )["o1", "o2"]
        bottom = bottom * bends4
        # The splitter is rotated: its o3 feeds the upper arm, o2 the lower one.
        upper = splitter["o1", "o3"] * top
        lower = splitter["o1", "o2"] * bottom
        return sax.reciprocal(
            {
                ("o1", "o2"): upper * combiner["o3", "o1"]
                + lower * combiner["o4", "o1"],
                ("o1", "o3"): upper * combiner["o3", "o2"]
                + lower * combiner["o4", "o2"],
                ("e1", "e2"): jnp.ones_like(jnp.abs(upper)),
            }
        )

    def _ysplitter(*, wl=wl0, extra_loss_dB=0.0) -> sax.SDict:
        """Ideal 50/50 Y-splitter, flat in wavelength, with the measured excess loss.

        ``extra_loss_dB`` is added to the measured excess loss on top of the 3 dB split.
        """
        loss_dB = YSPLITTER_EXCESS_DB[band] + extra_loss_dB
        one = jnp.ones_like(jnp.asarray(wl, dtype=float))
        t = jnp.sqrt(0.5) * _amplitude(loss_dB) * one
        return sax.reciprocal({("o1", "o2"): t, ("o1", "o3"): t})

    return {
        "mmi1x2": _mmi1x2,
        "mmi2x2": _mmi2x2,
        "crossing": _crossing,
        "bend_circular": _bend_circular,
        "taper_mm2sm": _taper_mm2sm,
        "taper_rib_to_strip": _taper_rib_to_strip,
        "heater": _heater,
        "mzi_heater_loop": _mzi_heater_loop,
        "ysplitter": _ysplitter,
    }


def get_models(band: str) -> dict[str, Callable[..., sax.SDict]]:
    """Return the foundry component models of ``band`` keyed by model name."""
    return _band_models(band, 1.31 if band == "oband" else 1.55)
