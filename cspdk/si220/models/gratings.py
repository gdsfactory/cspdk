"""Grating coupler data of the Run-50 standard components.

Measured performance of the rib gratings (components PDF p6 and p37); the strip
gratings share their grating section, so they use the same data:

* C-band (``SOI220_cband_TE_*_GC``): 5.5-6.5 dB coupling loss, 1 dB bandwidth
  > 35 nm, centre 1550-1580 nm.
* O-band (``SOI220_oband_TE_*_GC``): 5-6 dB coupling loss, 1 dB bandwidth > 30 nm,
  centre 1300-1330 nm.

Each is modelled as a Gaussian at the middle of the loss and centre ranges.
``sax.models.grating_coupler`` takes the 3 dB (full width at half maximum) power
bandwidth, so the 1 dB lower bound is converted: the 1 dB full width of a Gaussian
is 0.5764 times its FWHM, giving 35 / 0.5764 = 60.7 nm (C) and 30 / 0.5764 = 52.0 nm
(O). The elliptical grating is not a foundry component and reuses these data.
"""

from __future__ import annotations

import numpy as np
import sax
import sax.models as sm

nm = 1e-3

# 1 dB full width of a Gaussian in units of its 3 dB full width.
_ONE_DB_PER_FWHM = float(np.sqrt(np.log(10**0.1) / np.log(2)))

GRATINGS: dict[str, dict[str, float]] = {
    "cband": {"wl0": 1.565, "loss": 6.0, "bandwidth": 35 * nm / _ONE_DB_PER_FWHM},
    "oband": {"wl0": 1.315, "loss": 5.5, "bandwidth": 30 * nm / _ONE_DB_PER_FWHM},
}


def _grating_coupler_cband(*, wl: sax.FloatArrayLike = 1.55) -> sax.SDict:
    """C-band grating coupler model."""
    return sm.grating_coupler(wl=wl, **GRATINGS["cband"])


def _grating_coupler_oband(*, wl: sax.FloatArrayLike = 1.31) -> sax.SDict:
    """O-band grating coupler model."""
    return sm.grating_coupler(wl=wl, **GRATINGS["oband"])
