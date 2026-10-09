"""SAX models for Sparameter circuit simulations."""

from __future__ import annotations

import inspect
from collections.abc import Callable

import jax.numpy as jnp
import sax
from gdsfactory.typings import CrossSectionSpec
from numpy.typing import NDArray

from cspdk.si220.tech import get_band, is_rib

from . import couplers, foundry, gratings, oband, waveguides

sax.set_port_naming_strategy("optical")

FloatArray = NDArray[jnp.floating]
Float = float | FloatArray


################
# Band dispatch
################


def _foundry_models(band: str) -> dict[str, Callable[..., sax.SDict]]:
    """The foundry component models of ``band``, under every model name using them."""
    models = foundry.get_models(band)
    return {
        **models,
        "mmi1x2_strip": models["mmi1x2"],
        "mmi1x2_rib": models["mmi1x2"],
        "mmi2x2_strip": models["mmi2x2"],
        "mmi2x2_rib": models["mmi2x2"],
        "crossing_rib": models["crossing"],
    }


_CBAND: dict[str, Callable[..., sax.SDict]] = {
    **waveguides.get_models(),
    **couplers.get_models(),
    # Run-50 grating data (models/gratings.py); strip, rib and elliptical share it.
    "grating_coupler_rectangular_strip": gratings._grating_coupler_cband,
    "grating_coupler_rectangular_rib": gratings._grating_coupler_cband,
    "grating_coupler_rectangular": gratings._grating_coupler_cband,
    "grating_coupler_elliptical": gratings._grating_coupler_cband,
    **_foundry_models("cband"),
}
_OBAND = {**oband.get_models(), **_foundry_models("oband")}


def _dispatch(name: str, cross_section: CrossSectionSpec, **kwargs) -> sax.SDict:
    """Call the band implementation of model ``name`` selected by ``cross_section``.

    SAX (and circulax, which bakes the defaults into its components) calls a model
    with every parameter of its signature, filling unset ones with the signature
    defaults. The public models below therefore expose only the parameters a circuit
    sets, with concrete C-band defaults (``wl=1.55``). Band constants (``wl0``,
    ``neff``, ``ng``, grating and MMI data) are left out, so the O-band model keeps
    its own values instead of receiving C-band ones. The coupler's ``length``,
    ``offset`` and ``bend_radius`` default to ``None``: they mean "the cell geometry",
    which depends on band and type. With an O-band ``cross_section``, pass ``wl``
    explicitly.
    """
    cross_section = cross_section or "strip_cband"
    target = _CBAND[name]
    if get_band(cross_section) == "oband" and name in _OBAND:
        target = _OBAND[name]
    signature = inspect.signature(target)
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    call_kwargs = {
        key: value
        for key, value in kwargs.items()
        if value is not None and (accepts_kwargs or key in signature.parameters)
    }
    if "cross_section" in signature.parameters:
        # A model named *_rib / *_strip is that geometry whatever the cross-section says.
        if name.endswith(("_rib", "_strip")):
            call_kwargs["cross_section"] = name.rsplit("_", 1)[1]
        else:
            call_kwargs["cross_section"] = "rib" if is_rib(cross_section) else "strip"
    return target(**call_kwargs)


################
# Models
################
# One public ``def`` per model, here and nowhere else: GDSFactory+ finds models by
# scanning the source of this package for public functions returning ``sax.SDict``,
# so the band implementations above and in the submodules stay private.


def straight_strip(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss_dB_cm: float = 3.0,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """Straight strip waveguide model."""
    return _dispatch("straight_strip", **locals())


def straight_rib(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss_dB_cm: float = 3.0,
    cross_section: CrossSectionSpec = "rib_cband",
) -> sax.SDict:
    """Straight rib waveguide model."""
    return _dispatch("straight_rib", **locals())


def straight(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss_dB_cm: float = 3.0,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """Straight waveguide model."""
    return _dispatch("straight", **locals())


def wire_corner(
    *,
    wl: Float = 1.55,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """Wire corner model."""
    return _dispatch("wire_corner", **locals())


def bend_s(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss_dB_cm: float = 3.0,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """Bend S model."""
    return _dispatch("bend_s", **locals())


def bend_euler(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss_dB_cm: float = 3,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """Euler bend model."""
    return _dispatch("bend_euler", **locals())


def bend_euler_strip(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss_dB_cm: float = 3,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """Euler bend strip model."""
    return _dispatch("bend_euler_strip", **locals())


def bend_euler_rib(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss_dB_cm: float = 3,
    cross_section: CrossSectionSpec = "rib_cband",
) -> sax.SDict:
    """Euler bend rib model."""
    return _dispatch("bend_euler_rib", **locals())


def taper(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss_dB_cm: float = 0.0,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """Taper model."""
    return _dispatch("taper", **locals())


def taper_rib(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss_dB_cm: float = 0.0,
    cross_section: CrossSectionSpec = "rib_cband",
) -> sax.SDict:
    """Taper rib model."""
    return _dispatch("taper_rib", **locals())


def taper_strip_to_ridge(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss_dB_cm: float = 0.0,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """Taper strip to ridge model."""
    return _dispatch("taper_strip_to_ridge", **locals())


def trans_rib10(
    *,
    wl: Float = 1.55,
    loss_dB_cm: float = 0.0,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """Taper strip to ridge 10um model."""
    return _dispatch("trans_rib10", **locals())


def trans_rib20(
    *,
    wl: Float = 1.55,
    loss_dB_cm: float = 0.0,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """Taper strip to ridge 20um model."""
    return _dispatch("trans_rib20", **locals())


def trans_rib50(
    *,
    wl: Float = 1.55,
    loss_dB_cm: float = 0.0,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """Taper strip to ridge 50um model."""
    return _dispatch("trans_rib50", **locals())


def directional_coupler_no_phase(
    *,
    wl: float = 1.55,
    coupler_length: float = 10.0,
    gap: float = 0.5,
    offset: float = 20,
    bend_radius: float = 25,
    width: float = 1.0,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """Directional coupler coupling-region model (no propagation phase)."""
    return _dispatch("directional_coupler_no_phase", **locals())


def directional_coupler(
    *,
    wl: float = 1.55,
    length: float | None = None,
    gap: float = 0.27,
    offset: float | None = None,
    bend_radius: float | None = None,
    width: float = 1.0,
    with_euler: bool = False,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """Directional coupler model."""
    return _dispatch("directional_coupler", **locals())


def coupler_ring_coupling_area(
    *,
    wl: float = 1.55,
    gap: float = 0.1,
    radius: float = 5.0,
    length_x: float = 1.0,
    loss_dB: float = 0.0,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """Ring coupler coupling-region model."""
    return _dispatch("coupler_ring_coupling_area", **locals())


def coupler_ring(
    *,
    wl: float = 1.55,
    gap: float = 0.1,
    radius: float = 40.0,
    length_x: float = 1.0,
    p: float = 0,
    wl0: float = 0,
    loss_dB: float = 0.0,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """Ring coupler model."""
    return _dispatch("coupler_ring", **locals())


def coupler_strip(
    *,
    wl: float = 1.55,
    length: float | None = None,
    gap: float = 0.27,
    offset: float | None = None,
    bend_radius: float | None = None,
    width: float = 1.0,
    with_euler: bool = False,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """Directional coupler model."""
    return _dispatch("coupler_strip", **locals())


def coupler_rib(
    *,
    wl: float = 1.55,
    length: float | None = None,
    gap: float = 0.27,
    offset: float | None = None,
    bend_radius: float | None = None,
    width: float = 1.0,
    with_euler: bool = False,
    cross_section: CrossSectionSpec = "rib_cband",
) -> sax.SDict:
    """Directional coupler model."""
    return _dispatch("coupler_rib", **locals())


def coupler(
    *,
    wl: float = 1.55,
    length: float | None = None,
    gap: float = 0.27,
    offset: float | None = None,
    bend_radius: float | None = None,
    width: float = 1.0,
    with_euler: bool = False,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """Directional coupler model."""
    return _dispatch("coupler", **locals())


def mmi1x2_strip(
    *,
    wl: Float = 1.55,
    loss_dB: Float | None = None,
    fwhm: Float = 0.2,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """MMI 1x2 strip model.

    ``loss_dB`` is the excess loss; ``None`` uses the foundry measurement.
    """
    return _dispatch("mmi1x2_strip", **locals())


def mmi1x2_rib(
    *,
    wl: Float = 1.55,
    loss_dB: Float | None = None,
    fwhm: Float = 0.2,
    cross_section: CrossSectionSpec = "rib_cband",
) -> sax.SDict:
    """MMI 1x2 rib model.

    ``loss_dB`` is the excess loss; ``None`` uses the foundry measurement.
    """
    return _dispatch("mmi1x2_rib", **locals())


def mmi1x2(
    *,
    wl: Float = 1.55,
    loss_dB: Float | None = None,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """MMI 1x2 model.

    ``loss_dB`` is the excess loss; ``None`` uses the foundry measurement.
    """
    return _dispatch("mmi1x2", **locals())


def mmi2x2_strip(
    *,
    wl: Float = 1.55,
    loss_dB: Float | None = None,
    fwhm: Float = 0.2,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """MMI 2x2 strip model.

    ``loss_dB`` is the excess loss; ``None`` uses the foundry measurement.
    """
    return _dispatch("mmi2x2_strip", **locals())


def mmi2x2_rib(
    *,
    wl: Float = 1.55,
    loss_dB: Float | None = None,
    fwhm: Float = 0.2,
    cross_section: CrossSectionSpec = "rib_cband",
) -> sax.SDict:
    """MMI 2x2 rib model.

    ``loss_dB`` is the excess loss; ``None`` uses the foundry measurement.
    """
    return _dispatch("mmi2x2_rib", **locals())


def mmi2x2(
    *,
    wl: Float = 1.55,
    loss_dB: Float | None = None,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """MMI 2x2 model.

    ``loss_dB`` is the excess loss; ``None`` uses the foundry measurement.
    """
    return _dispatch("mmi2x2", **locals())


def grating_coupler_rectangular_strip(
    *,
    wl: Float = 1.55,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """Grating coupler rectangular strip model."""
    return _dispatch("grating_coupler_rectangular_strip", **locals())


def grating_coupler_rectangular_rib(
    *,
    wl: Float = 1.55,
    cross_section: CrossSectionSpec = "rib_cband",
) -> sax.SDict:
    """Grating coupler rectangular rib model."""
    return _dispatch("grating_coupler_rectangular_rib", **locals())


def grating_coupler_rectangular(
    *,
    wl: Float = 1.55,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """Grating coupler rectangular model."""
    return _dispatch("grating_coupler_rectangular", **locals())


def grating_coupler_elliptical(
    *,
    wl: Float = 1.55,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """Grating coupler elliptical model."""
    return _dispatch("grating_coupler_elliptical", **locals())


# Run-50 foundry cells. Each is the rectangular grating with its cross-section; the
# O-band ones default to the O-band centre wavelength.


def SOI220_cband_TE_strip_GC(
    *,
    wl: Float = 1.55,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """Run-50 C-band strip grating coupler model."""
    return _dispatch("grating_coupler_rectangular", **locals())


def SOI220_cband_TE_rib_GC(
    *,
    wl: Float = 1.55,
    cross_section: CrossSectionSpec = "rib_cband",
) -> sax.SDict:
    """Run-50 C-band rib grating coupler model."""
    return _dispatch("grating_coupler_rectangular", **locals())


def SOI220_oband_TE_strip_GC(
    *,
    wl: Float = 1.31,
    cross_section: CrossSectionSpec = "strip_oband",
) -> sax.SDict:
    """Run-50 O-band strip grating coupler model."""
    return _dispatch("grating_coupler_rectangular", **locals())


def SOI220_oband_TE_rib_GC(
    *,
    wl: Float = 1.31,
    cross_section: CrossSectionSpec = "rib_oband",
) -> sax.SDict:
    """Run-50 O-band rib grating coupler model."""
    return _dispatch("grating_coupler_rectangular", **locals())


def straight_heater_metal(
    *,
    wl: Float = 1.55,
    voltage: float = 0.0,
    vpi: float = 1.0,
    length: float = 10.0,
    loss_dB_cm: float = 3.0,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """Thermo-optic phase shifter on a strip waveguide."""
    return _dispatch("straight_heater_metal", **locals())


def straight_heater_meander(
    *,
    wl: Float = 1.55,
    voltage: float = 0.0,
    vpi: float = 1.0,
    length: float = 10.0,
    loss_dB_cm: float = 3.0,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """Meander thermo-optic phase shifter (heated length only)."""
    return _dispatch("straight_heater_meander", **locals())


def crossing_rib(
    *,
    wl: Float = 1.55,
    cross_section: CrossSectionSpec = "rib_cband",
) -> sax.SDict:
    """Crossing rib model (deprecated cell ``crossing_rib``; use ``crossing``)."""
    return _dispatch("crossing_rib", **locals())


def crossing(
    *,
    wl: Float = 1.55,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """Crossing model with the foundry's measured insertion loss."""
    return _dispatch("crossing", **locals())


def bend_circular(
    *,
    wl: Float = 1.55,
    radius: float | None = None,
    angle: float = 90.0,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """Circular bend model with the foundry's measured loss per 90 degrees.

    ``radius=None`` is the cross-section radius, like the cell.
    """
    return _dispatch("bend_circular", **locals())


def taper_mm2sm(
    *,
    wl: Float = 1.55,
    length: float = 450.0,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """Multimode-to-single-mode taper model with the foundry's measured loss."""
    return _dispatch("taper_mm2sm", **locals())


def taper_rib_to_strip(
    *,
    wl: Float = 1.55,
    length: float = 50.0,
    length_rib: float = 5.0,
    length_strip: float = 5.0,
    cross_section: CrossSectionSpec = "rib_cband",
) -> sax.SDict:
    """Rib-to-strip taper model with the foundry's measured loss."""
    return _dispatch("taper_rib_to_strip", **locals())


def heater(
    *,
    wl: Float = 1.55,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """Heater filament with pads: an ideal electrical connection between e1 and e2."""
    return _dispatch("heater", **locals())


def mzi_heater_loop(
    *,
    wl: Float = 1.55,
    delta_length: float = 160.0,
    length_x: float = 616.3,
    length_y: float = 35.0,
    radius: float = 30.0,
    voltage: float = 0.0,
    vpi: float = 1.0,
    loss_dB_cm: float = 3.0,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """MZI with looped arms and a heater on the upper arm."""
    return _dispatch("mzi_heater_loop", **locals())


################
# Cornerstone Run-50 template cells
################
# One model per foundry cell, named like it, with the cell's band and type fixed.


def SOI220_cband_TE_strip_1x2_MMI(
    *,
    wl: Float = 1.55,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """Cornerstone C-band strip 1x2 MMI."""
    return _dispatch("mmi1x2", **locals())


def SOI220_cband_TE_strip_2x2_MMI(
    *,
    wl: Float = 1.55,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """Cornerstone C-band strip 2x2 MMI."""
    return _dispatch("mmi2x2", **locals())


def SOI220_cband_TE_strip_crossing(
    *,
    wl: Float = 1.55,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """Cornerstone C-band strip crossing."""
    return _dispatch("crossing", **locals())


def SOI220_cband_TE_rib_1x2_MMI(
    *,
    wl: Float = 1.55,
    cross_section: CrossSectionSpec = "rib_cband",
) -> sax.SDict:
    """Cornerstone C-band rib 1x2 MMI."""
    return _dispatch("mmi1x2", **locals())


def SOI220_cband_TE_rib_2x2_MMI(
    *,
    wl: Float = 1.55,
    cross_section: CrossSectionSpec = "rib_cband",
) -> sax.SDict:
    """Cornerstone C-band rib 2x2 MMI."""
    return _dispatch("mmi2x2", **locals())


def SOI220_cband_TE_rib_crossing(
    *,
    wl: Float = 1.55,
    cross_section: CrossSectionSpec = "rib_cband",
) -> sax.SDict:
    """Cornerstone C-band rib crossing."""
    return _dispatch("crossing", **locals())


def SOI220_oband_TE_strip_1x2_MMI(
    *,
    wl: Float = 1.31,
    cross_section: CrossSectionSpec = "strip_oband",
) -> sax.SDict:
    """Cornerstone O-band strip 1x2 MMI."""
    return _dispatch("mmi1x2", **locals())


def SOI220_oband_TE_strip_2x2_MMI(
    *,
    wl: Float = 1.31,
    cross_section: CrossSectionSpec = "strip_oband",
) -> sax.SDict:
    """Cornerstone O-band strip 2x2 MMI."""
    return _dispatch("mmi2x2", **locals())


def SOI220_oband_TE_strip_crossing(
    *,
    wl: Float = 1.31,
    cross_section: CrossSectionSpec = "strip_oband",
) -> sax.SDict:
    """Cornerstone O-band strip crossing."""
    return _dispatch("crossing", **locals())


def SOI220_oband_TE_rib_1x2_MMI(
    *,
    wl: Float = 1.31,
    cross_section: CrossSectionSpec = "rib_oband",
) -> sax.SDict:
    """Cornerstone O-band rib 1x2 MMI."""
    return _dispatch("mmi1x2", **locals())


def SOI220_oband_TE_rib_2x2_MMI(
    *,
    wl: Float = 1.31,
    cross_section: CrossSectionSpec = "rib_oband",
) -> sax.SDict:
    """Cornerstone O-band rib 2x2 MMI."""
    return _dispatch("mmi2x2", **locals())


def SOI220_oband_TE_rib_crossing(
    *,
    wl: Float = 1.31,
    cross_section: CrossSectionSpec = "rib_oband",
) -> sax.SDict:
    """Cornerstone O-band rib crossing."""
    return _dispatch("crossing", **locals())


################
# Run 50 fixed cells
################
# The foundry measured the Y-splitters by cut-back (CORNERSTONE 50th call standard
# components, p. 17 C band and p. 45 O band): the transmission through one output of
# a cascade of splitters falls by 4.0123 dB per splitter in the C band for 450 nm
# waveguides (3.7101 dB for 400 nm) and 4.5006 dB in the O band. That includes the
# 3.0103 dB of an ideal 50/50 split, so the excess loss is 1.0020 dB (C band, the
# 450 nm template cell) and 1.4903 dB (O band). The models are ideal, wavelength-flat
# 50/50 splitters with that excess loss.


def _ysplitter(*, wl: Float, loss_dB: Float) -> sax.SDict:
    """Lossy ideal 50/50 splitter, flat in wavelength."""
    one = jnp.ones_like(jnp.asarray(wl, dtype=float))
    t = jnp.sqrt(0.5 * 10 ** (-jnp.asarray(loss_dB) / 10)) * one
    return sax.reciprocal({("o1", "o2"): t, ("o1", "o3"): t})


def SOI220_cband_TE_strip_1x2_Ysplitter(
    *,
    wl: Float = 1.55,
    loss_dB: Float = 1.002,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """C-band 1x2 Y-splitter of the Run 50 template (450 nm strip).

    Args:
        wl: wavelength in um.
        loss_dB: excess loss in dB on top of the 3 dB split (measured 1.0 dB; 0.7 dB
            for 400 nm waveguides).
        cross_section: unused; the splitter geometry is fixed.
    """
    return _ysplitter(wl=wl, loss_dB=loss_dB)


def SOI220_oband_TE_strip_1x2_Ysplitter(
    *,
    wl: Float = 1.31,
    loss_dB: Float = 1.4903,
    cross_section: CrossSectionSpec = "strip_oband",
) -> sax.SDict:
    """O-band 1x2 Y-splitter of the Run 50 template (400 nm strip).

    Args:
        wl: wavelength in um.
        loss_dB: excess loss in dB on top of the 3 dB split (measured 1.5 dB).
        cross_section: unused; the splitter geometry is fixed.
    """
    return _ysplitter(wl=wl, loss_dB=loss_dB)


def SOI220_cband_TE_strip_90deg_bend(
    *,
    wl: Float = 1.55,
    radius: float = 5.0,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """Cornerstone C-band strip 90 degree bend."""
    return _dispatch("bend_circular", **locals())


def SOI220_oband_TE_strip_90deg_bend(
    *,
    wl: Float = 1.31,
    radius: float = 5.0,
    cross_section: CrossSectionSpec = "strip_oband",
) -> sax.SDict:
    """Cornerstone O-band strip 90 degree bend."""
    return _dispatch("bend_circular", **locals())


def SOI220_cband_TE_rib_90deg_bend(
    *,
    wl: Float = 1.55,
    radius: float = 25.0,
    cross_section: CrossSectionSpec = "rib_cband",
) -> sax.SDict:
    """Cornerstone C-band rib 90 degree bend."""
    return _dispatch("bend_circular", **locals())


def SOI220_cband_TE_strip_MM2SM_taper(
    *,
    wl: Float = 1.55,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """Cornerstone C-band strip multimode-to-single-mode taper."""
    return _dispatch("taper_mm2sm", **locals())


def SOI220_cband_TE_rib_MM2SM_taper(
    *,
    wl: Float = 1.55,
    cross_section: CrossSectionSpec = "rib_cband",
) -> sax.SDict:
    """Cornerstone C-band rib multimode-to-single-mode taper."""
    return _dispatch("taper_mm2sm", **locals())


def SOI220_cband_TE_rib2strip_taper(
    *,
    wl: Float = 1.55,
    cross_section: CrossSectionSpec = "rib_cband",
) -> sax.SDict:
    """Cornerstone C-band rib rib-to-strip taper."""
    return _dispatch("taper_rib_to_strip", **locals())


def SOI220_cband_TE_strip_MZI(
    *,
    wl: Float = 1.55,
    length_x: float = 616.3,
    voltage: float = 0.0,
    vpi: float = 1.0,
    loss_dB_cm: float = 3.0,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """Cornerstone C-band strip MZI with heater."""
    return _dispatch("mzi_heater_loop", **locals())


def SOI220_cband_TE_rib_MZI(
    *,
    wl: Float = 1.55,
    length_x: float = 649.5,
    voltage: float = 0.0,
    vpi: float = 1.0,
    loss_dB_cm: float = 3.0,
    cross_section: CrossSectionSpec = "rib_cband",
) -> sax.SDict:
    """Cornerstone C-band rib MZI with heater."""
    return _dispatch("mzi_heater_loop", **locals())


def Heater(
    *,
    wl: Float = 1.55,
    cross_section: CrossSectionSpec = "strip_cband",
) -> sax.SDict:
    """Cornerstone heater."""
    return _dispatch("heater", **locals())


def get_models() -> dict[str, Callable[..., sax.SDict]]:
    """Return the public models, which dispatch to the C- or O-band behaviour."""
    return {
        "straight_strip": straight_strip,
        "straight_rib": straight_rib,
        "straight": straight,
        "wire_corner": wire_corner,
        "bend_s": bend_s,
        "bend_euler": bend_euler,
        "bend_euler_strip": bend_euler_strip,
        "bend_euler_rib": bend_euler_rib,
        "taper": taper,
        "taper_rib": taper_rib,
        "taper_strip_to_ridge": taper_strip_to_ridge,
        "trans_rib10": trans_rib10,
        "trans_rib20": trans_rib20,
        "trans_rib50": trans_rib50,
        "directional_coupler_no_phase": directional_coupler_no_phase,
        "directional_coupler": directional_coupler,
        "coupler_ring_coupling_area": coupler_ring_coupling_area,
        "coupler_ring": coupler_ring,
        "coupler_strip": coupler_strip,
        "coupler_rib": coupler_rib,
        "coupler": coupler,
        "mmi1x2_strip": mmi1x2_strip,
        "mmi1x2_rib": mmi1x2_rib,
        "mmi1x2": mmi1x2,
        "mmi2x2_strip": mmi2x2_strip,
        "mmi2x2_rib": mmi2x2_rib,
        "mmi2x2": mmi2x2,
        "grating_coupler_rectangular_strip": grating_coupler_rectangular_strip,
        "grating_coupler_rectangular_rib": grating_coupler_rectangular_rib,
        "grating_coupler_rectangular": grating_coupler_rectangular,
        "grating_coupler_elliptical": grating_coupler_elliptical,
        "SOI220_cband_TE_strip_GC": SOI220_cband_TE_strip_GC,
        "SOI220_cband_TE_rib_GC": SOI220_cband_TE_rib_GC,
        "SOI220_oband_TE_strip_GC": SOI220_oband_TE_strip_GC,
        "SOI220_oband_TE_rib_GC": SOI220_oband_TE_rib_GC,
        "straight_heater_metal": straight_heater_metal,
        "straight_heater_meander": straight_heater_meander,
        "crossing_rib": crossing_rib,
        "crossing": crossing,
        "SOI220_cband_TE_strip_1x2_Ysplitter": SOI220_cband_TE_strip_1x2_Ysplitter,
        "SOI220_oband_TE_strip_1x2_Ysplitter": SOI220_oband_TE_strip_1x2_Ysplitter,
        "bend_circular": bend_circular,
        "taper_mm2sm": taper_mm2sm,
        "taper_rib_to_strip": taper_rib_to_strip,
        "heater": heater,
        "mzi_heater_loop": mzi_heater_loop,
        "SOI220_cband_TE_strip_1x2_MMI": SOI220_cband_TE_strip_1x2_MMI,
        "SOI220_cband_TE_strip_2x2_MMI": SOI220_cband_TE_strip_2x2_MMI,
        "SOI220_cband_TE_strip_crossing": SOI220_cband_TE_strip_crossing,
        "SOI220_cband_TE_rib_1x2_MMI": SOI220_cband_TE_rib_1x2_MMI,
        "SOI220_cband_TE_rib_2x2_MMI": SOI220_cband_TE_rib_2x2_MMI,
        "SOI220_cband_TE_rib_crossing": SOI220_cband_TE_rib_crossing,
        "SOI220_oband_TE_strip_1x2_MMI": SOI220_oband_TE_strip_1x2_MMI,
        "SOI220_oband_TE_strip_2x2_MMI": SOI220_oband_TE_strip_2x2_MMI,
        "SOI220_oband_TE_strip_crossing": SOI220_oband_TE_strip_crossing,
        "SOI220_oband_TE_rib_1x2_MMI": SOI220_oband_TE_rib_1x2_MMI,
        "SOI220_oband_TE_rib_2x2_MMI": SOI220_oband_TE_rib_2x2_MMI,
        "SOI220_oband_TE_rib_crossing": SOI220_oband_TE_rib_crossing,
        "SOI220_cband_TE_strip_90deg_bend": SOI220_cband_TE_strip_90deg_bend,
        "SOI220_oband_TE_strip_90deg_bend": SOI220_oband_TE_strip_90deg_bend,
        "SOI220_cband_TE_rib_90deg_bend": SOI220_cband_TE_rib_90deg_bend,
        "SOI220_cband_TE_strip_MM2SM_taper": SOI220_cband_TE_strip_MM2SM_taper,
        "SOI220_cband_TE_rib_MM2SM_taper": SOI220_cband_TE_rib_MM2SM_taper,
        "SOI220_cband_TE_rib2strip_taper": SOI220_cband_TE_rib2strip_taper,
        "SOI220_cband_TE_strip_MZI": SOI220_cband_TE_strip_MZI,
        "SOI220_cband_TE_rib_MZI": SOI220_cband_TE_rib_MZI,
        "Heater": Heater,
    }
