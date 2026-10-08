"""SAX models for Sparameter circuit simulations."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from functools import wraps

import jax.numpy as jnp
import sax
import sax.models as sm
from numpy.typing import NDArray

from cspdk.si220.tech import get_band, is_rib

from .couplers import *
from .waveguides import *

sax.set_port_naming_strategy("optical")

nm = 1e-3

FloatArray = NDArray[jnp.floating]
Float = float | FloatArray


################
# MMIs
################


def mmi1x2_strip(
    *,
    wl: Float = 1.55,
    wl0: float = 1.55,
    loss_dB: Float = 0.3,
    fwhm: Float = 0.2,
) -> sax.SDict:
    """MMI 1x2 strip model."""
    return sm.mmi1x2(
        wl=wl,
        wl0=wl0,
        fwhm=fwhm,
        loss_dB=loss_dB,
    )


def mmi1x2_rib(
    *,
    wl: Float = 1.55,
    wl0: float = 1.55,
    loss_dB: Float = 0.3,
    fwhm: Float = 0.2,
) -> sax.SDict:
    """MMI 1x2 rib model."""
    return sm.mmi1x2(
        wl=wl,
        wl0=wl0,
        fwhm=fwhm,
        loss_dB=loss_dB,
    )


def mmi1x2(
    wl: Float = 1.55,
    loss_dB: Float = 0.3,
    cross_section="strip",
) -> sax.SDict:
    """MMI 1x2 model."""
    wl = jnp.asarray(wl)  # type: ignore
    fs = {
        "strip": mmi1x2_strip,
        "rib": mmi1x2_rib,
    }
    f = fs[cross_section]
    return f(
        wl=wl,
        loss_dB=loss_dB,
    )


def mmi2x2_strip(
    *,
    wl: Float = 1.55,
    wl0: float = 1.55,
    loss_dB: Float = 0.3,
    fwhm: Float = 0.2,
) -> sax.SDict:
    """MMI 2x2 strip model."""
    return sm.mmi2x2(
        wl=wl,
        wl0=wl0,
        fwhm=fwhm,
        loss_dB=loss_dB,
    )


def mmi2x2_rib(
    *,
    wl: Float = 1.55,
    wl0: float = 1.55,
    loss_dB: Float = 0.3,
    fwhm: Float = 0.2,
) -> sax.SDict:
    """MMI 2x2 rib model."""
    return sm.mmi2x2(
        wl=wl,
        wl0=wl0,
        fwhm=fwhm,
        loss_dB=loss_dB,
    )


def mmi2x2(
    wl: Float = 1.55,
    loss_dB: Float = 0.3,
    cross_section="strip",
) -> sax.SDict:
    """MMI 2x2 model."""
    wl = jnp.asarray(wl)  # type: ignore
    fs = {
        "strip": mmi2x2_strip,
        "rib": mmi2x2_rib,
    }
    f = fs[cross_section]
    return f(
        wl=wl,
        loss_dB=loss_dB,
    )


##############################
# grating couplers Rectangular
##############################


def grating_coupler_rectangular_strip(
    *,
    wl: Float = 1.55,
) -> sax.SDict:
    """Grating coupler rectangular strip model."""
    return sm.grating_coupler(
        wl=wl,
        loss=6,
        bandwidth=35 * nm,
    )


def grating_coupler_rectangular_rib(
    *,
    wl: Float = 1.55,
) -> sax.SDict:
    """Grating coupler rectangular rib model."""
    return sm.grating_coupler(
        wl=wl,
        loss=6,
        bandwidth=35 * nm,
    )


def grating_coupler_rectangular(
    wl: Float = 1.55,
    cross_section="strip",
) -> sax.SDict:
    """Grating coupler rectangular model."""
    # TODO: take more grating_coupler_rectangular arguments into account
    wl = jnp.asarray(wl)  # type: ignore
    fs = {
        "strip": grating_coupler_rectangular_strip,
        "rib": grating_coupler_rectangular_rib,
    }
    f = fs[cross_section]
    return f(wl=wl)  # type: ignore


##############################
# grating couplers Elliptical
##############################


def grating_coupler_elliptical(
    *,
    wl: Float = 1.55,
) -> sax.SDict:
    """Grating coupler elliptical model."""
    return sm.grating_coupler(
        wl=wl,
        loss=6,
        bandwidth=35 * nm,
    )


################
# Imported
################


def heater() -> sax.SDict:
    """Heater model."""
    raise NotImplementedError("No model for 'heater'")


def straight_heater_metal(
    wl: Float | Sequence[float] = 1.55,
    neff: float = 2.34,
    voltage: float = 0,
    vpi: float = 1.0,  # Voltage required for π-phase shift
    length: float = 10,
    loss_dB_cm: sax.FloatArrayLike = 3.0,
) -> sax.SDict:
    """Returns simple phase shifter model.

    Args:
        wl: wavelength.
        neff: effective index.
        voltage: applied voltage.
        vpi: voltage required for a π-phase shift.
        length: length.
        loss_dB_cm: The Propagation loss in dB/cm.


    ```

     o1 =========== o2
    ```
    """
    wl = jnp.asarray(wl)  # type: ignore
    # Calculate additional phase shift due to applied voltage.
    deltaphi = (voltage / vpi) * jnp.pi
    phase = 2 * jnp.pi * neff * length / wl + deltaphi
    amplitude = jnp.asarray(10 ** (-1e-4 * loss_dB_cm * length / 20), dtype=complex)
    transmission = amplitude * jnp.exp(1j * phase)
    return sax.reciprocal(
        {
            ("o1", "o2"): transmission,
            ("l_e1", "r_e1"): 0.0,
            ("l_e2", "r_e2"): 0.0,
            ("l_e3", "r_e3"): 0.0,
            ("l_e4", "r_e4"): 0.0,
        }
    )


def crossing_rib(
    *,
    wl: Float = 1.55,
) -> sax.SDict:
    """Crossing rib model."""
    return sm.crossing_ideal(wl=wl)


def crossing(
    *,
    wl: Float = 1.55,
) -> sax.SDict:
    """Crossing model."""
    return sm.crossing_ideal(wl=wl)


################
# Models Dict
################


def _get_cband_models() -> dict[str, Callable[..., sax.SDict]]:
    """Return the cband models defined in this module."""
    models = {}
    for name, func in list(globals().items()):
        # Skip get_models itself and private functions
        if name.startswith("_") or name in {"get_models", "heater"}:
            continue
        if not callable(func):
            continue
        try:
            sig = inspect.signature(func)
        except (ValueError, TypeError):
            continue
        # Check for sax.SDict return type (case-insensitive)
        return_anno = str(sig.return_annotation)
        if "sdict" in return_anno.lower():
            models[name] = func
    return models


def _dispatch_model(
    cband_model: Callable[..., sax.SDict],
    oband_model: Callable[..., sax.SDict] | None,
) -> Callable[..., sax.SDict]:
    """Return a shared model that selects its implementation by cross-section.

    SAX calls a model with every parameter of its signature, filling unset ones with
    the signature defaults, and drops settings that are not in the signature. The
    wrapper therefore exposes the union of both bands' parameters plus
    ``cross_section``, all defaulting to ``None`` ("not set"), so the cross-section
    always reaches the dispatcher and the selected model applies its own defaults.
    """
    parameters = {}
    for band_model in (cband_model, oband_model):
        if band_model is None:
            continue
        for parameter in inspect.signature(band_model).parameters.values():
            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            parameters.setdefault(
                parameter.name,
                parameter.replace(kind=inspect.Parameter.KEYWORD_ONLY, default=None),
            )
    parameters["cross_section"] = inspect.Parameter(
        "cross_section", inspect.Parameter.KEYWORD_ONLY, default="strip_cband"
    )

    @wraps(cband_model)
    def model(*args, **kwargs) -> sax.SDict:
        cross_section = kwargs.pop("cross_section", None) or "strip_cband"
        target = (
            oband_model
            if get_band(cross_section) == "oband" and oband_model is not None
            else cband_model
        )
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
            call_kwargs["cross_section"] = "rib" if is_rib(cross_section) else "strip"
        return target(*args, **call_kwargs)

    model.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        list(parameters.values()), return_annotation=sax.SDict
    )
    return model


def _build_models() -> dict[str, Callable[..., sax.SDict]]:
    """Build the shared model dict dispatching to cband or oband behavior."""
    from cspdk.si220.models import oband

    cband_models = _get_cband_models()
    oband_models = oband.get_models()
    return {
        name: _dispatch_model(model, oband_models.get(name))
        for name, model in cband_models.items()
    }


# Build once, before any name can be referenced, and bind the dispatch wrappers
# into this module's namespace.  Cell SAX metadata references
# ``module="cspdk.si220.models"`` with ``qualname=<name>``; without this
# binding that resolves to the raw implementation imported from
# ``.waveguides``/``.couplers``, which only understands the model-level
# ``"strip"``/``"rib"`` vocabulary and rejects the PDK-level
# ``strip_cband``/``strip_oband`` names.
_MODELS = _build_models()
globals().update(_MODELS)


def get_models() -> dict[str, Callable[..., sax.SDict]]:
    """Return shared model names dispatching to cband or oband behavior."""
    return _MODELS
