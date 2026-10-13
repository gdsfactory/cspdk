"""Directional Couplers."""

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import sax
import xarray as xr
from jaxtyping import ArrayLike

from cspdk.si220.tech import TECH

from . import oband
from .waveguides import _bend_euler, _straight_rib, _straight_strip

if TYPE_CHECKING:
    SDict = sax.SDict
else:
    SDict = "sax.SDict"

CWD = Path(__file__).resolve().parent


def _load(name: str) -> xr.DataArray:
    with jax.ensure_compile_time_eval():
        return (
            xr.open_dataarray(CWD / f"{name}.nc")
            .load()
            .expand_dims({"kappa": ["kappa"]}, -1)
        )


def _load_tables(prefix: str) -> dict[tuple[str, str], xr.DataArray]:
    """Load the {(band, cross_section): table} coupling tables for one coupler type."""
    band_suffixes = {"cband": "", "oband": "_oband"}
    return {
        (band, xs): _load(f"{prefix}_{xs}{suffix}")
        for band, suffix in band_suffixes.items()
        for xs in ("strip", "rib")
    }


# All coupling tables are generated with samples/coupler_cmt.py
_XARR_DC = _load_tables("directional_coupler")
_XARR_RACETRACK = _load_tables("coupler_racetrack")
xarr_dc_strip = _XARR_DC["cband", "strip"]
xarr_racetrack_strip = _XARR_RACETRACK["cband", "strip"]
_STRAIGHT = {
    ("cband", "strip"): _straight_strip,
    ("cband", "rib"): _straight_rib,
    ("oband", "strip"): oband._straight_strip,
    ("oband", "rib"): oband._straight_rib,
}
_BEND = {"cband": _bend_euler, "oband": oband._bend_euler}
_WIDTH = {"cband": TECH.width_cband, "oband": TECH.width_oband}


def _interpolate_kappa(xarr: xr.DataArray, **kwargs: ArrayLike) -> jnp.ndarray:
    # Extract interpolation dims from xarray
    dims = [d for d in xarr.coords if d != "kappa"]

    # Ensure required args are provided
    missing = [d for d in dims if d not in kwargs]
    if missing:
        raise ValueError(f"Missing required interpolation inputs: {missing}")

    # Broadcast all input arrays
    arrays = [jnp.asarray(kwargs[dim]) for dim in dims]
    broadcasted = jnp.broadcast_arrays(*arrays)
    shape = broadcasted[0].shape

    # Prepare kwargs for interpolation
    interp_args = {dim: arr.ravel() for dim, arr in zip(dims, broadcasted, strict=True)}

    # Interpolate
    result = sax.interpolate_xarray(xarr, **interp_args)["kappa"]
    return result.reshape(shape)


def _sbend_geometry(
    gap: float, cross_section: str, band: str = "cband"
) -> tuple[float, float]:
    """Return (lateral offset per arm, equivalent circular radius) of the coupler cell.

    The cell's S-bends span dx along the coupler and bring the ports to a pitch of dy.
    """
    if cross_section == "rib":
        dx, dy = TECH.dx_coupler_rib, TECH.dy_coupler_rib
    else:
        dx, dy = TECH.dx_coupler, TECH.dy_coupler
    offset = (dy - gap - _WIDTH[band]) / 2
    theta = 2 * jnp.arctan(offset / dx)
    return offset, dx / (2 * jnp.sin(theta))


def _directional_coupler_no_phase(
    *,
    wl: float = 1.55,
    coupler_length: float = 10.0,
    gap: float = 0.5,
    offset: float = 20,
    bend_radius: float = 25,
    width: float = 1.0,
    cross_section: str = "strip",
    band: str = "cband",
) -> SDict:
    r"""Directional coupler coupling-region model (no propagation phase).

    Semi-analytical model for directional couplers developed by GDSFactory.
    Use at your own risk.

    Args:
        wl: wavelength [µm]; between 1.5 and 1.6 µm.
        gap: gap between the two waveguides [µm]; between 0.05 and 1.5 µm.
        coupler_length: length of the ring coupler [µm]; between 0 and 100 µm.
        offset: offset between the two waveguides [µm]; between 5 and 100 µm.
        bend_radius: bend radius of the ring coupler [µm]; between 5 and 100 µm.
        width: width of the waveguides [µm]; between 0.1 and 10 µm.
        cross_section: cross section of the waveguide.
        band: "cband" or "oband" coupling table.
    """
    kappa = _interpolate_kappa(
        xarr=_XARR_DC[band, cross_section],
        wavelength=wl,
        radius=bend_radius,
        gap=gap,
        length_x=coupler_length,
        v_offset=offset,
    )

    tau = jnp.sqrt(1 - jnp.array(kappa) ** 2)

    return sax.reciprocal(
        {
            ("o1", "o4"): tau,
            ("o1", "o3"): 1j * kappa,
            ("o2", "o4"): 1j * kappa,
            ("o2", "o3"): tau,
        }
    )


def _directional_coupler(
    *,
    wl: float = 1.55,
    length: float | None = None,
    gap: float = TECH.gap_strip,
    offset: float | None = None,
    bend_radius: float | None = None,
    width: float = 1.0,
    with_euler: bool = False,
    cross_section: str = "strip",
    band: str = "cband",
) -> SDict:
    r"""Directional coupler model.

    Semi-analytical model for directional couplers developed by GDSFactory.
    Use at your own risk. The coupling tables come from coupled-mode theory on
    supermode solves (samples/coupler_cmt.py) and are not foundry validated.

    Args:
        wl: wavelength [µm]
        gap: gap between the two waveguides [µm]
        length: length of the coupling region [µm]. Defaults to the cell default.
        offset: lateral S-bend offset per arm [µm]. Defaults to the cell geometry.
        bend_radius: equivalent S-bend radius [µm]. Defaults to the cell geometry.
        with_euler: if True, the directional coupler will have an Euler bend.
        width: width of the waveguides [µm].
        cross_section: cross section of the waveguide.
        band: "cband" or "oband" coupling table and waveguide models.
    """
    if with_euler:
        raise NotImplementedError("Euler bend is not implemented yet")
    cell_offset, cell_radius = _sbend_geometry(gap, cross_section, band)
    offset = cell_offset if offset is None else offset
    bend_radius = cell_radius if bend_radius is None else bend_radius

    if length is None:
        if cross_section == "rib":
            length = TECH.length_coupler_rib
        elif band == "oband":
            length = TECH.length_coupler_oband
        else:
            length = TECH.length_coupler
    # Each arm (half the coupling length plus an S-bend) is a straight before or after
    # the coupling region, so every path picks up one input and one output arm.
    sbend_length = 2 * bend_radius * jnp.arccos(1 - offset / 2 / bend_radius)
    arm = _STRAIGHT[band, cross_section](wl=wl, length=length / 2 + sbend_length)
    arms = arm["o1", "o2"] ** 2
    dc = _directional_coupler_no_phase(
        wl=wl,
        coupler_length=length,
        gap=gap,
        offset=offset,
        bend_radius=bend_radius,
        cross_section=cross_section,
        band=band,
    )
    return sax.reciprocal(
        {
            ("o1", "o4"): dc["o1", "o4"] * arms,
            ("o1", "o3"): dc["o1", "o3"] * arms,
            ("o2", "o4"): dc["o2", "o4"] * arms,
            ("o2", "o3"): dc["o2", "o3"] * arms,
        }
    )


_coupler_strip = _directional_coupler
_coupler_rib = _directional_coupler
_coupler = _directional_coupler


def _coupler_ring_coupling_area(
    *,
    wl: float = 1.55,
    gap: float = 0.1,
    radius: float = 5.0,
    length_x: float = 1.0,
    loss_dB: float = 0.0,
    cross_section: str = "strip",
    band: str = "cband",
) -> SDict:
    r"""Ring coupler coupling-region model.

    This is a semi-analytical model developed by GDSFactory.
    GDSFactory does not guarantee the accuracy of this model.
    The coupling tables come from coupled-mode theory on supermode solves
    (samples/coupler_cmt.py) and have not been validated by the foundry.
    Please use at your own discretion.

    Args:
        wl: wavelength [µm]; 1.5-1.6 µm (cband) or 1.26-1.36 µm (oband).
        gap: gap between the two waveguides [µm]; between 0.05 and 1.05 µm.
        radius: radius of the ring [µm]; between 5 and 245 µm.
        length_x: length of the ring coupler [µm]; between 0 and 28 µm.
        loss_dB: excess loss of the coupling region [dB].
        cross_section: cross section of the waveguide.
        band: "cband" or "oband" coupling table and waveguide models.
    """
    kappa = _interpolate_kappa(
        xarr=_XARR_RACETRACK[band, cross_section],
        wavelength=wl,
        gap=gap,
        radius=radius,
        length_x=length_x,
    )

    tau = jnp.sqrt(1 - jnp.array(kappa) ** 2)

    # propagation phase and loss through the straight coupling section
    t = _STRAIGHT[band, cross_section](wl=wl, length=length_x)["o1", "o2"]
    t = t * 10 ** (-loss_dB / 20)
    kappa = kappa * t
    tau = tau * t

    return sax.reciprocal(
        {
            ("o1", "o4"): tau,
            ("o1", "o3"): 1j * kappa,
            ("o2", "o4"): 1j * kappa,
            ("o2", "o3"): tau,
        }
    )


def _coupler_ring(  # this is not the complete model!!!!
    *,
    wl: float = 1.55,
    gap: float = 0.1,
    radius: float = 40.0,
    length_x: float = 1.0,
    p: float = 0,
    wl0: float = 0,  # this is not used in the model
    loss_dB: float = 0.0,
    cross_section: str = "strip",
    band: str = "cband",
) -> SDict:
    r"""Ring coupler model.

    This is a semi-analytical model developed by GDSFactory.
    GDSFactory does not guarantee the accuracy of this model.
    This model has not been validated by the foundry (see
    coupler_ring_coupling_area for the provenance of the coupling tables).
    Please use at your own discretion.

    Args:
        wl: wavelength [µm]; 1.5-1.6 µm (cband) or 1.26-1.36 µm (oband).
        gap: gap between the two waveguides [µm]; between 0.05 and 1.05 µm.
        radius: radius of the ring [µm]; between 5 and 245 µm.
        length_x: length of the ring coupler [µm]; between 0 and 28 µm.
        p: ignored; circular bends are assumed.
        wl0: center wavelength (um).
        loss_dB: excess loss of the coupling region [dB].
        cross_section: cross section of the waveguide.
        band: "cband" or "oband" coupling table and waveguide models.
    """
    # Quarter-circle bends sit on ports o2 and o3 only; o1 and o4 are the bus.
    quarter_circle = jnp.pi * radius / 2
    bend_sdict = _BEND[band](wl=wl, length=quarter_circle, cross_section=cross_section)
    bend = bend_sdict["o1", "o2"]
    c = _coupler_ring_coupling_area(
        wl=wl,
        gap=gap,
        radius=radius,
        length_x=length_x,
        loss_dB=loss_dB,
        cross_section=cross_section,
        band=band,
    )
    return sax.reciprocal(
        {
            ("o1", "o4"): c["o1", "o4"],
            ("o1", "o3"): c["o1", "o3"] * bend,
            ("o2", "o4"): bend * c["o2", "o4"],
            ("o2", "o3"): bend * c["o2", "o3"] * bend,
        }
    )


def get_models() -> dict[str, Callable[..., sax.SDict]]:
    """Return the C-band coupler models keyed by model name."""
    return {
        "directional_coupler_no_phase": _directional_coupler_no_phase,
        "directional_coupler": _directional_coupler,
        "coupler_ring_coupling_area": _coupler_ring_coupling_area,
        "coupler_ring": _coupler_ring,
        "coupler_strip": _coupler_strip,
        "coupler_rib": _coupler_rib,
        "coupler": _coupler,
    }
