"""Circulax-native active circuit models for si220 cband.

These models implement multi-domain (electrical + optical) components
that cannot be expressed as SAX S-parameter functions. They are merged
into the PDK's models dict alongside SAX models.
"""

from __future__ import annotations

import jax.numpy as jnp
from circulax.components.base_component import (
    PhysicsReturn,
    Signals,
    States,
    component,
)


@component(ports=("o1", "o2", "l_e2", "r_e2"), states=("i_fwd",))
def ThermalPhaseShifter(
    signals: Signals,
    s: States,
    ohms_per_um: float = 0.375,
    eta_pi_per_W: float = 12.5,
    length: float = 320.0,
    loss_dBcm: float = 1.0,
    neff0: float = 2.4,
    wl: float = 1.55,
) -> PhysicsReturn:
    """Thermal phase shifter driven by ohmic dissipation.

    Resistance scales with heater length: ``R = ohms_per_um * length``.

    Uses a VCVS constraint (o2 = T * o1) with a matched input admittance
    (Y_in = 1/z0 = 1) instead of S→Y conversion. S→Y has a 1/(1-T²)
    denominator that blows up when |T| ≈ 1. A naive VCVS with shared
    branch current (o1: i_fwd) has S11 = (1-T)/(1+T), producing
    phase-dependent back-reflection; stamping Y_in = 1 gives S11 = 0.
    ``wl`` is in micrometres (SAX convention).

    @tags eo dc sweep, circulax simulation
    """
    if ohms_per_um <= 0 or length <= 0:
        raise ValueError(
            f"Both ohms_per_um ({ohms_per_um}) and length ({length}) must be strictly positive."
        )
    R = ohms_per_um * length
    v_bias = jnp.real(signals.l_e2 - signals.r_e2)
    p_diss = v_bias * v_bias / R
    dphi = jnp.pi * eta_pi_per_W * p_diss

    loss_val = loss_dBcm * (length / 10000.0)
    T_mag = 10.0 ** (-loss_val / 20.0)
    phi = 2.0 * jnp.pi * (neff0 * (length / wl)) + dphi
    T = T_mag * jnp.exp(-1j * phi)

    constraint = signals.o2 - T * signals.o1
    i_elec = v_bias / R
    return {
        "o1": signals.o1,
        "o2": -s.i_fwd,
        "l_e2": i_elec,
        "r_e2": -i_elec,
        "i_fwd": constraint,
    }, {}


def get_active_models() -> dict:
    """Return circulax active models keyed by component name."""
    return {
        "straight_heater_meander": ThermalPhaseShifter,
        "straight_heater_metal": ThermalPhaseShifter,
    }
