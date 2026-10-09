"""Thermal MZI: DC current sweep showing complementary optical outputs.

Builds a Mach-Zehnder interferometer with thermal phase shifters
(straight_heater_meander) and sweeps heater current to show the
sinusoidal transfer function at the two output photodetectors.

Requires circulax for active circuit simulation.

@tags eo dc sweep, circulax simulation, mzi, thermal phase shifter
"""

from __future__ import annotations

import gdsfactory as gf
import numpy as np
from cspdk.si220 import cells
from matplotlib import pyplot as plt


@gf.cell
def thermal_mzi(
    length_heater: float = 320.0,
    dL: float = 0.0,
) -> gf.Component:
    """MZI with thermal phase shifters on both arms.

    Args:
        length_heater: length of the heater phase shifter in um.
        dL: path length difference between arms in um.
    """
    c = gf.Component()
    arm_length = dL / 2 if dL > 0 else 0.1

    sp = c << cells.mmi1x2()
    cp = c << cells.mmi2x2()

    b1 = c << cells.bend_euler()
    b1.connect("o1", sp.ports["o2"])

    sl = c << cells.straight(length=arm_length)
    sl.name = "sl"
    sl.connect("o1", b1.ports["o2"])

    h_top = c << cells.straight_heater_meander(length=length_heater)
    h_top.connect("o1", sl.ports["o2"])

    b2 = c << cells.bend_euler()
    b2.connect("o2", h_top.ports["o2"])

    sr = c << cells.straight(length=arm_length)
    sr.name = "sr"
    sr.connect("o1", b2.ports["o1"])

    b3 = c << cells.bend_euler()
    b3.connect("o1", sr.ports["o2"])

    cp.connect("o2", b3.ports["o2"])

    b4 = c << cells.bend_euler()
    b4.connect("o1", sp.ports["o3"])

    h_bot = c << cells.straight_heater_meander(length=length_heater)
    h_bot.connect("o1", b4.ports["o2"])

    b5 = c << cells.bend_euler()
    b5.connect("o2", h_bot.ports["o2"])

    gf.routing.route_bundle(
        c,
        [b5.ports["o1"]],
        [cp.ports["o1"]],
        cross_section="strip_cband",
    )

    c.add_ports(sp.ports.filter(orientation=0), prefix="in")
    c.add_ports(cp.ports.filter(orientation=180), prefix="out")
    return c


def plot_expected_transfer_function(
    length_heater: float = 320.0,
    i_max_mA: float = 60.0,
    n_points: int = 1000,
) -> plt.Figure:
    """Plot the expected MZI transfer function from model parameters.

    Args:
        length_heater: heater length in um (must match cell).
        i_max_mA: maximum sweep current in mA.
        n_points: number of sweep points.
    """
    ohms_per_um = 0.375
    eta_pi_per_W = 12.5
    R = ohms_per_um * length_heater

    current = np.linspace(0, i_max_mA * 1e-3, n_points)
    P = current**2 * R
    dphi = np.pi * eta_pi_per_W * P

    PD1 = 0.5 * (1 + np.sin(dphi))
    PD2 = 0.5 * (1 - np.sin(dphi))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(current * 1e3, PD1, label="PD1 (bar)", linewidth=1.5)
    ax1.plot(current * 1e3, PD2, label="PD2 (cross)", linewidth=1.5)
    ax1.set_ylabel("Normalized optical power")
    ax1.set_title(
        f"Thermal MZI transfer function (L={length_heater} um, R={R:.0f} ohm)"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(current * 1e3, dphi / np.pi, linewidth=1.5, color="C2")
    ax2.set_xlabel("Heater current [mA]")
    ax2.set_ylabel("Phase shift [pi rad]")
    ax2.axhline(
        y=0.5, color="gray", linestyle="--", alpha=0.5, label="pi/2 (quadrature)"
    )
    ax2.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5, label="pi (extinction)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    P_pi = 1.0 / eta_pi_per_W
    i_pi = np.sqrt(P_pi / R) * 1e3
    fig.suptitle(
        f"I_pi = {i_pi:.1f} mA  |  P_pi = {P_pi * 1e3:.1f} mW  |  V_pi = {i_pi * 1e-3 * R:.2f} V",
        y=0.02,
        fontsize=10,
        color="gray",
    )
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    thermal_mzi().show()
    plot_expected_transfer_function()
    plt.show()
