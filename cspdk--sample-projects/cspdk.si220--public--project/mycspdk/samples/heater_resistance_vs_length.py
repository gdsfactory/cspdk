"""Heater resistance vs length: demonstrates R = ohms_per_um * length.

The ThermalPhaseShifter model uses a linear resistivity parameter
(ohms_per_um) so that resistance scales with heater length. This script
visualizes the scaling and compares heaters of different lengths.

@tags eo dc sweep, circulax simulation, heater, resistance
"""

from __future__ import annotations

import gdsfactory as gf
import numpy as np
from cspdk.si220 import cells
from matplotlib import pyplot as plt


def plot_resistance_vs_length(
    lengths: tuple[float, ...] = (100.0, 200.0, 320.0, 500.0, 800.0),
    ohms_per_um: float = 0.375,
) -> plt.Figure:
    """Plot heater resistance as a function of length.

    Args:
        lengths: heater lengths in um to plot.
        ohms_per_um: linear resistivity (ohm/um) from ThermalPhaseShifter model.
    """
    lengths_arr = np.array(lengths)
    R_arr = ohms_per_um * lengths_arr

    L_continuous = np.linspace(0, max(lengths) * 1.1, 200)
    R_continuous = ohms_per_um * L_continuous

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(
        L_continuous, R_continuous, "k--", alpha=0.4, label=f"R = {ohms_per_um} * L"
    )
    ax1.scatter(lengths_arr, R_arr, s=80, zorder=5, color="C0")
    for L, R in zip(lengths_arr, R_arr):
        ax1.annotate(
            f"{R:.0f} ohm",
            (L, R),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
        )
    ax1.set_xlabel("Heater length [um]")
    ax1.set_ylabel("Resistance [ohm]")
    ax1.set_title("Resistance scales linearly with length")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, None)
    ax1.set_ylim(0, None)

    current = np.linspace(0, 60e-3, 500)
    for L in lengths:
        R = ohms_per_um * L
        V = current * R
        ax2.plot(current * 1e3, V, label=f"L={L:.0f} um (R={R:.0f} ohm)")
    ax2.set_xlabel("Current [mA]")
    ax2.set_ylabel("Voltage [V]")
    ax2.set_title("I-V curves for different heater lengths")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def show_heater_cells(
    lengths: tuple[float, ...] = (100.0, 320.0, 800.0),
) -> gf.Component:
    """Create heater cells at different lengths for visual comparison.

    Args:
        lengths: heater lengths in um.
    """
    c = gf.Component()
    y_offset = 0.0
    for length in lengths:
        heater = c << cells.straight_heater_meander(length=length)
        heater.dmovey(y_offset)
        y_offset += 150.0
    return c


if __name__ == "__main__":
    fig = plot_resistance_vs_length()
    plt.show()

    c = show_heater_cells()
    c.show()
