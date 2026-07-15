"""Thermal MZI: expected transfer function from model parameters.

Plots the complementary optical outputs of an MZI with thermal phase
shifters as a function of heater current.

@tags eo dc sweep, circulax simulation, mzi, thermal phase shifter
"""

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    ohms_per_um = 0.375
    eta_pi_per_W = 12.5
    length_heater = 320.0
    R = ohms_per_um * length_heater

    current = np.linspace(0, 60e-3, 1000)
    P = current**2 * R
    dphi = np.pi * eta_pi_per_W * P

    PD1 = 0.5 * (1 + np.sin(dphi))
    PD2 = 0.5 * (1 - np.sin(dphi))

    P_pi = 1.0 / eta_pi_per_W
    i_pi = np.sqrt(P_pi / R) * 1e3

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

    fig.suptitle(
        f"I_pi = {i_pi:.1f} mA  |  P_pi = {P_pi * 1e3:.1f} mW  |  V_pi = {i_pi * 1e-3 * R:.2f} V",
        y=0.02,
        fontsize=10,
        color="gray",
    )
    plt.tight_layout()
    plt.show()
