"""Heater IV curve: electrical characterization of a thermal phase shifter.

Shows the linear I-V relationship (pure resistor) and the quadratic
power dissipation curve, with annotation of P_pi.

@tags eo dc sweep, circulax simulation, heater, iv curve
"""

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    ohms_per_um = 0.375
    eta_pi_per_W = 12.5
    length = 320.0

    R = ohms_per_um * length
    P_pi = 1.0 / eta_pi_per_W
    i_pi = np.sqrt(P_pi / R)
    V_pi = i_pi * R

    current = np.linspace(0, 60e-3, 500)
    V = current * R
    P = current**2 * R
    dphi = np.pi * eta_pi_per_W * P

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.plot(current * 1e3, V, linewidth=2, color="C0")
    ax.axhline(y=V_pi, color="C3", linestyle="--", alpha=0.6)
    ax.axvline(x=i_pi * 1e3, color="C3", linestyle="--", alpha=0.6)
    ax.plot(i_pi * 1e3, V_pi, "o", color="C3", markersize=8)
    ax.annotate(
        f"V_pi = {V_pi:.2f} V\nI_pi = {i_pi * 1e3:.1f} mA",
        (i_pi * 1e3, V_pi),
        textcoords="offset points",
        xytext=(15, -15),
        fontsize=9,
        color="C3",
    )
    ax.set_xlabel("Current [mA]")
    ax.set_ylabel("Voltage [V]")
    ax.set_title(f"I-V curve (R = {R:.0f} ohm)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(current * 1e3, P * 1e3, linewidth=2, color="C1")
    ax.axhline(y=P_pi * 1e3, color="C3", linestyle="--", alpha=0.6)
    ax.axvline(x=i_pi * 1e3, color="C3", linestyle="--", alpha=0.6)
    ax.plot(i_pi * 1e3, P_pi * 1e3, "o", color="C3", markersize=8)
    ax.annotate(
        f"P_pi = {P_pi * 1e3:.1f} mW",
        (i_pi * 1e3, P_pi * 1e3),
        textcoords="offset points",
        xytext=(15, -15),
        fontsize=9,
        color="C3",
    )
    ax.set_xlabel("Current [mA]")
    ax.set_ylabel("Power [mW]")
    ax.set_title("Power dissipation (P = I^2 * R)")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(P * 1e3, dphi / np.pi, linewidth=2, color="C2")
    ax.axvline(x=P_pi * 1e3, color="C3", linestyle="--", alpha=0.6)
    ax.axhline(y=1.0, color="C3", linestyle="--", alpha=0.6)
    ax.plot(P_pi * 1e3, 1.0, "o", color="C3", markersize=8)
    ax.set_xlabel("Power [mW]")
    ax.set_ylabel("Phase shift [pi rad]")
    ax.set_title("Phase shift vs power (linear)")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Heater: L={length} um, R={R:.0f} ohm, eta_pi={eta_pi_per_W} /W",
        fontsize=11,
    )
    plt.tight_layout()
    plt.show()
