"""Heater resistance vs length: demonstrates R = ohms_per_um * length.

The ThermalPhaseShifter model uses a linear resistivity parameter
so that resistance scales with heater length.

@tags eo dc sweep, circulax simulation, heater, resistance
"""

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    ohms_per_um = 0.375
    lengths = np.array([100.0, 200.0, 320.0, 500.0, 800.0])
    R_arr = ohms_per_um * lengths

    L_continuous = np.linspace(0, max(lengths) * 1.1, 200)
    R_continuous = ohms_per_um * L_continuous

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(
        L_continuous, R_continuous, "k--", alpha=0.4, label=f"R = {ohms_per_um} * L"
    )
    ax1.scatter(lengths, R_arr, s=80, zorder=5, color="C0")
    for ll, rr in zip(lengths, R_arr):
        ax1.annotate(
            f"{rr:.0f} ohm",
            (ll, rr),
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
    for ll in lengths:
        rr = ohms_per_um * ll
        V = current * rr
        ax2.plot(current * 1e3, V, label=f"L={ll:.0f} um (R={rr:.0f} ohm)")
    ax2.set_xlabel("Current [mA]")
    ax2.set_ylabel("Voltage [V]")
    ax2.set_title("I-V curves for different heater lengths")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
