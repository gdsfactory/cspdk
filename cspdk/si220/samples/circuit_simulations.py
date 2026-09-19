"""Circuit simulation."""

import jax.numpy as jnp
import sax

from cspdk.si220 import PDK, cells

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    PDK.activate()
    # c = cells.mzi(delta_length=12, splitter=cells.mmi2x2)
    c = cells.mzi(
        delta_length=12,
        splitter=cells.coupler,
        cross_section="strip_oband",
    )
    c.show()
    netlist = c.get_netlist()
    c.plot_netlist()
    models = PDK.models
    circuit, _ = sax.circuit(netlist, models=models)  # type: ignore
    wl = jnp.linspace(1.26, 1.36, 256)

    S = circuit(wl=wl)
    plt.figure(figsize=(14, 4))
    plt.title("MZI")
    plt.plot(1e3 * wl, jnp.abs(S["o1", "o3"]) ** 2)  # type: ignore
    plt.plot(1e3 * wl, jnp.abs(S["o1", "o4"]) ** 2)  # type: ignore
    plt.xlabel("λ [nm]")
    plt.ylabel("T")
    plt.grid(True)
    plt.show()
