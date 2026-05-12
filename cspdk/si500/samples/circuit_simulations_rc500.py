"""Sample circuit simulations."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import sax

from cspdk.si500 import PDK, cells

if __name__ == "__main__":
    c = cells.mzi_sc(delta_length=100)
    c.show()
    c.plot_netlist()
    netlist = c.get_netlist()
    models = PDK.models
    circuit, _ = sax.circuit(netlist, models=models)  # type: ignore
    wl = jnp.linspace(1.5, 1.6, 256)

    S = circuit(wl=wl)
    plt.figure(figsize=(14, 4))
    plt.title("MZI")
    plt.plot(1e3 * wl, jnp.abs(S["o1", "o2"]) ** 2)  # type: ignore
    plt.xlabel("λ [nm]")
    plt.ylabel("T")
    plt.grid(True)
    plt.show()
