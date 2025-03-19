"""Circuit simulation with routes."""

import gdsfactory as gf
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sax

from cspdk.si220 import PDK, cells

def lattice_filter():

if __name__ == "__main__":
    c = gf.Component()
    mzi_lattice = cells.mzi_lattice(delta_length=10)

    netlist = c.get_netlist(recursive=True)
    c.plot_netlist(recursive=True)
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
