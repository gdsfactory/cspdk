import gdsfactory as gf
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sax

import cspdk
from cspdk import PDK

if __name__ == "__main__":
    c = gf.Component()
    mzi1 = c << cspdk.cells.mzi_sc(delta_length=10)
    mzi2 = c << cspdk.cells.mzi_sc(delta_length=100)
    mzi2.move((200, 200))
    route = cspdk.tech.get_route_sc(mzi1.ports["o2"], mzi2.ports["o1"])
    c.add(route.references)
    c.add_port(name="o1", port=mzi1.ports["o1"])
    c.add_port(name="o2", port=mzi2.ports["o2"])
    c.show()
    c.plot_netlist_flat()
    netlist = c.get_netlist_recursive()
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
