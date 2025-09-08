"""Circuit simulation."""

if __name__ == "__main__":
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import sax

    from cspdk.si220.cband import PDK, cells

    c = cells.ring_single(radius=10)
    c.show()
    netlist = c.get_netlist()
    c.plot_netlist()
    models = PDK.models
    circuit, _ = sax.circuit(netlist, models=models)  # type: ignore
    wl = jnp.linspace(1.5, 1.6, 3000)

    S = circuit(wl=wl)
    plt.figure(figsize=(14, 4))
    plt.title("MZI")
    plt.plot(1e3 * wl, jnp.abs(S["o1", "o2"]) ** 2)  # type: ignore
    plt.xlabel("λ [nm]")
    plt.ylabel("T")
    plt.grid(True)
    plt.show()
