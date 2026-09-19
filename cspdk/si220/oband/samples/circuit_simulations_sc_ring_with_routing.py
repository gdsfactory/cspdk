"""Circuit simulation with routes."""

if __name__ == "__main__":
    import gdsfactory as gf
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import sax

    from cspdk.si220.oband import PDK, cells, tech

    c = gf.Component()
    r1 = c << cells.ring_single(radius=5)
    r2 = c << cells.ring_single(radius=15)
    r2.dmove((200, 200))

    route = tech.route_bundle(c, [r1.ports["o2"]], [r2.ports["o1"]])
    c.add_port(name="o1", port=r1.ports["o1"])
    c.add_port(name="o2", port=r2.ports["o2"])
    c.show()
    netlist = c.get_netlist(recursive=True)
    c.plot_netlist(recursive=True)
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
