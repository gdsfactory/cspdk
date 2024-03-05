import sax

import cspdk
from cspdk.models import models

if __name__ == "__main__":
    c = cspdk.cells.mzi_sc()
    netlist = c.get_netlist()
    circuit, _ = sax.circuit(netlist, models=models)
    # result = circuit()

    # wl = np.linspace(1.5, 1.6)
    # S = circuit(wl=wl)

    # plt.figure(figsize=(14, 4))
    # plt.title("MZI")
    # plt.plot(1e3 * wl, jnp.abs(S["o1", "o2"]) ** 2)
    # plt.xlabel("λ [nm]")
    # plt.ylabel("T")
    # plt.grid(True)
    # plt.show()
