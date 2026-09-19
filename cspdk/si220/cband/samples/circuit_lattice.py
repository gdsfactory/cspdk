"""Circuit simulation of an MZI lattice stage."""

import jax.numpy as jnp
import sax

from cspdk.si220.cband import PDK, cells


def lattice_filter(delta_length: float = 10.0):
    """Return the default cband MZI lattice stage."""
    return cells.mzi_lattice(delta_length=delta_length)


def simulate_lattice(wavelengths=None):
    """Simulate the lattice stage over a wavelength sweep."""
    wavelengths = jnp.linspace(1.5, 1.6, 256) if wavelengths is None else wavelengths
    component = lattice_filter()
    circuit, _ = sax.circuit(component.get_netlist(recursive=True), models=PDK.models)
    return wavelengths, circuit(wl=wavelengths)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    wl, s_parameters = simulate_lattice()
    plt.plot(1e3 * wl, jnp.abs(s_parameters["o1", "o2"]) ** 2)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmission")
    plt.grid()
    plt.show()
