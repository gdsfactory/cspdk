"""Waveguide loop for cut-back loss measurement on Cornerstone Ge-on-Si.

The loop is a waveguide of a given length folded into a U with bends. Comparing
the transmission of loops of different lengths gives the propagation loss.
"""

from __future__ import annotations

import gdsfactory as gf
import jax.numpy as jnp
import sax
from cspdk.ge_on_si import PDK

CROSS_SECTION = "xs_rib"
WL0 = 3.8  # um


@gf.cell
def waveguide_loop(length: float = 1000.0) -> gf.Component:
    """Two straight arms of ``length / 2`` joined by a U of bends.

    Args:
        length: total length of the two straight arms in um.
    """
    c = gf.Component()
    radius = gf.get_cross_section(CROSS_SECTION).radius
    arm = gf.get_component("straight", length=length / 2, cross_section=CROSS_SECTION)
    lower = c << arm
    upper = c << arm
    upper.dmove((0, 4 * radius))
    gf.routing.route_single(
        c, lower.ports["o2"], upper.ports["o2"], cross_section=CROSS_SECTION
    )
    c.add_port("o1", port=lower.ports["o1"])
    c.add_port("o2", port=upper.ports["o1"])
    return c


def loop_transmission(
    length: float = 1000.0, span: float = 0.1, points: int = 101
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return wavelengths and the loop's power transmission.

    Args:
        length: total length of the two straight arms in um.
        span: wavelength span around 3.8 um, in um.
        points: number of wavelength points.
    """
    PDK.activate()
    component = waveguide_loop(length=length)
    circuit, _ = sax.circuit(component.get_netlist(recursive=True), models=PDK.models)
    wl = jnp.linspace(WL0 - span / 2, WL0 + span / 2, points)
    return wl, jnp.abs(circuit(wl=wl)["o1", "o2"]) ** 2


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    PDK.activate()
    waveguide_loop().show()
    for length in (1000.0, 2000.0, 4000.0):
        wl, t = loop_transmission(length=length)
        plt.plot(wl * 1e3, 10 * jnp.log10(t), label=f"{length:.0f} um")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Transmission [dB]")
    plt.legend()
    plt.show()
