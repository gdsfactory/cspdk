"""A ring and a loopback in the 6 x 3 mm design area of the e-beam event.

EVENT-ONLY: the CORNERSTONE e-beam event rules are provisional, and the die's
grating I/O is a placeholder. The ring sits on the bottom I/O row and a
loopback (a straight, for normalising the grating response) on the next row.
"""

from __future__ import annotations

import gdsfactory as gf
import jax.numpy as jnp
import sax
from cspdk.si220_ebeam_event import PDK, cells
from cspdk.si220_ebeam_event.tech import route_single

WL0 = 1.55  # um


@gf.cell
def ring_die(gap: float = 0.25, radius: float = 25.0) -> gf.Component:
    """The event die with a ring on the bottom row and a loopback above it.

    Args:
        gap: gap between the bus and the ring in um (>= 0.2).
        radius: ring bend radius in um (>= 25).
    """
    c = gf.Component()
    die = c << cells.die()
    n_io = len(die.ports) // 2
    # Inputs o1..o{n_io} run bottom to top; outputs run top to bottom, so input
    # o{i} faces output o{2 n_io + 1 - i} on the same row.
    ring = c << cells.ring_single(gap=gap, radius=radius)
    ring.move((-ring.x, die.ports["o1"].y - ring.ports["o1"].y))
    route_single(c, die.ports["o1"], ring.ports["o1"])
    route_single(c, ring.ports["o2"], die.ports[f"o{2 * n_io}"])
    route_single(c, die.ports["o2"], die.ports[f"o{2 * n_io - 1}"])
    return c


def ring_transmission(
    gap: float = 0.25, radius: float = 25.0, span: float = 0.02, points: int = 2001
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return wavelengths and the ring's through-port power transmission.

    The ring coupler model is a placeholder (fixed power coupling, no gap
    dependence); the waveguide uses the event's group index (3.85) and loss
    (2.5 dB/cm), so the free spectral range is meaningful.

    Args:
        gap: gap between the bus and the ring in um.
        radius: ring bend radius in um.
        span: wavelength span around 1.55 um, in um.
        points: number of wavelength points.
    """
    PDK.activate()
    ring = cells.ring_single(gap=gap, radius=radius)
    circuit, _ = sax.circuit(ring.get_netlist(recursive=True), models=PDK.models)
    wl = jnp.linspace(WL0 - span / 2, WL0 + span / 2, points)
    return wl, jnp.abs(circuit(wl=wl)["o1", "o2"]) ** 2


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from cspdk.si220_ebeam_event import check, post_process

    PDK.activate()
    design = ring_die()
    for violation in check(design):
        print(violation)
    post_process(design).show()  # what the foundry exposes on 4/0
    wl, t = ring_transmission()
    plt.plot(wl * 1e3, 10 * jnp.log10(t))
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Transmission [dB]")
    plt.show()
