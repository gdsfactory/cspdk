"""MZI between grating couplers on Cornerstone SiN300, in C-band or O-band.

The band follows the cross-section: ``xs_nc`` for C-band, ``xs_no`` for O-band.
"""

from __future__ import annotations

import gdsfactory as gf
import jax.numpy as jnp
import sax
from cspdk.sin300 import PDK

_BAND_WL = {"xs_nc": 1.55, "xs_no": 1.31}


@gf.cell
def mzi_with_gratings(
    delta_length: float = 100.0,
    cross_section: str = "xs_nc",
    grating_pitch: float = 127.0,
) -> gf.Component:
    """MZI with one input and two output grating couplers.

    Args:
        delta_length: arm length difference in um.
        cross_section: xs_nc (C-band) or xs_no (O-band).
        grating_pitch: spacing between the output grating couplers in um.
    """
    band = cross_section.removeprefix("xs_")
    c = gf.Component()
    mzi = c << gf.get_component(f"mzi_{band}", delta_length=delta_length)
    gc = gf.get_component(f"grating_coupler_rectangular_{band}")

    gc_in = c << gc
    gc_in.drotate(180)
    gc_in.dmove(
        (mzi.dxmin - 100 - gc_in.dxmax, mzi.ports["o1"].dy - gc_in.ports["o1"].dy)
    )

    gf.routing.route_single(
        c, gc_in.ports["o1"], mzi.ports["o1"], cross_section=cross_section
    )
    c.add_port("in", port=gc_in.ports["o2"])

    for i, (mzi_port, name) in enumerate((("o2", "out1"), ("o3", "out2"))):
        gc_out = c << gc
        gc_out.dmove(
            (
                mzi.dxmax + 100 - gc_out.ports["o1"].dx,
                mzi.ports["o1"].dy + (i - 0.5) * grating_pitch - gc_out.ports["o1"].dy,
            )
        )
        gf.routing.route_single(
            c, mzi.ports[mzi_port], gc_out.ports["o1"], cross_section=cross_section
        )
        c.add_port(name, port=gc_out.ports["o2"])
    return c


def mzi_spectrum(
    delta_length: float = 100.0,
    cross_section: str = "xs_nc",
    span: float = 0.05,
    points: int = 501,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Return wavelengths and the transmission to each output grating coupler.

    Args:
        delta_length: arm length difference in um.
        cross_section: xs_nc (C-band) or xs_no (O-band).
        span: wavelength span around the band centre in um.
        points: number of wavelength points.
    """
    PDK.activate()
    component = mzi_with_gratings(
        delta_length=delta_length, cross_section=cross_section
    )
    circuit, _ = sax.circuit(component.get_netlist(recursive=True), models=PDK.models)
    wl0 = _BAND_WL[cross_section]
    wl = jnp.linspace(wl0 - span / 2, wl0 + span / 2, points)
    s = circuit(wl=wl)
    return wl, {out: jnp.abs(s["in", out]) ** 2 for out in ("out1", "out2")}


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    PDK.activate()
    mzi_with_gratings().show()
    for xs in _BAND_WL:
        wl, t = mzi_spectrum(cross_section=xs)
        for out, power in t.items():
            plt.plot(wl * 1e3, 10 * jnp.log10(power), label=f"{xs} {out}")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Transmission [dB]")
    plt.legend()
    plt.show()
