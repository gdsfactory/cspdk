"""MZI between grating couplers on Cornerstone Si SOI 340 nm.

The band follows the cross-section: xs_sc340 (C-band strip) or xs_so340 (O-band strip) or xs_rc340 (C-band rib).
"""

from __future__ import annotations

import gdsfactory as gf
import jax.numpy as jnp
import sax
from cspdk.si340 import PDK

# cross-section -> (cell-name suffix, band centre wavelength in um)
_BANDS = {
    "xs_sc340": ("sc", 1.55),
    "xs_so340": ("so", 1.31),
    "xs_rc340": ("rc", 1.55),
}


@gf.cell
def mzi_with_gratings(
    delta_length: float = 100.0,
    cross_section: str = "xs_sc340",
    grating_pitch: float = 127.0,
) -> gf.Component:
    """MZI with one input and two output grating couplers.

    Args:
        delta_length: arm length difference in um.
        cross_section: xs_sc340 or xs_so340 or xs_rc340.
        grating_pitch: spacing between the output grating couplers in um.
    """
    band = _BANDS[cross_section][0]
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

    for mzi_port, name, dy in (
        ("o2", "out1", -grating_pitch / 2),
        ("o3", "out2", grating_pitch / 2),
    ):
        gc_out = c << gc
        gc_out.dmove(
            (
                mzi.dxmax + 100 - gc_out.ports["o1"].dx,
                mzi.ports["o1"].dy + dy - gc_out.ports["o1"].dy,
            )
        )
        gf.routing.route_single(
            c, mzi.ports[mzi_port], gc_out.ports["o1"], cross_section=cross_section
        )
        c.add_port(name, port=gc_out.ports["o2"])
    return c


def mzi_spectrum(
    delta_length: float = 100.0,
    cross_section: str = "xs_sc340",
    span: float = 0.05,
    points: int = 501,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Return wavelengths and the transmission to each output grating coupler.

    Args:
        delta_length: arm length difference in um.
        cross_section: the band's cross-section, as for mzi_with_gratings.
        span: wavelength span around the band centre in um.
        points: number of wavelength points.
    """
    PDK.activate()
    component = mzi_with_gratings(
        delta_length=delta_length, cross_section=cross_section
    )
    circuit, _ = sax.circuit(component.get_netlist(recursive=True), models=PDK.models)
    wl0 = _BANDS[cross_section][1]
    wl = jnp.linspace(wl0 - span / 2, wl0 + span / 2, points)
    s = circuit(wl=wl)
    return wl, {out: jnp.abs(s["in", out]) ** 2 for out in ("out1", "out2")}


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    PDK.activate()
    mzi_with_gratings().show()
    for xs in _BANDS:
        wl, t = mzi_spectrum(cross_section=xs)
        for out, power in t.items():
            plt.plot(wl * 1e3, 10 * jnp.log10(power), label=f"{xs} {out}")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Transmission [dB]")
    plt.legend()
    plt.show()
