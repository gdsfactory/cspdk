"""Sample routing with layer_transitions and the rib-to-strip transition."""

import gdsfactory as gf

from cspdk.si340 import PDK, cells, tech


@gf.cell
def sample_routing_different_widths() -> gf.Component:
    """Route two straights with different widths to test auto-taper."""
    c = gf.Component()
    s1 = c << cells.straight(length=10, cross_section=tech.xs_sc340(width=0.4))
    s2 = c << cells.straight(length=10, cross_section=tech.xs_sc340(width=1.0))
    s2.dmove((100, 50))
    tech.route_single(
        c,
        s1.ports["o2"],
        s2.ports["o1"],
    )
    return c


@gf.cell
def sample_routing_rib_to_strip() -> gf.Component:
    """Route a strip MMI to a rib waveguide through the foundry transition."""
    c = gf.Component()
    mmi = c << cells.mmi1x2_sc()
    transition = c << cells.taper_rib_to_strip()
    rib = c << cells.straight_rc(length=100)
    transition.dmirror_x()
    transition.dmove(transition.ports["o2"].center, (300, 150))
    rib.connect("o2", transition.ports["o1"])
    tech.route_single(c, mmi.ports["o2"], transition.ports["o2"])
    c.add_port("o1", port=mmi.ports["o1"])
    c.add_port("o2", port=rib.ports["o1"])
    return c


if __name__ == "__main__":
    PDK.activate()
    c = gf.grid([sample_routing_different_widths(), sample_routing_rib_to_strip()])
    c.show()
