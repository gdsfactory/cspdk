"""Sample routing two straights with different widths using layer_transitions."""

import gdsfactory as gf

from cspdk.sin200 import PDK, cells, tech


@gf.cell
def sample_routing_different_widths() -> gf.Component:
    """Route two 638 nm straights with different widths to test auto-taper."""
    c = gf.Component()
    s1 = c << cells.straight(length=10, cross_section=tech.xs_n638(width=0.3))
    s2 = c << cells.straight(length=10, cross_section=tech.xs_n638(width=0.6))
    s2.dmove((200, 100))
    tech.route_single(
        c,
        s1.ports["o2"],
        s2.ports["o1"],
        cross_section="xs_n638",
        straight="straight_n638",
        bend="bend_euler_n638",
    )
    return c


if __name__ == "__main__":
    PDK.activate()
    c = sample_routing_different_widths()
    c.show()
