"""Sample routing two straights with different widths using layer_transitions."""

import gdsfactory as gf

from cspdk.si500 import PDK, cells, tech


@gf.cell
def sample_routing_different_widths() -> gf.Component:
    """Route two straights with different widths to test auto-taper."""
    c = gf.Component()
    s1 = c << cells.straight(length=10, cross_section=tech.xs_rc(width=0.4))
    s2 = c << cells.straight(length=10, cross_section=tech.xs_rc(width=1.0))
    s2.dmove((100, 50))
    tech.route_single(
        c,
        s1.ports["o2"],
        s2.ports["o1"],
    )
    return c


if __name__ == "__main__":
    PDK.activate()
    c = sample_routing_different_widths()
    c.show()
