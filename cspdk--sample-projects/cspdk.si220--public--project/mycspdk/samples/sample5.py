"""Extrude a free-form path with a Cornerstone Si220 cross-section."""

import gdsfactory as gf


@gf.cell
def sample5_path() -> gf.Component:
    """Strip waveguide along arcs, Euler bends and straights."""
    p = gf.Path()
    p += gf.path.arc(radius=10, angle=90)  # Circular arc
    p += gf.path.straight(length=10)  # Straight section
    p += gf.path.euler(radius=10, angle=-90)  # Euler bend (aka "racetrack" curve)
    p += gf.path.straight(length=40)
    p += gf.path.arc(radius=20, angle=-45)
    p += gf.path.straight(length=10)
    p += gf.path.arc(radius=20, angle=45)
    p += gf.path.straight(length=10)
    return p.extrude(cross_section="strip_cband")


if __name__ == "__main__":
    sample5_path().show()
