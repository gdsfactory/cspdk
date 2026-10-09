"""Pack a ring-resonator sweep on Cornerstone Si220."""

import gdsfactory as gf
from cspdk.si220 import cells


@gf.cell
def sample4_pack() -> gf.Component:
    """Pack single rings sweeping radius and coupling gap into one block."""
    rings = [
        cells.ring_single(radius=radius, gap=gap)
        for radius in (10, 15, 20, 30)
        for gap in (0.2, 0.25, 0.3)
    ]
    bins = gf.pack(
        rings,  # Must be a list or tuple of Components
        spacing=20,  # Minimum distance between adjacent rings
        aspect_ratio=(1, 1),  # Shape of the box
        max_size=(1000, 1000),  # Limits the size into which the rings are packed
        sort_by_area=True,  # Pre-sorts the rings by area
    )
    return bins[0]


if __name__ == "__main__":
    sample4_pack().show()
