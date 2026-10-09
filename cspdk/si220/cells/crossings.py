"""Waveguide crossings drawn from the Cornerstone Run-50 foundry geometry."""

import warnings

import gdsfactory as gf
import kfactory as kf
from gdsfactory.typings import CrossSectionSpec

from cspdk.si220._schematic import FOUNDRY_SCHEMATICS as _FOUNDRY
from cspdk.si220._schematic import crossing_rib_schematic, crossing_schematic
from cspdk.si220.tech import LAYER, get_band, is_rib

# Arm width profiles of the foundry crossings, as (distance from the centre,
# half-width) in um, from the arm tip to the corner where two arms meet on the
# diagonal (distance == half-width). Extracted from the cells
# SOI220_{band}_TE_{kind}_crossing of
# references/october-2026/CORNERSTONE-MPW-Run-50-GDSII-Template.gds; all four arms
# share the profile. The rib profiles start with a 5 um straight.
CROSSING_PROFILES: dict[tuple[str, str], tuple[tuple[float, float], ...]] = {
    ("cband", "strip"): (
        (4.62, 0.225),
        (4.235, 0.219),
        (3.85, 0.3),
        (3.465, 0.6),
        (3.08, 0.667),
        (2.695, 0.685),
        (2.31, 0.709),
        (1.925, 0.8),
        (1.54, 0.794),
        (1.155, 0.924),
        (0.924, 0.924),
    ),
    ("oband", "strip"): (
        (4.235, 0.2),
        (3.85, 0.3),
        (3.465, 0.6),
        (3.08, 0.584),
        (2.695, 0.584),
        (2.31, 0.653),
        (1.925, 0.714),
        (1.54, 0.85),
        (1.155, 0.925),
        (0.962, 0.962),
    ),
    ("cband", "rib"): (
        (9.62, 0.225),
        (4.62, 0.225),
        (4.235, 0.4),
        (3.85, 0.452),
        (3.465, 0.7),
        (3.08, 0.672),
        (2.695, 0.658),
        (2.31, 0.654),
        (1.925, 0.7),
        (1.54, 0.725),
        (1.155, 0.825),
        (0.844, 0.844),
    ),
    ("oband", "rib"): (
        (9.62, 0.2),
        (4.62, 0.2),
        (4.235, 0.36),
        (3.85, 0.5),
        (3.465, 0.595),
        (3.08, 0.603),
        (2.695, 0.6),
        (2.31, 0.689),
        (1.925, 0.728),
        (1.54, 0.725),
        (1.155, 0.825),
        (0.845, 0.845),
    ),
}


def _crossing_outline(
    profile: tuple[tuple[float, float], ...],
) -> list[tuple[float, float]]:
    """Closed outline of four arms sharing ``profile``, centred on the origin."""
    # One quarter: down the left side of the south arm to the corner, then out
    # along the bottom side of the west arm. Rotating it by -90 deg three times
    # gives the other quarters.
    quarter = [(-hw, -d) for d, hw in profile]
    quarter += [(-d, -hw) for d, hw in reversed(profile[:-1])]
    points = []
    for _ in range(4):
        points += quarter
        quarter = [(y, -x) for x, y in quarter]
    return points


@gf.cell(tags=["crossings"], schematic_function=crossing_schematic)
def crossing(cross_section: CrossSectionSpec = "strip_cband") -> gf.Component:
    """Return the foundry waveguide crossing for the cross-section's band and type.

    The arm profile is the Cornerstone Run-50 one (see ``CROSSING_PROFILES``). Rib
    crossings get a 5 um straight on each arm and a slab that is the core grown by
    the cross-section's slab offset, clipped to the arm ends.

    Args:
        cross_section: strip or rib, C- or O-band cross-section.
    """
    band = get_band(cross_section)
    rib = is_rib(cross_section)
    profile = CROSSING_PROFILES[band, "rib" if rib else "strip"]
    xs = gf.get_cross_section(cross_section)

    c = gf.Component()
    core = c.add_polygon(_crossing_outline(profile), layer=LAYER.WG)
    if rib:
        dbu = c.kcl.dbu
        core_region = kf.kdb.Region(core.polygon)
        for layer, offset in zip(xs.bbox_layers, xs.bbox_offsets):
            slab = core_region.sized(round(offset / dbu)) & kf.kdb.Region(
                core_region.bbox()
            )
            c.add_polygon(slab, layer=layer)

    length, half_width = profile[0]
    width = 2 * half_width
    for name, center, orientation in (
        ("o1", (-length, 0), 180),
        ("o2", (0, length), 90),
        ("o3", (length, 0), 0),
        ("o4", (0, -length), 270),
    ):
        c.add_port(
            name,
            center=center,
            orientation=orientation,
            width=width,
            layer=LAYER.WG,
            port_type="optical",
        )
    return c


@gf.cell(tags=["crossings"], schematic_function=crossing_rib_schematic)
def crossing_rib(cross_section: CrossSectionSpec = "rib_cband") -> gf.Component:
    """Deprecated: use ``crossing`` with a rib cross-section.

    Args:
        cross_section: rib cross-section selecting the band.
    """
    warnings.warn(
        "crossing_rib is deprecated; use crossing", DeprecationWarning, stacklevel=2
    )
    return crossing(cross_section=cross_section)


################
# Foundry cells
################


@gf.cell(
    tags=["crossings", "foundry"],
    schematic_function=_FOUNDRY["SOI220_cband_TE_strip_crossing"],
)
def SOI220_cband_TE_strip_crossing() -> gf.Component:
    """Cornerstone C-band strip crossing (Run-50 template cell)."""
    return crossing(cross_section="strip_cband")


@gf.cell(
    tags=["crossings", "foundry"],
    schematic_function=_FOUNDRY["SOI220_cband_TE_rib_crossing"],
)
def SOI220_cband_TE_rib_crossing() -> gf.Component:
    """Cornerstone C-band rib crossing (Run-50 template cell)."""
    return crossing(cross_section="rib_cband")


@gf.cell(
    tags=["crossings", "foundry"],
    schematic_function=_FOUNDRY["SOI220_oband_TE_strip_crossing"],
)
def SOI220_oband_TE_strip_crossing() -> gf.Component:
    """Cornerstone O-band strip crossing (Run-50 template cell)."""
    return crossing(cross_section="strip_oband")


@gf.cell(
    tags=["crossings", "foundry"],
    schematic_function=_FOUNDRY["SOI220_oband_TE_rib_crossing"],
)
def SOI220_oband_TE_rib_crossing() -> gf.Component:
    """Cornerstone O-band rib crossing (Run-50 template cell)."""
    return crossing(cross_section="rib_oband")
