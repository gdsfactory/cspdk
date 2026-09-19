"""This module contains the building blocks for the CSPDK PDK."""

from functools import partial

import gdsfactory as gf
from gdsfactory.typings import (
    CrossSectionSpec,
)

from cspdk.si220._schematic import mmi1x2_schematic, mmi2x2_schematic
from cspdk.si220.tech import get_band, is_rib

################
# MMIs
################


@gf.cell(tags=["mmis"], schematic_function=mmi1x2_schematic)
def mmi1x2(
    width: float | None = None,
    width_taper: float = 1.5,
    length_taper: float = 20.0,
    length_mmi: float | None = None,
    width_mmi: float = 6.0,
    gap_mmi: float | None = None,
    cross_section: CrossSectionSpec = "strip_cband",
) -> gf.Component:
    """An mmi1x2.

    An mmi1x2 is a splitter that splits a single input to two outputs

    Args:
        width: the width of the waveguides connecting at the mmi ports.
        width_taper: the width at the base of the mmi body.
        length_taper: the length of the tapers going towards the mmi body.
        length_mmi: the length of the mmi body.
        width_mmi: the width of the mmi body.
        gap_mmi: the gap between the tapers at the mmi body.
        cross_section: a cross section or its name or a function generating a cross section.
    """
    band = get_band(cross_section)
    uses_rib = is_rib(cross_section)
    if length_mmi is None:
        length_mmi = (
            (40.8 if uses_rib else 40.0)
            if band == "oband"
            else (32.7 if uses_rib else 31.0)
        )
    if gap_mmi is None:
        gap_mmi = 1.55 if band == "oband" and uses_rib else 1.64
    return gf.c.mmi1x2(
        width=width,
        width_taper=width_taper,
        length_taper=length_taper,
        length_mmi=length_mmi,
        width_mmi=width_mmi,
        gap_mmi=gap_mmi,
        cross_section=cross_section,
    )


mmi1x2_rib = partial(mmi1x2, cross_section="rib_cband")


@gf.cell(tags=["mmis"], schematic_function=mmi2x2_schematic)
def mmi2x2(
    width: float | None = None,
    width_taper: float = 1.5,
    length_taper: float = 20.0,
    length_mmi: float | None = None,
    width_mmi: float = 6.0,
    gap_mmi: float | None = None,
    cross_section: CrossSectionSpec = "strip_cband",
) -> gf.Component:
    """An mmi2x2.

    An mmi2x2 is a 2x2 splitter

    Args:
        width: the width of the waveguides connecting at the mmi ports
        width_taper: the width at the base of the mmi body
        length_taper: the length of the tapers going towards the mmi body
        length_mmi: the length of the mmi body
        width_mmi: the width of the mmi body
        gap_mmi: the gap between the tapers at the mmi body
        cross_section: a cross section or its name or a function generating a cross section.
    """
    band = get_band(cross_section)
    uses_rib = is_rib(cross_section)
    if length_mmi is None:
        length_mmi = (
            (55.0 if uses_rib else 53.5)
            if band == "oband"
            else (44.8 if uses_rib else 42.5)
        )
    if gap_mmi is None:
        gap_mmi = 0.53 if band == "oband" or uses_rib else 0.5
    return gf.c.mmi2x2(
        width=width,
        width_taper=width_taper,
        length_taper=length_taper,
        length_mmi=length_mmi,
        width_mmi=width_mmi,
        gap_mmi=gap_mmi,
        cross_section=cross_section,
    )


mmi2x2_rib = partial(mmi2x2, cross_section="rib_cband")
