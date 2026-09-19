"""Utilities shared by band-aware Si220 cells."""

import gdsfactory as gf
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

_BAND_AWARE_COMPONENTS = {
    "bend_euler",
    "coupler",
    "coupler_ring",
    "grating_coupler_elliptical",
    "grating_coupler_rectangular",
    "mmi1x2",
    "mmi2x2",
    "straight",
}


def get_band_component(
    component: ComponentSpec, cross_section: CrossSectionSpec
) -> gf.Component:
    """Resolve a shared Si220 component using the selected optical band."""
    component_name = (
        component
        if isinstance(component, str)
        else getattr(component, "__name__", None)
    )
    if component_name in _BAND_AWARE_COMPONENTS:
        return gf.get_component(component, cross_section=cross_section)
    return gf.get_component(component)
