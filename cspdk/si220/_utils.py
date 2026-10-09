"""Utilities shared by band-aware Si220 cells."""

import inspect
from functools import partial

import gdsfactory as gf
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from cspdk.si220.tech import _OPTICAL_NAME, get_band, is_rib


def _default_optical_cross_section(component: ComponentSpec) -> str | None:
    """Return the component's default optical cross-section, if it still uses it.

    ``None`` when the component is already built, has no optical ``cross_section``
    parameter (e.g. metal cells default to ``metal_routing``), or the caller already
    chose a cross-section for it.
    """
    if isinstance(component, gf.Component):
        return None
    if isinstance(component, dict):
        if "cross_section" in component.get("settings", {}):
            return None
        component = component["component"]
    if isinstance(component, partial) and "cross_section" in component.keywords:
        return None
    factory = gf.get_active_pdk().get_cell(component)
    parameter = inspect.signature(factory).parameters.get("cross_section")
    default = None if parameter is None else parameter.default
    if isinstance(default, str) and _OPTICAL_NAME.match(default):
        return default
    return None


def get_band_component(
    component: ComponentSpec, cross_section: CrossSectionSpec
) -> gf.Component:
    """Build a Si220 component in the band of ``cross_section``.

    Any cell with an optical ``cross_section`` parameter gets ``cross_section``, so a
    container never mixes bands. Rib-only cells (default ``rib_*``) stay rib and only
    switch band.
    """
    default = _default_optical_cross_section(component)
    if default is None:
        return gf.get_component(component)
    if is_rib(default) and not is_rib(cross_section):
        cross_section = f"rib_{get_band(cross_section)}"
    return gf.get_component(component, cross_section=cross_section)
