"""Utilities shared by band-aware Si220 cells."""

import inspect
from functools import partial

import gdsfactory as gf
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from cspdk.si220.tech import _OPTICAL_NAME, cross_sections, get_band, is_rib


def _is_optical(cross_section: CrossSectionSpec) -> bool:
    """Whether a spec is one of the si220 optical (strip/rib, C/O-band) cross-sections."""
    if isinstance(cross_section, str):
        return bool(_OPTICAL_NAME.match(cross_section))
    xs = gf.get_cross_section(cross_section)
    return xs.sections[0].name in ("core_cband", "core_oband")


def _current_cross_section(
    component: ComponentSpec,
) -> tuple[CrossSectionSpec | None, bool]:
    """Return the cross-section a component spec would use, and whether it was set explicitly."""
    if isinstance(component, dict):
        settings = component.get("settings", {})
        if "cross_section" in settings:
            return settings["cross_section"], True
        component = component["component"]
    if isinstance(component, partial) and "cross_section" in component.keywords:
        return component.keywords["cross_section"], True
    factory = gf.get_active_pdk().get_cell(component)
    parameter = inspect.signature(factory).parameters.get("cross_section")
    return (None if parameter is None else parameter.default), False


def _variant(cross_section: CrossSectionSpec, band: str, rib: bool) -> CrossSectionSpec:
    """``cross_section`` as the given band and type, keeping a custom width."""
    source_band = get_band(cross_section)
    source_rib = is_rib(cross_section)
    if source_band == band and source_rib == rib:
        return cross_section
    factory = cross_sections[f"{'rib' if rib else 'strip'}_{band}"]
    source_factory = cross_sections[f"{'rib' if source_rib else 'strip'}_{source_band}"]
    width = gf.get_cross_section(cross_section).width
    if width == source_factory().width:
        return factory.__name__
    return factory(width=width)


def get_band_component(
    component: ComponentSpec, cross_section: CrossSectionSpec
) -> gf.Component:
    """Build a Si220 component in the band of ``cross_section``.

    A container never mixes bands. A child without an explicit cross-section gets
    ``cross_section``, as rib if the child is rib-only. A child given an explicit
    cross-section keeps its type and width and only switches band. Metal and other
    non-optical children are built as they are.
    """
    if isinstance(component, gf.Component):
        return component
    current, explicit = _current_cross_section(component)
    if current is None or not _is_optical(current):
        return gf.get_component(component)
    band = get_band(cross_section)
    if explicit:
        target = _variant(current, band, is_rib(current))
    elif is_rib(current):
        target = _variant(cross_section, band, rib=True)
    else:
        target = cross_section
    return gf.get_component(component, cross_section=target)
