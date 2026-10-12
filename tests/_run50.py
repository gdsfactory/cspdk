"""The Cornerstone MPW Run-50 GDSII template, shared by the si220 Run-50 tests."""

from __future__ import annotations

import functools

import klayout.db as kdb

from cspdk.si220.config import PATH

TEMPLATE = (
    PATH.repo
    / "references"
    / "october-2026"
    / "CORNERSTONE-MPW-Run-50-GDSII-Template.gds"
)


@functools.cache
def template() -> kdb.Layout:
    """The Run-50 template layout, read once per test session. Do not modify it."""
    layout = kdb.Layout()
    layout.read(str(TEMPLATE))
    return layout


def template_region(cell: str, layer: tuple[int, int]) -> kdb.Region:
    """The merged shapes of template cell ``cell`` on ``layer``."""
    layout = template()
    index = layout.find_layer(*layer)
    if index is None:
        return kdb.Region()
    return kdb.Region(layout.cell(cell).begin_shapes_rec(index)).merged()
