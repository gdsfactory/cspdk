"""Mask preparation: the foundry's merge of the drawn layers into the exposure (4/0).

1. Size WG (3/0) by +5 um into TRENCH_TMP (3/1).
2. Merge EXPOSURE (4/0) + GRA (6/0) + (3/1 - 3/0) into EXPOSURE (4/0).

The 200 nm rules also apply to the merged 4/0. Plain sizing breaks them on the
outer edge of 3/1, at least 5 um from any core: it leaves 1-2 nm jogs on arcs,
and where two trenches meet at a shallow angle (a ring and its bus) the Si
between them narrows to a point. So 3/1 is smoothed, and gaps in it narrower
than 210 nm are filled (etched). CORNERSTONE agreed to this. Gaps formed by
drawn 4/0 or 6/0 are left for ``check`` to report.
"""

from __future__ import annotations

import gdsfactory as gf
import klayout.db as kdb

from cspdk.si220_ebeam_event.tech import LAYER, TECH

__all__ = ["exposure_region", "post_process"]

SMOOTH = 0.005  # um, removes the 1-2 nm sizing jogs
FILL_MARGIN = 0.01  # um, added to the minimum space for the gap fill
# Edges meeting at this angle or more are not compared: rectangle corners, and
# the near-90 degree corners where an arc meets a port face.
IGNORE_ANGLE = 80


def _trench_tmp(wg: kdb.Region, dbu: float) -> kdb.Region:
    """Return WG sized by +5 um (3/1), smoothed and with narrow gaps filled."""
    grown = wg.sized(round(TECH.trench_width / dbu))
    smooth = round(SMOOTH / dbu)
    grown = grown.smoothed(smooth, False)
    fill_below = round((TECH.min_space + FILL_MARGIN) / dbu)
    for _ in range(3):  # filling can leave a new, smaller gap
        gaps = grown.space_check(
            fill_below, metrics=kdb.Metrics.Euclidian, ignore_angle=IGNORE_ANGLE
        )
        if gaps.is_empty():
            break
        grown = (grown + gaps.polygons()).merged()
    # The fill polygons can meet the trench edge at a point: a 1 dbu opening
    # and closing removes the zero-width spikes and notches this leaves.
    return grown.sized(-1).sized(1).sized(1).sized(-1).smoothed(smooth, False)


def exposure_region(component: gf.Component) -> kdb.Region:
    """Return the merged exposure (4/0) of ``component``, in database units."""
    wg = component.get_region(LAYER.WG, merge=True)
    exposure = (
        component.get_region(LAYER.EXPOSURE)
        + component.get_region(LAYER.GRA)
        + (_trench_tmp(wg, component.kcl.dbu) - wg)
    )
    return exposure.merged()


def post_process(
    component: gf.Component, exposure: kdb.Region | None = None
) -> gf.Component:
    """Return a flat copy of ``component`` with the merged exposure on 4/0.

    3/1 and texts are dropped; ports are kept.

    Args:
        component: the design.
        exposure: its :func:`exposure_region`, if already computed.
    """
    if exposure is None:
        exposure = exposure_region(component)
    c = gf.Component()
    skip = {gf.get_layer(LAYER.EXPOSURE), gf.get_layer(LAYER.TRENCH_TMP)}
    for index in component.kcl.layer_indexes():
        if index not in skip:
            region = kdb.Region(component.begin_shapes_rec(index))
            if not region.is_empty():
                c.shapes(index).insert(region)
    c.shapes(gf.get_layer(LAYER.EXPOSURE)).insert(exposure)
    c.add_ports(component.ports)
    return c
