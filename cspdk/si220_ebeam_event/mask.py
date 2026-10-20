"""Mask preparation of the CORNERSTONE e-beam event (foundry post-processing).

The event spec (17 Oct 2026) prepares the single e-beam exposure on 4/0 as:

1. size WG (3/0) by +5 um into the temporary layer TRENCH_TMP (3/1);
2. merge EXPOSURE (4/0) + GRA (6/0) + (3/1 - 3/0) into EXPOSURE (4/0).

The result is what is written and etched (120 nm). :func:`exposure_region`
returns it as a klayout region; :func:`post_process` returns a flat component
with it on 4/0, so designers can see and check what will be exposed.

The 200 nm rules apply to this merged 4/0 too (CORNERSTONE, Oct 2026), and
plain sizing breaks them in two places, both on the outer edge of the grown
3/1, at least 5 um from any core:

* sizing a discretised arc leaves 1-2 nm jogs, which are smoothed away;
* where two trenches meet at a shallow angle (a ring and its bus), the unetched
  Si between them narrows to a point; gaps in 3/1 narrower than the minimum
  space (plus a 10 nm margin) are filled, so they are etched.

CORNERSTONE agreed that cspdk may prepare the exposure this way. Gaps formed
by drawn 4/0 or 6/0 are left alone, for ``check`` to report.
"""

from __future__ import annotations

import gdsfactory as gf
import klayout.db as kdb
from gdsfactory.typings import LayerSpec

from cspdk.si220_ebeam_event.tech import LAYER, TECH

__all__ = ["exposure_region", "layer_region", "post_process"]

# Smoothing tolerance for the sizing jogs and margin on the gap fill, in um.
SMOOTH = 0.005
FILL_MARGIN = 0.01
# Edge pairs meeting at this angle or more are not measured: rectangle corners
# (90 deg) and the slightly-under-90 deg corners where a discretised arc meets
# a port face would otherwise be flagged as zero-width features.
IGNORE_ANGLE = 80


def layer_region(component: gf.Component, layer: LayerSpec) -> kdb.Region:
    """Return the flattened, merged shapes of ``component`` on ``layer``.

    Args:
        component: the component, including all its sub-cells.
        layer: the layer to collect.
    """
    index = gf.get_layer(layer)
    region = kdb.Region(component.begin_shapes_rec(index))
    region.merge()
    return region


def exposure_region(component: gf.Component) -> kdb.Region:
    """Return the post-processed e-beam exposure (4/0) of ``component``.

    EXPOSURE (4/0) + GRA (6/0) + (WG sized by +5 um - WG), merged, in
    database units of ``component``'s layout. The sized WG is cleaned first
    (see the module docstring).

    Args:
        component: the design to post-process.
    """
    dbu = component.kcl.dbu
    wg = layer_region(component, LAYER.WG)
    trench_tmp = wg.sized(round(TECH.trench_width / dbu))  # 3/1
    trench_tmp = trench_tmp.smoothed(round(SMOOTH / dbu), False)
    fill_below = round((TECH.min_space + FILL_MARGIN) / dbu)
    for _ in range(3):  # a fill can expose a new gap; one pass is the usual case
        gaps = trench_tmp.space_check(
            fill_below, metrics=kdb.Metrics.Euclidian, ignore_angle=IGNORE_ANGLE
        )
        if gaps.is_empty():
            break
        trench_tmp = (trench_tmp + gaps.polygons()).merged()
    exposure = (
        layer_region(component, LAYER.EXPOSURE)
        + layer_region(component, LAYER.GRA)
        + (trench_tmp - wg)
    )
    exposure.merge()
    return exposure


def post_process(component: gf.Component) -> gf.Component:
    """Return a flat copy of ``component`` with the post-processed exposure on 4/0.

    EXPOSURE (4/0) holds :func:`exposure_region`. Every other layer is copied
    unchanged (flattened polygons; texts are dropped), except the temporary
    TRENCH_TMP (3/1), which the foundry reserves and which is left out. The
    ports of ``component`` are kept.

    Args:
        component: the design to post-process.
    """
    c = gf.Component()
    skip = {gf.get_layer(LAYER.EXPOSURE), gf.get_layer(LAYER.TRENCH_TMP)}
    for index in component.kcl.layer_indexes():
        if index in skip:
            continue
        region = kdb.Region(component.begin_shapes_rec(index))
        if not region.is_empty():
            c.shapes(index).insert(region)
    c.shapes(gf.get_layer(LAYER.EXPOSURE)).insert(exposure_region(component))
    c.add_ports(component.ports)
    return c
