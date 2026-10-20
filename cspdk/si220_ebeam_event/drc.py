"""Design-rule check of the e-beam event, with klayout.db (no deck).

Rules (see README.md):

* ``WG.W``/``.S``, ``EXPOSURE.W``/``.S``, ``GRA.W``/``.S``: 200 nm minimum
  feature and gap on 3/0, 4/0 (as drawn) and 6/0;
* ``MERGED.W``/``.S``: the same on the merged exposure (4/0);
* ``EXPOSURE.A``: merged exposure at most 2,000,000 um^2;
* ``DESIGN.SIZE``: the design fits in 6 mm (east-west) x 3 mm, and inside
  FLOORPLAN (99/0) when one is drawn;

and cspdk checks that the foundry would ignore or reject:

* ``TRENCH_TMP.R``: nothing drawn on 3/1;
* ``ROUTING.E``: no routing-error markers (1000/0) from failed routes;
* ``LAYER.U``: no shapes on other layers (they would not be written).

Width uses the projection metric; space is Euclidean, so corner-to-corner gaps
count. Edges meeting at 80 degrees or more are not compared within a polygon
(curves would give false errors), but gaps between polygons always are.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import islice

import gdsfactory as gf
import klayout.db as kdb

from cspdk.si220_ebeam_event.mask import IGNORE_ANGLE, exposure_region
from cspdk.si220_ebeam_event.tech import LAYER, TECH

__all__ = ["Violation", "check"]

MAX_LOCATIONS = 10
# Layers that may hold shapes; anything else is reported as LAYER.U.
ALLOWED_LAYERS = (
    LAYER.WG,
    LAYER.EXPOSURE,
    LAYER.GRA,
    LAYER.TRENCH_TMP,
    LAYER.FLOORPLAN,
    LAYER.LABEL_SETTINGS,
    LAYER.LABEL_INSTANCE,
    LAYER.routing_error_marker,
)


@dataclass(frozen=True)
class Violation:
    """One violated rule.

    Attributes:
        rule: rule name, e.g. ``"WG.S"``.
        layer: layer name and number, e.g. ``"WG 3/0"``.
        description: the rule in words.
        count: number of markers.
        locations: (x, y) centres in um of up to the first 10 markers.
        value: the measured area for ``EXPOSURE.A`` (um^2), else None.
    """

    rule: str
    layer: str
    description: str
    count: int
    locations: tuple[tuple[float, float], ...] = ()
    value: float | None = None

    def __str__(self) -> str:
        """Readable one-line summary."""
        where = ", ".join(f"({x:g}, {y:g})" for x, y in self.locations)
        text = f"{self.rule} [{self.layer}] {self.description}: {self.count} marker(s)"
        if self.value is not None:
            text += f", measured {self.value:,.0f}"
        return f"{text} at {where}" if where else text


def _layer_name(name: str) -> str:
    layer = getattr(LAYER, name)
    return f"{name} {layer.layer}/{layer.datatype}"


def _locations(markers: kdb.EdgePairs | kdb.Region, dbu: float) -> tuple:
    centres = (marker.bbox().center() for marker in markers.each())
    return tuple(
        (round(c.x * dbu, 3), round(c.y * dbu, 3))
        for c in islice(centres, MAX_LOCATIONS)
    )


def _width_space(
    region: kdb.Region, name: str, layer: str, dbu: float
) -> list[Violation]:
    violations = []
    width = math.ceil(TECH.min_width / dbu - 1e-9)
    space = math.ceil(TECH.min_space / dbu - 1e-9)
    euclidian = kdb.Metrics.Euclidian
    for rule, kind, minimum, markers in (
        (
            "W",
            "feature",
            TECH.min_width,
            region.width_check(
                width, metrics=kdb.Metrics.Projection, ignore_angle=IGNORE_ANGLE
            ),
        ),
        (
            "S",
            "gap",
            TECH.min_space,
            region.notch_check(space, metrics=euclidian, ignore_angle=IGNORE_ANGLE)
            + region.isolated_check(space, metrics=euclidian),
        ),
    ):
        if not markers.is_empty():
            violations.append(
                Violation(
                    rule=f"{name}.{rule}",
                    layer=layer,
                    description=f"{kind} < {minimum * 1e3:g} nm",
                    count=markers.count(),
                    locations=_locations(markers, dbu),
                )
            )
    return violations


def check(
    component: gf.Component, exposure: kdb.Region | None = None
) -> list[Violation]:
    """Return the event design-rule violations of ``component``; empty if it passes.

    Args:
        component: the design.
        exposure: its :func:`~cspdk.si220_ebeam_event.mask.exposure_region`, if
            already computed.
    """
    dbu = component.kcl.dbu
    violations: list[Violation] = []
    for name in ("WG", "EXPOSURE", "GRA"):
        region = component.get_region(getattr(LAYER, name), merge=True)
        violations += _width_space(region, name, _layer_name(name), dbu)

    if exposure is None:
        exposure = exposure_region(component)
    merged = _layer_name("EXPOSURE") + " merged"
    violations += _width_space(exposure, "MERGED", merged, dbu)

    area = exposure.area() * dbu**2
    if area > TECH.max_exposure_area:
        violations.append(
            Violation(
                rule="EXPOSURE.A",
                layer=merged,
                description=f"exposure area > {TECH.max_exposure_area:,.0f} um^2",
                count=1,
                value=area,
            )
        )

    # The exposure contains every core sized by 5 um, so its bbox is the design's.
    width, height = TECH.die_size
    extent = exposure.bbox()
    floorplan = component.get_region(LAYER.FLOORPLAN, merge=True)
    if not extent.empty() and (
        extent.width() * dbu > width
        or extent.height() * dbu > height
        or not (floorplan.is_empty() or (exposure - floorplan).is_empty())
    ):
        violations.append(
            Violation(
                rule="DESIGN.SIZE",
                layer=merged,
                description=(
                    f"design larger than {width:g} x {height:g} um or outside FLOORPLAN"
                ),
                count=1,
                locations=_locations(kdb.Region(extent), dbu),
            )
        )

    for rule, name, description in (
        ("TRENCH_TMP.R", "TRENCH_TMP", "reserved for post-processing, must be empty"),
        ("ROUTING.E", "routing_error_marker", "failed route"),
    ):
        region = component.get_region(getattr(LAYER, name), merge=True)
        if not region.is_empty():
            violations.append(
                Violation(
                    rule=rule,
                    layer=_layer_name(name),
                    description=description,
                    count=region.count(),
                    locations=_locations(region, dbu),
                )
            )

    allowed = {gf.get_layer(layer) for layer in ALLOWED_LAYERS}
    for index in component.kcl.layer_indexes():
        if index in allowed:
            continue
        region = kdb.Region(component.begin_shapes_rec(index))
        if not region.is_empty():
            info = component.kcl.get_info(index)
            violations.append(
                Violation(
                    rule="LAYER.U",
                    layer=f"{info.layer}/{info.datatype}",
                    description="shapes on a layer that is not written",
                    count=region.count(),
                    locations=_locations(region, dbu),
                )
            )
    return violations
