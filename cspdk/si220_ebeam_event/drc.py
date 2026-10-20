"""Design-rule check of the e-beam event, with klayout.db (no deck).

Rules (see README.md):

* ``WG.W``/``.S``, ``EXPOSURE.W``/``.S``, ``GRA.W``/``.S``: 200 nm minimum
  feature and gap on 3/0, 4/0 (as drawn) and 6/0;
* ``MERGED.W``/``.S``: the same on the merged exposure (4/0);
* ``EXPOSURE.A``: merged exposure at most 2,000,000 um^2;
* ``DESIGN.SIZE``: the design fits in 6 mm (east-west) x 3 mm;
* ``TRENCH_TMP.R``: nothing drawn on 3/1 (a cspdk check).

Width uses the projection metric, space the Euclidean one, so corner-to-corner
gaps count.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice

import gdsfactory as gf
import klayout.db as kdb

from cspdk.si220_ebeam_event.mask import IGNORE_ANGLE, exposure_region
from cspdk.si220_ebeam_event.tech import LAYER, TECH

__all__ = ["Violation", "check"]

MAX_LOCATIONS = 10


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
    for rule, kind, minimum, run_check, metrics in (
        ("W", "feature", TECH.min_width, region.width_check, kdb.Metrics.Projection),
        ("S", "gap", TECH.min_space, region.space_check, kdb.Metrics.Euclidian),
    ):
        markers = run_check(
            round(minimum / dbu), metrics=metrics, ignore_angle=IGNORE_ANGLE
        )
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
    if not extent.empty() and (
        extent.width() * dbu > width or extent.height() * dbu > height
    ):
        violations.append(
            Violation(
                rule="DESIGN.SIZE",
                layer=merged,
                description=f"design larger than {width:g} x {height:g} um",
                count=1,
                locations=_locations(kdb.Region(extent), dbu),
            )
        )

    reserved = component.get_region(LAYER.TRENCH_TMP, merge=True)
    if not reserved.is_empty():
        violations.append(
            Violation(
                rule="TRENCH_TMP.R",
                layer=_layer_name("TRENCH_TMP"),
                description="reserved for post-processing, must be empty",
                count=reserved.count(),
                locations=_locations(reserved, dbu),
            )
        )
    return violations
