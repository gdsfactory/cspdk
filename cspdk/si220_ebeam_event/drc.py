"""Design-rule check of the CORNERSTONE e-beam event (pure klayout.db, no deck).

Rules (event spec, 17 Oct 2026, provisional):

* ``<LAYER>.W``: minimum feature 200 nm on WG (3/0), EXPOSURE (4/0, as drawn)
  and GRA (6/0);
* ``<LAYER>.S``: minimum gap 200 nm on the same layers;
* ``EXPOSURE.A``: the post-processed exposure on 4/0
  (:func:`cspdk.si220_ebeam_event.mask.exposure_region`) is at most
  2,000,000 um^2;
* ``DESIGN.SIZE``: the post-processed exposure fits in the 6 mm x 3 mm design
  area of one submission (6 mm east-west, 3 mm north-south);
* ``TRENCH_TMP.R``: nothing is drawn on 3/1, which the foundry uses as a
  temporary layer during post-processing (a cspdk check, not a foundry rule).

Width uses klayout's projection metric: only edges that face each other with
overlapping projections are measured, perpendicular to the edges. Space uses
the Euclidean metric, so a gap between two corners on a diagonal is measured
too. In both, edges meeting at 80 degrees or more are not compared with each
other, which avoids false errors on discretised curves; acute spikes
(< 80 degrees) are still caught. Width/space run on the drawn layers; the spec
does not give width/space rules for the post-processed 4/0.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice

import gdsfactory as gf
import klayout.db as kdb

from cspdk.si220_ebeam_event.mask import exposure_region, layer_region
from cspdk.si220_ebeam_event.tech import LAYER, TECH

__all__ = ["Violation", "check"]

# Layers with width/space rules: name -> layer.
CHECKED_LAYERS = {"WG": LAYER.WG, "EXPOSURE": LAYER.EXPOSURE, "GRA": LAYER.GRA}
MAX_LOCATIONS = 10
# Edge pairs meeting at this angle or more are not measured: rectangle corners
# (90 deg) and the slightly-under-90 deg corners where a discretised arc meets
# a port face would otherwise be flagged as zero-width features.
IGNORE_ANGLE = 80


@dataclass(frozen=True)
class Violation:
    """One violated rule, with the places it is violated.

    Attributes:
        rule: rule name, e.g. ``"WG.S"``.
        layer: layer name and number, e.g. ``"WG 3/0"``.
        description: the rule in words.
        count: number of violating markers (1 for the area rule).
        locations: (x, y) centres in um of up to the first 10 markers.
        value: the measured value for the area rule (um^2), else None.
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


def check(component: gf.Component) -> list[Violation]:
    """Return the e-beam event design-rule violations of ``component``.

    An empty list means the layout passes. Hierarchy is flattened.

    Args:
        component: the design to check.
    """
    dbu = component.kcl.dbu
    violations: list[Violation] = []
    for name, layer in CHECKED_LAYERS.items():
        region = layer_region(component, layer)
        for rule, kind, minimum, run_check, metrics in (
            (
                "W",
                "feature",
                TECH.min_width,
                region.width_check,
                kdb.Metrics.Projection,
            ),
            ("S", "gap", TECH.min_space, region.space_check, kdb.Metrics.Euclidian),
        ):
            markers = run_check(
                round(minimum / dbu), metrics=metrics, ignore_angle=IGNORE_ANGLE
            )
            if markers.is_empty():
                continue
            violations.append(
                Violation(
                    rule=f"{name}.{rule}",
                    layer=_layer_name(name),
                    description=f"{kind} < {minimum * 1e3:g} nm",
                    count=markers.count(),
                    locations=_locations(markers, dbu),
                )
            )

    exposure = exposure_region(component)
    area = exposure.area() * dbu**2
    if area > TECH.max_exposure_area:
        violations.append(
            Violation(
                rule="EXPOSURE.A",
                layer=_layer_name("EXPOSURE") + " post-processed",
                description=f"exposure area > {TECH.max_exposure_area:,.0f} um^2",
                count=1,
                value=area,
            )
        )

    # The post-processed exposure contains every drawn core (sized by 5 um), so
    # its bounding box is the extent of the design.
    width, height = TECH.die_size
    extent = exposure.bbox()
    if not extent.empty() and (
        extent.width() * dbu > width or extent.height() * dbu > height
    ):
        violations.append(
            Violation(
                rule="DESIGN.SIZE",
                layer=_layer_name("EXPOSURE") + " post-processed",
                description=(
                    f"design larger than the {width:g} x {height:g} um design area"
                ),
                count=1,
                locations=_locations(kdb.Region(extent), dbu),
            )
        )

    reserved = layer_region(component, LAYER.TRENCH_TMP)
    if not reserved.is_empty():
        violations.append(
            Violation(
                rule="TRENCH_TMP.R",
                layer=_layer_name("TRENCH_TMP"),
                description="layer reserved for post-processing, must be empty",
                count=reserved.count(),
                locations=_locations(reserved, dbu),
            )
        )
    return violations
