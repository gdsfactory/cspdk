"""Check a layout against the CORNERSTONE MPW Run 50 submission rules.

The rules are those of the 50th-call design guideline
(``references/october-2026/CORNERSTONE-50th-Call-Design-Guideline.pdf``): file
format (section 5.5), designated GDS layers (section 5.2 and Table 6), the 1 nm
manufacturing grid (section 5.5), the design area (section 5.1) and the top-cell
name (section 8). They do not replace CORNERSTONE's KLayout pre-DRC script, which
checks the geometric design rules.

Run from the command line with::

    python -m cspdk.si220.submission design.gds
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import klayout.db as kdb

# GDS layers a design may use (50th-call guideline section 5.2 and Table 6).
DESIGNATED_LAYERS: dict[tuple[int, int], str] = {
    (6, 0): "Silicon Etch 1 (grating couplers)",
    (3, 0): "Silicon Etch 2, light field (waveguides)",
    (4, 0): "Silicon Etch 2, dark field",
    (5, 0): "Silicon Etch 3 (rib protect)",
    (39, 0): "Heater filaments",
    (41, 0): "Heater contact pads",
    (43, 2): "Laser bottom contacts",
    (47, 2): "Laser trench etch",
    (99, 0): "Cell outline",
    (100, 0): "Labels",
    (140, 1): "Physical PD area",
    (143, 1): "Physical laser area",
    (144, 1): "Laser waveguide",
    (147, 1): "PD active area",
    (150, 1): "PD exclusion zone",
    (151, 1): "Laser exclusion zone",
    (161, 2): "1st graphene channel",
    (162, 2): "2nd graphene channel",
    (163, 2): "1st Au/graphene contacts",
    (164, 2): "2nd Au/graphene contacts",
    (165, 2): "1st Au/Ti metallization",
    (166, 2): "2nd Au/Ti metallization",
    (169, 1): "Graphene exclusion zone",
}

# Design areas in um (section 5.1): full and half cell.
DESIGN_AREAS: dict[str, tuple[float, float]] = {
    "full": (11470.0, 4900.0),
    "half": (5500.0, 4900.0),
}

CELL_OUTLINE = (99, 0)
GRID_UM = 0.001
FORMATS = (".gds", ".gds2", ".gdsii", ".oas")
TOP_CELL = re.compile(r"^Cell0_\w+$")


def check_submission(path: str | Path) -> list[str]:
    """Return the problems that would stop CORNERSTONE accepting the layout.

    An empty list means the layout meets the format, layer, grid, design-area and
    top-cell rules of MPW Run 50. Run CORNERSTONE's pre-DRC script as well.

    Args:
        path: GDSII or OASIS file to check.
    """
    path = Path(path)
    if path.suffix.lower() not in FORMATS:
        return [f"{path.name}: submit a GDSII (.gds) or OASIS (.oas) file"]

    layout = kdb.Layout()
    try:
        layout.read(str(path))
    except RuntimeError as error:
        return [f"{path.name}: could not read the layout ({error})"]

    tops = list(layout.top_cells())
    if len(tops) != 1:
        # repr() keeps control characters in untrusted cell names out of the output.
        names = ", ".join(repr(c.name) for c in tops) or "none"
        return [f"expected exactly one top cell, found {len(tops)}: {names}"]
    top = tops[0]
    problems: list[str] = []
    if not TOP_CELL.match(top.name):
        problems.append(
            f"top cell {top.name!r} must be named 'Cell0_<Name of Institution>'"
        )

    used: dict[tuple[int, int], int] = {}
    for index in layout.layer_indexes():
        if top.bbox_per_layer(index).empty():
            continue
        info = layout.get_info(index)
        if info.is_named():  # OASIS layers with a name but no layer number
            problems.append(
                f"layer {info.name!r} has no GDS layer number; it is not a "
                "designated MPW-50 layer"
            )
        else:
            used[info.layer, info.datatype] = index
    for layer, datatype in sorted(set(used) - set(DESIGNATED_LAYERS)):
        problems.append(f"layer {layer}/{datatype} is not a designated MPW-50 layer")

    for (layer, datatype), index in used.items():
        if _off_grid(layout, top, index):
            problems.append(f"layer {layer}/{datatype} has points off the 1 nm grid")

    problems.extend(_design_area_problems(layout, top, used))
    return problems


def _off_grid(layout: kdb.Layout, top: kdb.Cell, index: int) -> bool:
    """Whether any polygon point or text anchor on the layer is off the 1 nm grid."""
    steps = GRID_UM / layout.dbu  # database units per nm
    if steps <= 1 and abs(1 / steps - round(1 / steps)) < 1e-9:
        return False  # the database unit is a whole number of nm: always on grid
    region = kdb.Region(top.begin_shapes_rec(index))
    points = [t.position() for t in kdb.Texts(top.begin_shapes_rec(index)).each()]
    if abs(steps - round(steps)) < 1e-9:
        step = round(steps)
        return not region.grid_check(step, step).is_empty() or any(
            p.x % step or p.y % step for p in points
        )
    # A database unit that does not divide 1 nm: check every point, in nm.
    for poly in region.each():
        points.extend(poly.each_point_hull())
        for hole in range(poly.holes()):
            points.extend(poly.each_point_hole(hole))
    return any(
        abs(v / steps - round(v / steps)) > 1e-6 for p in points for v in (p.x, p.y)
    )


def _design_area_problems(
    layout: kdb.Layout, top: kdb.Cell, used: dict[tuple[int, int], int]
) -> list[str]:
    """The cell outline must be one full or half design area containing the design."""
    if CELL_OUTLINE not in used:
        return ["no cell outline on layer 99/0 (use die() or die_half())"]
    outline = kdb.Region(top.begin_shapes_rec(used[CELL_OUTLINE])).merged()
    box = outline.bbox().to_dtype(layout.dbu)
    size = (round(box.width(), 3), round(box.height(), 3))
    if (
        outline.count() != 1
        or not outline.is_box()
        or size not in DESIGN_AREAS.values()
    ):
        areas = " or ".join(f"{w:g} x {h:g} um" for w, h in DESIGN_AREAS.values())
        return [
            f"the 99/0 cell outline is {size[0]:g} x {size[1]:g} um; it must be one {areas} rectangle"
        ]

    problems = []
    for (layer, datatype), index in sorted(used.items()):
        if (layer, datatype) == CELL_OUTLINE:
            continue
        outside = kdb.Region(top.begin_shapes_rec(index)) - outline
        texts_outside = kdb.Texts(top.begin_shapes_rec(index)).not_interacting(outline)
        if not outside.is_empty() or not texts_outside.is_empty():
            problems.append(
                f"layer {layer}/{datatype} extends outside the cell outline"
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    """Check the files given on the command line; exit 1 if any has a problem."""
    paths = sys.argv[1:] if argv is None else argv
    if not paths:
        print("usage: python -m cspdk.si220.submission design.gds [...]")
        return 2
    failed = False
    for path in paths:
        problems = check_submission(path)
        failed |= bool(problems)
        status = f"{len(problems)} problem(s)" if problems else "OK"
        print(f"{path}: {status}")
        for problem in problems:
            print(f"  - {problem}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
