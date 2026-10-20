"""Check and post-process submitted GDS files of the CORNERSTONE e-beam event.

For each GDS: run the event design rules (:func:`cspdk.si220_ebeam_event.check`),
report the post-processed exposure area against its 2,000,000 um^2 budget and,
with ``--output-dir``, write the post-processed layout
(:func:`cspdk.si220_ebeam_event.post_process`) as ``<name>_exposure.gds``.
The exit status is 1 if any file breaks a rule, else 0::

    python -m cspdk.si220_ebeam_event.tapeout design1.gds design2.gds -o out/
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import gdsfactory as gf

from cspdk.si220_ebeam_event import PDK
from cspdk.si220_ebeam_event.drc import check
from cspdk.si220_ebeam_event.mask import exposure_region, post_process
from cspdk.si220_ebeam_event.tech import TECH

__all__ = ["main"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cspdk.si220_ebeam_event.tapeout",
        description=(
            "Check and post-process GDS files for the CORNERSTONE e-beam event "
            "(EVENT-ONLY rules, provisional)."
        ),
    )
    parser.add_argument("gds", nargs="+", type=Path, help="GDS files to check")
    parser.add_argument(
        "--cell",
        default=None,
        help="top cell to check (needed when a file has several top cells)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="write each post-processed layout here as <name>_exposure.gds",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Check, report and optionally post-process GDS files; return the exit status.

    Args:
        argv: command-line arguments (defaults to ``sys.argv[1:]``).
    """
    args = _parser().parse_args(argv)
    PDK.activate()
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    failed = False
    for path in args.gds:
        component = gf.import_gds(path, cellname=args.cell)
        violations = check(component)
        area = exposure_region(component).area() * component.kcl.dbu**2
        share = area / TECH.max_exposure_area
        status = "FAIL" if violations else "PASS"
        print(
            f"{status} {path} (cell {component.name}): exposure {area:,.0f} um^2, "
            f"{share:.1%} of {TECH.max_exposure_area:,.0f}"
        )
        for violation in violations:
            print(f"  {violation}")
        if args.output_dir is not None:
            output = args.output_dir / f"{path.stem}_exposure.gds"
            exposed = post_process(component)
            exposed.name = output.stem
            exposed.write_gds(output)
            print(f"  wrote {output}")
        failed = failed or bool(violations)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
