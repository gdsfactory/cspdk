"""Check GDS files against the e-beam event rules and write their merged exposure.

    python -m cspdk.si220_ebeam_event.tapeout design1.gds design2.gds -o out/

Prints PASS/FAIL and the exposure area per file, writes
``<name>_exposure.gds`` with ``-o``, and exits 1 if any file fails.
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
        description="Check GDS files against the e-beam event rules.",
    )
    parser.add_argument("gds", nargs="+", type=Path, help="GDS files to check")
    parser.add_argument(
        "--cell",
        default=None,
        help="top cell, for files with several",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="write <name>_exposure.gds files here",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI on ``argv`` (default ``sys.argv[1:]``); return the exit status."""
    args = _parser().parse_args(argv)
    PDK.activate()
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    failed = False
    for path in args.gds:
        component = gf.import_gds(path, cellname=args.cell)
        exposure = exposure_region(component)
        violations = check(component, exposure)
        area = exposure.area() * component.kcl.dbu**2
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
            exposed = post_process(component, exposure)
            exposed.name = output.stem
            exposed.write_gds(output)
            print(f"  wrote {output}")
        failed = failed or bool(violations)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
