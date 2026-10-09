"""Extract the Run 50 fixed cells of the si220 PDK from the foundry template.

Writes each foundry cell of the CORNERSTONE MPW Run 50 GDSII template, unchanged
and under its own name, to ``cspdk/si220/gds/<cell name>.gds``, which the fixed
cells in ``cspdk/si220/cells/fixed.py`` import. Run from the repository root::

    uv run python scripts/extract_si220_run50.py
"""

from __future__ import annotations

import pathlib

import klayout.db as kdb

REPO = pathlib.Path(__file__).resolve().parent.parent
TEMPLATE = (
    REPO / "references" / "october-2026" / "CORNERSTONE-MPW-Run-50-GDSII-Template.gds"
)
OUT = REPO / "cspdk" / "si220" / "gds"

_GRAPHENE = [
    f"SOI220_cband_TE_strip_SLG_{device}{suffix}"
    for device in (
        "Bolometer",
        "Bolometer_metallic",
        "Bolometer_no_gate",
        "EAM",
        "PTE",
        "PTE_metallic",
    )
    for suffix in ("", "_with_GC")
]
_LASERS = [
    f"SOI220_{size}_LD_DC1550HP01_Integrated{suffix}"
    for size in ("Full", "Half")
    for suffix in ("", "_or_GC", "_with_GC")
]

CELLS = [
    "SOI220_cband_TE_strip_1x2_Ysplitter",
    "SOI220_oband_TE_strip_1x2_Ysplitter",
    "SOI220_cband_TE_strip_apodized_GC_focusing_v1p0",
    *_GRAPHENE,
    "SOI220_PD_FCI300B1_Integrated",
    "SOI220_PD_FCI300B1_Plain",
    *_LASERS,
]


def extract(name: str, template: kdb.Layout, out: pathlib.Path = OUT) -> pathlib.Path:
    """Write the template cell ``name`` and its children to ``out/<name>.gds``.

    Args:
        name: name of a top cell of the template.
        template: the template layout.
        out: directory to write the GDS file to.
    """
    source = template.cell(name)
    if source is None:
        raise ValueError(f"{name!r} is not a cell of {TEMPLATE.name}")
    layout = kdb.Layout()
    layout.dbu = template.dbu
    target = layout.create_cell(name)
    target.copy_tree(source)
    path = out / f"{name}.gds"
    options = kdb.SaveLayoutOptions()
    options.write_context_info = False
    options.gds2_write_timestamps = False
    layout.write(str(path), options)
    return path


def main() -> None:
    """Extract every Run 50 fixed cell."""
    template = kdb.Layout()
    template.read(str(TEMPLATE))
    for name in CELLS:
        print(extract(name, template).relative_to(REPO))


if __name__ == "__main__":
    main()
