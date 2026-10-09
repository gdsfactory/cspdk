"""Write a flavour's cell docs as Markdown with kwasm viewers.

Usage: python .github/write_cells.py <flavour>, e.g. si220 or sin300.
"""

import base64
import importlib
import inspect
import sys
import traceback

import kwasm.embed
import matplotlib
import matplotlib.pyplot as plt
from gdsfactory.serialization import clean_value_json

matplotlib.use("Agg")

TITLES = {
    "si220": "Cells Si SOI 220nm C-band and O-band",
    "si340": "Cells Si SOI 340nm",
    "si500": "Cells Si SOI 500nm",
    "sin200": "Cells SiN200 visible",
    "sin300": "Cells SiN300",
    "si_sus": "Cells suspended Si",
    "ge_on_si": "Cells Ge-on-Si",
}

flavour = sys.argv[1]
if flavour not in TITLES:
    sys.exit(f"unknown flavour {flavour!r}; expected one of {sorted(TITLES)}")

pdk_module = importlib.import_module(f"cspdk.{flavour}")
PATH = importlib.import_module(f"cspdk.{flavour}.config").PATH
cells = pdk_module._cells
pdk_module.PDK.activate()

filepath = PATH.repo / "docs" / f"cells_{flavour}.md"
kwasm_dir = PATH.repo / "docs" / "kwasm"
# One folder per flavour: cell names repeat across flavours.
gds_dir = kwasm_dir / "gds" / flavour


def _setup_kwasm_viewer() -> None:
    gds_dir.mkdir(parents=True, exist_ok=True)
    viewer_path = kwasm_dir / "viewer.html"
    if viewer_path.exists():
        return
    lyp_path = getattr(PATH, "lyp", None)
    lyp_b64 = ""
    if lyp_path is not None and lyp_path.exists():
        lyp_b64 = base64.b64encode(lyp_path.read_bytes()).decode("ascii")
    template = (
        kwasm.embed._read_artifacts()
        .replace("KWASM_GDS_B64", "")
        .replace("KWASM_LYP_B64", lyp_b64)
        .replace("KWASM_LYRDB_B64", "")
        .replace("KWASM_NETLIST_B64", "")
    )
    viewer_path.write_text(template)


def _simple_defaults(name: str) -> dict:
    """Parameter defaults that can be written as Python literals."""
    params = inspect.signature(cells[name]).parameters
    return {
        p: v.default
        for p, v in params.items()
        if isinstance(v.default, int | float | str | tuple)
    }


def _write_gds(name: str, defaults: dict) -> bool:
    try:
        c = cells[name](**defaults)
        c.write(str(gds_dir / f"{name}.gds"))
        c.plot()
        plt.savefig(str(gds_dir / f"{name}.png"), dpi=150, bbox_inches="tight")
    except Exception:
        traceback.print_exc()
        return False
    finally:
        plt.close("all")
    return True


_setup_kwasm_viewer()

with open(filepath, "w") as f:
    f.write(f"# {TITLES[flavour]}\n\n")

    for name in sorted(cells):
        if name.startswith("_"):
            continue
        print(name)
        defaults = _simple_defaults(name)
        kwargs = ", ".join(f"{p}={clean_value_json(v)!r}" for p, v in defaults.items())
        f.write(f"## {name}\n\n")
        f.write(f"::: cspdk.{flavour}.cells.{name}\n   :noindex:\n\n")
        if _write_gds(name, defaults):
            f.write('=== "Static"\n\n')
            f.write(f"    ![{name}](kwasm/gds/{flavour}/{name}.png)\n\n")
            f.write('=== "Dynamic"\n\n')
            f.write(
                f'    <iframe src="kwasm/viewer.html?url=gds/{flavour}/{name}.gds"'
                f' loading="lazy" width="100%" height="400"'
                f' style="border:none"></iframe>\n\n'
            )
        f.write("```python\n")
        f.write(f"from cspdk.{flavour} import cells, PDK\n\n")
        f.write("PDK.activate()\n\n")
        f.write(f"c = cells.{name}({kwargs})\n")
        f.write("c.draw_ports()\n")
        f.write("c.plot()\n")
        f.write("```\n\n")

print(f"Wrote {filepath}")
