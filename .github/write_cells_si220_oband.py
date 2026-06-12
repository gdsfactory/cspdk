"""Write cell docs as Markdown with kwasm viewers."""

import base64
import inspect
import traceback

import kwasm.embed
import matplotlib
import matplotlib.pyplot as plt
from gdsfactory.serialization import clean_value_json

matplotlib.use("Agg")

from cspdk.si220.oband import PDK
from cspdk.si220.oband import _cells as cells
from cspdk.si220.oband.config import PATH

PDK.activate()

filepath = PATH.repo / "docs" / "cells_si220_oband.md"
kwasm_dir = PATH.repo / "docs" / "kwasm"
gds_dir = kwasm_dir / "gds"

skip = {}
skip_plot: tuple[str, ...] = ("",)
skip_settings: tuple[str, ...] = ()


def _setup_kwasm_viewer() -> None:
    gds_dir.mkdir(parents=True, exist_ok=True)
    viewer_path = kwasm_dir / "viewer.html"
    if viewer_path.exists():
        return
    template = kwasm.embed._read_artifacts()
    template = template.replace("KWASM_GDS_B64", "")
    lyp_path = PATH.lyp
    if lyp_path.exists():
        lyp_b64 = base64.b64encode(lyp_path.read_bytes()).decode("ascii")
        template = template.replace("KWASM_LYP_B64", lyp_b64)
    else:
        template = template.replace("KWASM_LYP_B64", "")
    template = template.replace("KWASM_LYRDB_B64", "")
    template = template.replace("KWASM_NETLIST_B64", "")
    viewer_path.write_text(template)


def _write_gds(name: str) -> bool:
    try:
        sig = inspect.signature(cells[name])
        defaults = {}
        for p in sig.parameters:
            v = sig.parameters[p].default
            if isinstance(v, int | float | str | tuple):
                defaults[p] = v
        c = cells[name](**defaults)
        c.write(str(gds_dir / f"{name}.gds"))
        c.plot()
        plt.savefig(str(gds_dir / f"{name}.png"), dpi=150, bbox_inches="tight")
        plt.close("all")
    except Exception:
        traceback.print_exc()
        plt.close("all")
        return False
    else:
        return True


_setup_kwasm_viewer()

with open(filepath, "w+") as f:
    f.write("# Cells Si SOI 220nm Oband\n\n")

    for name in sorted(cells.keys()):
        if name in skip or name.startswith("_"):
            continue
        print(name)
        sig = inspect.signature(cells[name])
        kwargs = ", ".join(
            [
                f"{p}={clean_value_json(sig.parameters[p].default)!r}"
                for p in sig.parameters
                if isinstance(sig.parameters[p].default, int | float | str | tuple)
                and p not in skip_settings
            ]
        )
        f.write(f"## {name}\n\n")
        f.write(f"::: cspdk.si220.oband.cells.{name}\n   :noindex:\n\n")
        if name not in skip_plot:
            has_gds = _write_gds(name)
            if has_gds:
                f.write('=== "Static"\n\n')
                f.write(f"    ![{name}](kwasm/gds/{name}.png)\n\n")
                f.write('=== "Dynamic"\n\n')
                f.write(
                    f'    <iframe src="kwasm/viewer.html?url=gds/{name}.gds"'
                    f' loading="lazy" width="100%" height="400"'
                    f' style="border:none"></iframe>\n\n'
                )
            f.write("```python\n")
            f.write("import cspdk\n\n")
            f.write(f"c = cspdk.si220.oband.cells.{name}({kwargs}).copy()\n")
            f.write("c.draw_ports()\n")
            f.write("c.plot()\n")
            f.write("```\n\n")

print(f"Wrote {filepath}")
