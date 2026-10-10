"""Write a flavour's cell docs as Markdown with kwasm viewers.

Usage: python .github/write_cells.py <flavour>, e.g. si220 or sin300.
"""

import importlib
import inspect
import sys
import tempfile
import traceback
from pathlib import Path

import kwasm
import kwasm.embed
import matplotlib
import matplotlib.pyplot as plt
from gdsfactory.serialization import clean_value_json
from gdsfactory.technology import LayerViews

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
# One viewer page per flavour, embedding that flavour's KLayout layer
# properties; all pages share one copy of the kwasm script.
viewer_name = f"viewer_{flavour}.html"
kwasm_script = f"kwasm-{kwasm.__version__}.js"
lyp_path = PATH.module / "klayout" / "layers.lyp"

# Host page: fetch the GDS named by ?url=, then mount kwasm with the layer
# properties (the same options kwasm.embed builds for a single component).
VIEWER_HTML = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>kwasm</title><style>html,body,#viewer{height:100%;margin:0}</style>
</head><body><div id="viewer"></div>
<script src="KWASM_SCRIPT"></script>
<script>
const lyp = KWASM_LYP;
const url = new URLSearchParams(location.search).get("url");
fetch(url)
  .then((response) => {
    if (!response.ok) throw new Error(url + ": HTTP " + response.status);
    return response.arrayBuffer();
  })
  .then((buffer) => {
    const bytes = new Uint8Array(buffer);
    let binary = "";
    for (let i = 0; i < bytes.length; i += 0x8000) {
      binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
    }
    const options = { gds: btoa(binary), layers: true, lyp };
    Kwasm.mount(document.getElementById("viewer"), options);
  });
</script></body></html>
"""


def _setup_kwasm_viewer() -> None:
    """Write the kwasm script and this flavour's viewer page.

    The page embeds the flavour's klayout/layers.lyp, or, for flavours without
    one, layer properties built from its layers.yaml.
    """
    gds_dir.mkdir(parents=True, exist_ok=True)
    (kwasm_dir / kwasm_script).write_text(kwasm.embed._read_artifacts())
    if lyp_path.is_file():
        lyp = lyp_path.read_text()
    else:
        with tempfile.TemporaryDirectory() as tmp:
            lyp = LayerViews(PATH.lyp_yaml).to_lyp(Path(tmp) / "layers.lyp").read_text()
    page = VIEWER_HTML.replace("KWASM_SCRIPT", kwasm_script).replace(
        "KWASM_LYP", kwasm.embed._json_script(lyp)
    )
    (kwasm_dir / viewer_name).write_text(page)


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
                f'    <iframe src="kwasm/{viewer_name}?url=gds/{flavour}/{name}.gds"'
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
