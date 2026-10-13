"""Write a flavour's SAX model docs as Markdown, with a transmission plot per model.

Usage: python .github/write_models.py <flavour>, e.g. si220 or sin300.
"""

import importlib
import inspect
import sys

import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

TITLES = {
    "si220": "Models Si SOI 220nm C-band and O-band",
    "si340": "Models Si SOI 340nm",
    "si500": "Models Si SOI 500nm",
    "sin200": "Models SiN200 visible",
    "sin300": "Models SiN300",
    "si_sus": "Models suspended Si",
    "ge_on_si": "Models Ge-on-Si",
}
MAX_CURVES = 6  # transmission curves per plot

flavour = sys.argv[1]
if flavour not in TITLES:
    sys.exit(f"unknown flavour {flavour!r}; expected one of {sorted(TITLES)}")

pdk = importlib.import_module(f"cspdk.{flavour}").PDK
PATH = importlib.import_module(f"cspdk.{flavour}.config").PATH
page = PATH.repo / "docs" / f"models_{flavour}.md"
# One folder per flavour: model names repeat across flavours.
plot_dir = PATH.repo / "docs" / "models" / flavour
plot_dir.mkdir(parents=True, exist_ok=True)


def _summary(model) -> str:
    """First paragraph of the docstring, on one line."""
    doc = inspect.getdoc(model) or ""
    return " ".join(doc.split("\n\n")[0].split())


def _parameters(model) -> list[tuple[str, str]]:
    return [
        (name, repr(p.default))
        for name, p in inspect.signature(model).parameters.items()
        if p.default is not inspect.Parameter.empty
    ]


def _plot(name: str, model) -> bool:
    """Plot |S_ij|^2 in dB between optical ports over +-3 % of the centre wavelength."""
    wl0 = inspect.signature(model).parameters["wl"].default
    wl = jnp.linspace(0.97 * wl0, 1.03 * wl0, 201)
    s = model(wl=wl)
    pairs = [
        (i, j)
        for (i, j) in sorted(s)
        if i < j and i.startswith("o") and j.startswith("o")
    ][:MAX_CURVES]
    if not pairs:
        return False
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for i, j in pairs:
        power = np.abs(np.broadcast_to(np.asarray(s[i, j]), wl.shape)) ** 2
        ax.plot(wl * 1e3, 10 * np.log10(np.maximum(power, 1e-12)), label=f"{i} → {j}")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("transmission (dB)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / f"{name}.png", dpi=110)
    plt.close(fig)
    return True


lines = [
    f"# {TITLES[flavour]}\n",
    "SAX models registered in `PDK.models`, evaluated at their default parameters. "
    "Plots show the transmission between optical ports over +-3 % of each model's "
    "centre wavelength.\n",
]
for name, model in sorted(pdk.models.items()):
    if isinstance(model, type):  # circulax-native (active) models
        continue
    lines.append(f"## {name}\n")
    if summary := _summary(model):
        lines.append(f"{summary}\n")
    if parameters := _parameters(model):
        lines.append("| Parameter | Default |\n|---|---|")
        lines.extend(f"| `{p}` | `{d}` |" for p, d in parameters)
        lines.append("")
    if _plot(name, model):
        lines.append(f"![{name}](models/{flavour}/{name}.png)\n")

page.write_text("\n".join(lines))
print(f"Wrote {page}")
