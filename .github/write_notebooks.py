"""Convert the published docs notebooks to Markdown pages.

The notebooks are stored without outputs and are published as code walkthroughs:
the Meep notebooks run simulations on GDSFactory+, which needs an API key the docs
build doesn't have. thermal_mzi_analysis is an internal model investigation that
predates the current heater model, so it is not published.

Usage: python .github/write_notebooks.py
"""

import sys
from pathlib import Path

import nbformat
from nbconvert import MarkdownExporter

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO / "docs" / "notebooks"
PUBLISHED = ("meep_dc", "meep_mmi")

sys.path.insert(0, str(REPO / "docs"))
from hooks import process  # noqa: E402

for name in PUBLISHED:
    notebook = nbformat.read(NOTEBOOKS / f"{name}.ipynb", as_version=4)
    body, _ = MarkdownExporter().from_notebook_node(notebook)
    page = NOTEBOOKS / f"{name}.md"
    page.write_text(process(body))
    print(f"Wrote {page}")
