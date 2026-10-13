# Cornerstone SiN 200 nm (visible) Sample Project

Sample designs for the **Cornerstone 200 nm silicon nitride for visible light** platform, built with
[GDSFactory+](https://gdsfactory.com) and the open-source Cornerstone PDK
(`cspdk.sin200`). The sample covers: 780 nm (`_n780`, cross-section `xs_n780`), 638 nm (`_n638`, cross-section `xs_n638`), 520 nm (`_n520`, cross-section `xs_n520`).

⚠ **Notice:** This project requires an active **GDSFactory+** subscription.
To learn more, visit **[GDSFactory.com](https://GDSFactory.com)**.

## What's inside

- **`mycspdk/mzi_gratings_n780.pic.yml`**: an MZI between an input grating
  coupler and two output grating couplers (780 nm).
- **`mycspdk/samples/mzi_with_gratings.py`**: the same circuit in Python for every
  band, plus `mzi_spectrum()`, which simulates both outputs with SAX.

## Getting started

```bash
uv sync --all-extras
uv run gfp test   # build every cell in the project
uv run python mycspdk/samples/mzi_with_gratings.py   # show the layout and spectra
```
