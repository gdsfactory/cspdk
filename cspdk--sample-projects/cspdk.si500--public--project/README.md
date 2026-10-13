# Cornerstone Si SOI 500 nm Sample Project

Sample designs for the **Cornerstone 500 nm silicon-on-insulator** platform, built with
[GDSFactory+](https://gdsfactory.com) and the open-source Cornerstone PDK
(`cspdk.si500`). The sample covers: C-band rib (`_rc`, cross-section `xs_rc500`).

The 500 nm platform is a 1550 nm process, so the sample uses the C-band rib cells only.

⚠ **Notice:** This project requires an active **GDSFactory+** subscription.
To learn more, visit **[GDSFactory.com](https://GDSFactory.com)**.

## What's inside

- **`mycspdk/mzi_gratings_rc.pic.yml`**: an MZI between an input grating
  coupler and two output grating couplers (C-band rib).
- **`mycspdk/samples/mzi_with_gratings.py`**: the same circuit in Python for every
  band, plus `mzi_spectrum()`, which simulates both outputs with SAX.

## Getting started

```bash
uv sync --all-extras
uv run gfp test   # build every cell in the project
uv run python mycspdk/samples/mzi_with_gratings.py   # show the layout and spectra
```
