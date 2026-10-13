# Cornerstone Si SOI 340 nm Sample Project

Sample designs for the **Cornerstone 340 nm silicon-on-insulator** platform, built with
[GDSFactory+](https://gdsfactory.com) and the open-source Cornerstone PDK
(`cspdk.si340`). The sample covers: C-band strip (`_sc`, cross-section `xs_sc340`), O-band strip (`_so`, cross-section `xs_so340`), C-band rib (`_rc`, cross-section `xs_rc340`).

⚠ **Notice:** This project requires an active **GDSFactory+** subscription.
To learn more, visit **[GDSFactory.com](https://GDSFactory.com)**.

## What's inside

- **`mycspdk/mzi_gratings_sc.pic.yml`**: an MZI between an input grating
  coupler and two output grating couplers (C-band strip).
- **`mycspdk/samples/mzi_with_gratings.py`**: the same circuit in Python for every
  band, plus `mzi_spectrum()`, which simulates both outputs with SAX.

## Getting started

```bash
uv sync --all-extras
uv run gfp test   # build every cell in the project
uv run python mycspdk/samples/mzi_with_gratings.py   # show the layout and spectra
```
