# Cornerstone SiN300 Sample Project

Sample designs for the **Cornerstone 300 nm silicon nitride** platform, built with
[GDSFactory+](https://gdsfactory.com) and the open-source Cornerstone PDK
(`cspdk.sin300`). Every cell comes in a C-band (`_nc`, cross-section `xs_nc`) and
an O-band (`_no`, cross-section `xs_no`) variant.

⚠ **Notice:** This project requires an active **GDSFactory+** subscription.
To learn more, visit **[GDSFactory.com](https://GDSFactory.com)**.

## What's inside

- **`mycspdk/mzi_gratings_nc.pic.yml`**: a C-band MZI between an input grating
  coupler and two output grating couplers.
- **`mycspdk/samples/mzi_with_gratings.py`**: the same circuit in Python for
  either band, plus `mzi_spectrum()`, which simulates both outputs with SAX.

## Getting started

```bash
uv sync --all-extras
uv run gfp test   # build every cell in the project
uv run python mycspdk/samples/mzi_with_gratings.py   # show the layout and spectra
```
