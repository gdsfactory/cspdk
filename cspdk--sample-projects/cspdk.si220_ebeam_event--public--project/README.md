# Cornerstone SOI 220 e-beam Sample Project

> **Event-only:** uses `cspdk.si220_ebeam_event`, the provisional PDK for the
> CORNERSTONE e-beam workshop. Not for regular MPW submissions.

A template for one submission, built with [GDSFactory+](https://gdsfactory.com)
and the open-source Cornerstone PDK.

⚠ **Notice:** This project requires an active **GDSFactory+** subscription.
To learn more, visit **[GDSFactory.com](https://GDSFactory.com)**.

## What's inside

The 6 x 3 mm design area with a ring on the bottom I/O row and a loopback on
the next, for normalising the grating response.

- **`mycspdk/ring_die.pic.yml`**: the design as a schematic.
- **`mycspdk/samples/ring_die.py`**: the design in Python, plus
  `ring_transmission()`, which simulates the ring with SAX.

## Getting started

```bash
uv sync --all-extras
uv run gfp test   # build every cell in the project
uv run python mycspdk/samples/ring_die.py   # check, show the exposure, plot the ring
uv run python -m cspdk.si220_ebeam_event.tapeout my_design.gds -o out/   # before submitting
```
