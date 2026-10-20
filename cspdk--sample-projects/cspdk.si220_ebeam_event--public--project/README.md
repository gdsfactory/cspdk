# Cornerstone SOI 220 e-beam Sample Project: EVENT-ONLY

> **EVENT-ONLY.** This project uses `cspdk.si220_ebeam_event`, the PDK for the
> CORNERSTONE e-beam workshop event. It is **not** for regular CORNERSTONE MPW
> submissions. The event rules are provisional, and the die's grating I/O is a
> placeholder.

Sample designs for the **CORNERSTONE SOI 220 nm e-beam workshop** (a single
120 nm etch), built with [GDSFactory+](https://gdsfactory.com) and the
open-source Cornerstone PDK (`cspdk.si220_ebeam_event`, cross-section `rib`).

⚠ **Notice:** This project requires an active **GDSFactory+** subscription.
To learn more, visit **[GDSFactory.com](https://GDSFactory.com)**.

## What's inside

A template for one submission: the 6 x 3 mm design area with a ring on the
bottom I/O row and a loopback on the next row, for normalising the grating
response.

- **`mycspdk/ring_die.pic.yml`**: the design as a schematic.
- **`mycspdk/samples/ring_die.py`**: the design in Python, plus
  `ring_transmission()`, which simulates the ring with SAX.

Draw waveguide cores on 3/0 and grating teeth on 6/0. The foundry etches a
5 um trench around every core and merges everything onto 4/0 for exposure.

## Getting started

```bash
uv sync --all-extras
uv run gfp test   # build every cell in the project
uv run python mycspdk/samples/ring_die.py   # check, show the exposure, plot the ring
```

## Before you submit

Run the event design rules and see what the foundry will expose:

```bash
uv run python -m cspdk.si220_ebeam_event.tapeout my_design.gds -o out/
```

It checks:
- 200 nm minimum feature and gap on 3/0, 4/0 and 6/0;
- at most 2,000,000 um^2 of exposure after post-processing;
- that the design fits in 6 x 3 mm.

It also writes `out/my_design_exposure.gds`.
