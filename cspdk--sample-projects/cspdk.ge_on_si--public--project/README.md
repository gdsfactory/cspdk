# Cornerstone Ge-on-Si Sample Project

Sample designs for the **Cornerstone germanium-on-silicon for the mid-infrared (3.8 um)** platform, built with
[GDSFactory+](https://gdsfactory.com) and the open-source Cornerstone PDK
(`cspdk.ge_on_si`, cross-section `xs_rib`).

⚠ **Notice:** This project requires an active **GDSFactory+** subscription.
To learn more, visit **[GDSFactory.com](https://GDSFactory.com)**.

## What's inside

A waveguide loop for cut-back loss measurement: a waveguide of a given length,
folded into a U with bends.

- **`mycspdk/waveguide_loop.pic.yml`**: the loop as a schematic.
- **`mycspdk/samples/waveguide_loop.py`**: the loop in Python, plus
  `loop_transmission()`, which simulates its transmission with SAX. Comparing
  loops of different lengths gives the propagation loss.

## Getting started

```bash
uv sync --all-extras
uv run gfp test   # build every cell in the project
uv run python mycspdk/samples/waveguide_loop.py   # show the layout and transmission
```
