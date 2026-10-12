# References

Process documentation for Cornerstone, migrated from [`doplaydo/pdk-reference-files`](https://github.com/doplaydo/pdk-reference-files).

## Contents

- **`CORNERSTONE-GDSII-Template_All-Platforms_Jan_2024-1.gds`** — GDSII template for all Cornerstone platforms.
- **`CORNERSTONE-SOI-Standard-Components-Library-Mar-2023.pdf`** — SOI standard components library.
- **`CORNERSTONE-SiN-Standard-Components-Library-Feb-2022.pdf`** — SiN standard components library.
- **`CORNERSTONE-Suspended-Si-Standard-Components-Library-Feb-2022.pdf`** — Suspended Si standard components library.
- **`components.md`** — Component notes.

## October 2026 (MPW Run 50, SOI 220 nm)

In `october-2026/`:

- **`CORNERSTONE-50th-Call-Design-Guideline.pdf`** and **`CORNERSTONE-50th-Call-Standard-Components.pdf`** — 220 nm SOI only. Versus the 48th call they add hybrid graphene integration (layers 161–169); the silicon, heater, label and die rules are unchanged.
- **`CORNERSTONE-MPW-Run-50-GDSII-Template.gds`** — SOI 220 template with the full standard-component library (`SOI220_cband/oband_*`, laser, photodiode and graphene cells). Die outlines are unchanged; the packaging-template grating couplers moved 25 um down.

Keep `CORNERSTONE-GDSII-Template_All-Platforms_Jan_2024-1.gds`: Run 50 only covers SOI 220, and the other platforms' tests read the older template.

## si220 cells from before Run 50

`si220-pre-run50/` keeps the old si220 GDS files that exist in no other reference (old MZIs, modulators, defect detector, EBL apodised grating, the older SOI 220 template). See its README.
