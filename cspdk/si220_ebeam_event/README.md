# cspdk.si220_ebeam_event

> **EVENT-ONLY.** For the CORNERSTONE e-beam workshop, not for regular MPW
> submissions (use `cspdk.si220`). The rules are provisional: pin your design to
> the cspdk release tagged for the event.

```python
from cspdk.si220_ebeam_event import PDK, cells, check, post_process

PDK.activate()
c = cells.die()
assert check(c) == []      # event design rules
exposure = post_process(c)  # what is written on 4/0
```

## Process and layers

SOI 220 nm with a single 120 nm (±10 nm) e-beam etch, leaving a 100 nm slab.
Waveguides and gratings get the same etch. Rib waveguide: 450 nm core, 25 µm
minimum bend radius.

| Layer | GDS | Use |
|-------|-----|-----|
| `WG` | 3/0 | Waveguide cores. |
| `EXPOSURE` | 4/0 | E-beam exposure. You may draw extra etch here. |
| `GRA` | 6/0 | Grating teeth. |
| `TRENCH_TMP` | 3/1 | Foundry temporary layer. Do not draw. |
| `FLOORPLAN` | 99/0 | Design area outline. |

## Mask preparation

The foundry sizes 3/0 by +5 µm into 3/1, then merges 4/0 + 6/0 + (3/1 − 3/0)
into 4/0, so every core gets a 5 µm etched trench. `post_process` returns a
flat copy with the result on 4/0; `exposure_region` returns it as a klayout
region.

The 200 nm rules also apply to the merged 4/0. Plain sizing breaks them on the
outer edge of the trenches, at least 5 µm from any core, so cspdk smooths 3/1
and etches any gap in it narrower than 210 nm, such as the point of Si where a
ring's trench meets its bus trench. CORNERSTONE agreed to this. Gaps formed by
drawn 4/0 or 6/0 are left for `check` to report.

## Design rules

| Rule | Check |
|------|-------|
| `WG`, `EXPOSURE`, `GRA` `.W`/`.S` | 200 nm minimum feature/gap on 3/0, 4/0 (as drawn), 6/0 |
| `MERGED.W`/`.S` | the same on the merged 4/0 |
| `EXPOSURE.A` | merged 4/0 at most 2,000,000 µm² |
| `DESIGN.SIZE` | design fits in 6 mm (east-west) × 3 mm |
| `TRENCH_TMP.R` | nothing drawn on 3/1 |

The cells reject radii below 25 µm, including the tightest point of Euler and
S-bends: an Euler bend needs a radius of about 36 µm (the default) at p=0.5.

## Tapeout

```bash
python -m cspdk.si220_ebeam_event.tapeout design1.gds design2.gds -o out/
```

Prints PASS or FAIL per file with its exposure area, writes
`out/<name>_exposure.gds`, and exits 1 if any file fails. The sample project
`cspdk.si220_ebeam_event--public--project` is a template for one submission.

## Cells

`straight`, `bend_circular` (the routing bend), `bend_euler`, `bend_s`,
`coupler`, `coupler_ring`, `ring_single`, `grating_coupler` (the student-batch
rib grating, teeth on 6/0), `grating_coupler_sweep` (period and fill factor),
and `die` (6 × 3 mm with placeholder I/O: inputs west, outputs east, 250 µm
pitch, at least 500 µm apart).

## Models

| Model | Source |
|-------|--------|
| straight, bends | n_g 3.85 and 2.5 dB/cm from CORNERSTONE; neff 2.54 is a placeholder. |
| `grating_coupler` | Student-batch grating datasheet: 6 dB at 1.565 µm, 61 nm FWHM. |
| `coupler`, `coupler_ring` | Placeholders: the coupling is a parameter, independent of gap. |

## Open

- Measured e-beam data (coupling versus gap, grating response): CORNERSTONE
  plans test structures in the next run.
- Final I/O rules and measurement setup (placeholder until their automated
  tool is ready).
- Write-field size; cladding and BOX thicknesses; whether 100/0 labels are
  etched.
