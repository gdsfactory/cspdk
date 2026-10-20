# cspdk.si220_ebeam_event: EVENT-ONLY

> **EVENT-ONLY.** This flavour is for the CORNERSTONE e-beam workshop event
> only. It is **not** for regular CORNERSTONE MPW submissions: use
> `cspdk.si220` (or another flavour) for those.
>
> **The rules are provisional.** They come from the CORNERSTONE specification
> of 17 Oct 2026 and may change before the event. Pin your design to the
> cspdk release tagged for the event, not to `main`.

```python
from cspdk.si220_ebeam_event import PDK, cells, check, post_process

PDK.activate()
c = cells.die()
assert check(c) == []          # event design rules
exposure = post_process(c)     # what the foundry writes on 4/0
```

## Process

- SOI 220 nm, a single 120 nm (+-10 nm) e-beam etch; 100 nm slab remains.
  Waveguides and gratings get the same etch.
- Rib waveguide: 450 nm core, 25 um bend radius.

## Layers

| Layer | GDS | Use |
|-------|-----|-----|
| `WG` | 3/0 | Waveguide cores (light field). |
| `EXPOSURE` | 4/0 | E-beam exposure (dark field). You may draw extra etch here. |
| `GRA` | 6/0 | Grating teeth. |
| `TRENCH_TMP` | 3/1 | Foundry temporary layer. Do not draw. |
| `FLOORPLAN` | 99/0 | Design area outline. |

## Mask preparation (foundry post-processing)

1. Size 3/0 by +5 um into the temporary layer 3/1.
2. Merge 4/0 + 6/0 + (3/1 - 3/0) into 4/0.

So every core gets a 5 um etched trench on each side.
`post_process(component)` returns a flat copy with the result on 4/0, and
`exposure_region(component)` returns it as a klayout region.

The 200 nm rules also apply to the merged 4/0, the layer that is written.
Plain sizing would break them on the outer edge of the trenches, at least
5 um from any core, so cspdk cleans 3/1 before merging (CORNERSTONE agreed to
this):

- 1-2 nm jogs, left by sizing discretised arcs, are smoothed away;
- gaps between trenches narrower than 210 nm are filled (etched). They form
  where two trenches meet at a shallow angle, such as a ring and its bus.

Gaps formed by drawn 4/0 or 6/0 are not filled: `check` reports them.

## Design rules

`check(component)` returns a list of `Violation` records (empty when the
layout passes). It uses klayout.db only, with no external deck.

| Rule | Check |
|------|-------|
| `WG.W`, `EXPOSURE.W`, `GRA.W` | minimum feature 200 nm on 3/0, 4/0 (as drawn), 6/0 |
| `WG.S`, `EXPOSURE.S`, `GRA.S` | minimum gap 200 nm on the same layers |
| `MERGED.W`, `MERGED.S` | minimum feature and gap 200 nm on the post-processed 4/0 |
| `EXPOSURE.A` | post-processed 4/0 area at most 2,000,000 um^2 |
| `DESIGN.SIZE` | post-processed 4/0 fits in 6 mm (east-west) x 3 mm (north-south) |
| `TRENCH_TMP.R` | nothing drawn on 3/1 (a cspdk check, not a foundry rule) |

Width uses the projection metric (only edges that face each other are
measured); space uses the Euclidean metric, so corner-to-corner gaps count.
Edges meeting at 80 degrees or more are not compared.

Minimum bend radius 25 um: the cells reject smaller radii. `bend_euler` and
`bend_s` also check their tightest curvature, so an Euler bend needs a radius
of about 36 um (the default) at p=0.5.

Design area per submission: 6 mm x 3 mm (`die`).

## Sample project

`cspdk--sample-projects/cspdk.si220_ebeam_event--public--project` is a
template for one submission: a ring and a loopback routed to the die's
placeholder I/O, passing `check`.

## Tapeout

Check submitted GDS files, report each one's exposure area against the
2,000,000 um^2 budget, and write the post-processed layouts:

```bash
python -m cspdk.si220_ebeam_event.tapeout design1.gds design2.gds -o out/
```

It prints `PASS` or `FAIL` per file with the violated rules, writes
`out/<name>_exposure.gds` with the merged exposure on 4/0, and exits with 1 if
any file fails. Use `--cell` to choose the top cell of a file with several.

## Cells

`straight`, `bend_circular` (routing bend), `bend_euler`, `bend_s`,
`coupler`, `coupler_ring`, `ring_single`, `grating_coupler`,
`grating_coupler_sweep`, `die`.

- `grating_coupler`: the CORNERSTONE student-batch rib grating (10 um wide,
  350 um taper, 60 teeth of 335 nm at a 670 nm period, 10 degree fiber),
  with the teeth on 6/0.
- `grating_coupler_sweep`: loopback test rows over period and fill factor.
- `die`: the 6 x 3 mm design area with **provisional placeholder** grating
  I/O. Inputs on the west face the outputs on the east, at a 250 um vertical
  pitch, at least 500 um apart horizontally. An automated I/O tool is planned,
  so this will change.

## Models

| Model | Source |
|-------|--------|
| straight, bends | ng 3.85 and 2.5 dB/cm from the event spec. neff 2.54 is a **placeholder** (cspdk si220 rib mode solve, same geometry). |
| `grating_coupler` | Student-batch rib grating data: 6 dB, 61 nm FWHM, 1.565 um centre. Not measured on the e-beam process. |
| `coupler`, `coupler_ring` | **Placeholders**: the power coupling is a parameter, with no gap dependence. |

## Not known yet (TODO)

- Are labels on 100/0 and 101/0 etched or ignored? The cells draw none.
- Write-field size, and whether devices may cross write-field boundaries.
- Final measurement setup, and the final I/O (grating) placement rules.
- Measured data on the e-beam process (neff, coupling versus gap, grating
  response). CORNERSTONE has no ring or coupler data yet, and its grating
  measurement is of the MPW rib grating (630 nm period, shallow etch), not
  this one. It plans to fabricate test structures in the upcoming run.
- Event date, and the cspdk release to pin to it.
- Cladding and BOX thicknesses.
