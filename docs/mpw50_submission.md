# Preparing an MPW Run 50 submission (SOI 220 nm)

CORNERSTONE's 50th call is a 220 nm SOI passive run, and `cspdk.si220` matches its standard-component library. This page covers what a design needs before you upload it. Everything here comes from the [50th-call design guideline](https://cornerstone.sotonfab.co.uk/mpw/live-calls/); read the guideline itself for the full rules and the services' costs.

**Deadlines:** sign-up by 14/10/2026, design submission by 11/11/2026, both at 13:00 UK time.

## 1. Lay out the design on a die

The design area is either a **full cell** (11470 x 4900 µm) or a **half cell** (5500 x 4900 µm), drawn as the cell outline on layer 99/0. Use one of the die cells:

- `die()`: the full cell outline;
- `die_half()`: the half cell outline;
- `die_with_pads()`: the full-cell packaging template, with 16 grating couplers per side (the outermost pair joined by a loopback) and 31 pads per side.

Name the top cell `Cell0_<Name of Institution>`, and keep every shape inside the outline:

```python
import gdsfactory as gf
from cspdk.si220 import PDK, cells

PDK.activate()

c = gf.Component("Cell0_MyUniversity")
c << cells.die()
mzi = c << cells.mzi_heater_loop()
gc = c << cells.grating_coupler_rectangular()
gc.dmove((1000, 0))
c.write_gds("Cell0_MyUniversity.gds")
```

## 2. Check the file

`cspdk.si220.submission.check_submission` checks the submission rules in the guideline that aren't geometric design rules; DRC covers those in step 3. It checks that:

- the file is GDSII or OASIS (section 5.5);
- there is exactly one top cell, named `Cell0_<Name of Institution>` (section 8);
- only designated GDS layers are used (section 5.2 and Table 6);
- the layout is on the 1 nm manufacturing grid (section 5.5);
- the layer 99/0 cell outline is one full or half design area, and nothing extends outside it (section 5.1).

```python
from cspdk.si220.submission import check_submission

problems = check_submission("Cell0_MyUniversity.gds")
print(problems or "OK")
```

Or from the command line, which exits with status 1 if any file has a problem:

```bash
python -m cspdk.si220.submission Cell0_MyUniversity.gds
```

Every `cspdk.si220` cell draws only on designated layers, so problems usually come from your own shapes or from the die setup.

## 3. Run DRC with GDSFactory+

The geometric design rules are minimum feature sizes, gaps, overlaps and exclusion zones (section 5.3 and Table 6). Check them with GDSFactory+'s DRC. For `cspdk.si220` it runs CORNERSTONE's SOI 220 nm passive rules, including the 50th call's graphene rules, with nothing to configure.

Run DRC from the GDSFactory+ extension in VS Code, or from the command line:

```bash
gfp check drc Cell0_MyUniversity.gds --pdk cspdk.si220
```

Inside a GDSFactory+ project for `cspdk.si220` you can leave out `--pdk`.

The results are saved as a KLayout report database (`.lyrdb`). DRC runs on the GDSFactory+ service, so it needs a GDSFactory+ API key (`gfp configure`).

You can also cross-check with CORNERSTONE's own KLayout pre-DRC script, from the [live calls page](https://cornerstone.sotonfab.co.uk/mpw/live-calls/). CORNERSTONE's team runs a full DRC after submission, and the guideline offers 20-minute one-to-one drop-in sessions to review a layout before the deadline; the booking link is in the guideline.

## 4. Optional services

The guideline describes three add-on services, each with its own cost and longer delivery time:

- **Bonded photodiodes:** use the `photodiode` and `photodiode_integrated` cells.
- **An integrated C-band laser:** use the `laser_full*` and `laser_half*` design templates.
- **Hybrid graphene integration:** use the `graphene_*` cells.

Photodiode and laser integration both require the access cost with heaters. Graphene integration can't be combined with laser or photodiode integration.

## 5. Submit

Upload the GDSII or OASIS file and a purchase order through CORNERSTONE's mask submission form, linked from the guideline, before the deadline. You must have signed up for the call first.
