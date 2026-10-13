# CORNERSTONE Standard Components Library

## SOI 220 nm Platform (MPW Run 50, September 2026)

Sources, all in `references/october-2026/`:
`CORNERSTONE-50th-Call-Design-Guideline.pdf` (DG), `CORNERSTONE-50th-Call-Standard-Components.pdf`
(SC, page numbers below) and `CORNERSTONE-MPW-Run-50-GDSII-Template.gds`, which embeds the
whole SOI 220 library. The 50th call is SOI 220 only. Compared with the 48th call it only adds
the graphene components (layers 161-169). The SC measurements are TE transmission
plots. The per-device numbers below were read from them and are the values the `cspdk.si220`
models use. Sign-up deadline 14/10/2026, design submission 11/11/2026 (13:00 UK).

### Process Overview (DG section 4)

- SOI: 220 ± 20 nm Si (100) on a 2 um thermal BOX, 750 Ohm.cm substrate
- Etch 1: 70 ± 10 nm shallow etch for grating couplers (layer 6, deep-UV lithography)
- Etch 2: 120 ± 10 nm intermediate etch for strip and rib waveguides (layers 3 and 4)
- Etch 3: 100 nm continuation etch to the BOX wherever layer 5 does not protect, giving
  220 nm strip waveguides and 120 nm rib waveguides on a 100 nm slab
- 1 um ± 100 nm SiO2 top cladding
- Heaters: TiN filaments (layer 39) and Ti/Al contact pads (layer 41), both lift-off
- Optional services, each adding to the access cost with heaters and to the delivery time:
  - bonded photodiodes (University of Glasgow pick-and-place, +4 weeks);
  - bonded C-band laser dies (Glasgow, +14 weeks);
  - hybrid graphene integration (Cambridge Graphene Centre, +14 weeks).
- Graphene and laser/PD integration are mutually exclusive. Graphene users receive
  8 dies with heaters and no graphene, plus 2 dies with graphene and no heaters.
- Quality targets (DG Table 7):
  - straight single-mode rib waveguide loss < 4 dB/cm (TE, C-band);
  - strip MZI with the PDK heater < 30 mW per pi;
  - integrated PD responsivity > 0.5 A/W at 1550 nm;
  - integrated laser > 1 dBm into the waveguide at 200 mA, 1550 ± 3 nm.

### Layer Definitions (DG section 5.2 and Table 6)

| Layer | GDS | Field | cspdk name | Description |
|-------|-----|-------|------------|-------------|
| Silicon Etch 1 | 6/0 | Dark | `GRA` | 70 nm grating etch |
| Silicon Etch 2 | 3/0 | Light | `WG` | Waveguides; Cornerstone etches 5 um trenches around them (120 nm) |
| Silicon Etch 2 | 4/0 | Dark | `WG_DF` | Extra 120 nm etch inside layer 3 (e.g. photonic crystals) |
| Silicon Etch 3 (rib protect) | 5/0 | Light | `SLAB` | Protected from the etch to BOX; ≥ 5 um beyond layer 3 for rib |
| Heater filaments | 39/0 | Light | `HEATER` | TiN; 900 nm filament width recommended |
| Heater contact pads | 41/0 | Light | `PAD` | Ti/Al; also PD and laser routing |
| Laser bottom contacts | 43/2 | Light | `LASER_CONTACT` | 50 nm Ti / 50 nm Pt / 500 nm Au |
| Laser trench etch | 47/2 | Dark | `LASER_TRENCH` | Cladding, SOI, BOX and 5 um into the substrate |
| Physical PD area | 140/1 | - | `PD_AREA` | Information |
| Physical laser area | 143/1 | - | `LASER_AREA` | Information; Table 6 lists it as "Physical PD Area", the template uses it for the laser |
| Laser waveguide | 144/1 | - | `LASER_WG` | Information |
| PD active area | 147/1 | - | `PD_ACTIVE` | Information |
| PD exclusion zone | 150/1 | - | `PD_EXCLUSION` | Information, not DRC-checked |
| Laser exclusion zone | 151/1 | - | `LASER_EXCLUSION` | No layer 39 or 41 inside |
| 1st graphene channel | 161/2 | - | `GRAPHENE1` | Graphene process layer |
| 2nd graphene channel | 162/2 | - | `GRAPHENE2` | Graphene process layer |
| 1st Au/graphene contacts | 163/2 | - | `GRAPHENE_CONTACT1` | Names of 163-166 follow Table 6; section 5.2 uses other names |
| 2nd Au/graphene contacts | 164/2 | - | `GRAPHENE_CONTACT2` | |
| 1st Au/Ti metallization | 165/2 | - | `GRAPHENE_PAD1` | Graphene contact pads |
| 2nd Au/Ti metallization | 166/2 | - | `GRAPHENE_PAD2` | Graphene contact pads |
| Graphene exclusion zone | 169/1 | - | `GRAPHENE_EXCLUSION` | Grown by 50 um and merged into a cladding opening (layer 22) |
| Cell outline | 99/0 | - | `FLOORPLAN` | 11.47 x 4.9 mm2 or 5.5 x 4.9 mm2 |
| Labels | 100/0 | Dark | `LBL` | Merged into Silicon Etch 2; not DRC-checked |

Layer 60 (EBL grating, `GRA_EBL`) is no longer in the guideline (dropped in the 48th call).
The v3.1 DRC deck still checks it, so cspdk keeps it until Cornerstone confirms.

### Design Rules (DG Table 6 and section 5.3)

| Layer | Min feature | Min gap | Max feature length |
|-------|-------------|---------|--------------------|
| 6 | 200 nm | 250 nm | 20 um (gaps 250-350 nm only) |
| 6 | 200 nm | 350 nm | - |
| 3 | 350 nm | 200 nm | - |
| 4 | 200 nm | 350 nm | - |
| 5 | 250 nm | 250 nm | - |
| 39 | 600 nm | 10 um | - |
| 41 | 2 um | 10 um | - |
| 161/2, 162/2 | 3 um, 400 nm | 3 um, 400 nm | - |
| 163/2, 164/2 | 1.5 um, 425 nm | 1.5 um, 425 nm | - |
| 165/2, 166/2 | 10 um | 10 um | 200 um |
| 169/1 | 20 um | 20 um | - |
| 100 | 250 nm | 250 nm | - |

- Manufacturing grid 1 nm; designs in GDSII or OASIS; top cell named `Cell0_[Name of Institution]`
- Layer 6 must overlap layer 3 by at least 200 nm, and must not overlap layers 39 or 41
- At least 10 um overlap between layers 39 and 41 is recommended
- Waveguides at least 5 um apart are recommended to avoid coupling
- Layer 151/1 must not overlap layers 39 or 41
- Layers 5, 6, 39 and 41 must stay more than 50 um from layer 169/1
- PD layers 140/1, 147/1 and 150/1 are not DRC-checked
- Design area 11.47 x 4.9 mm2 (full) or 5.5 x 4.9 mm2 (half); the physical die is about
  12.5 x 5.3-5.6 mm2 including Cornerstone's border

### Waveguides

| Type | Etch | C-band width | O-band width | cspdk cross-section | Default / min radius |
|------|------|--------------|--------------|---------------------|----------------------|
| Strip | 220 nm (to BOX) | 450 nm | 400 nm | `strip_cband`, `strip_oband` | 10 / 5 um |
| Rib | 120 nm, 100 nm slab | 450 nm | 400 nm | `rib_cband`, `rib_oband` | 25 / 25 um |

The SC library gives no straight-waveguide loss. The only figure is the DG quality target,
< 4 dB/cm for a C-band rib waveguide. The cspdk straight and MZI models use 3 dB/cm.

### Grating Couplers (SC p. 6, p. 37)

| Parameter | C-band | O-band |
|-----------|--------|--------|
| Period (tooth / gap) | 630 nm (315 / 315 nm) | 500 nm (250 / 250 nm) |
| Periods | 60 | 80 |
| Grating width / taper length | 10 um / 350 um | 10 um / 350 um |
| Etch | 70 nm | 70 nm |
| Fibre angle | 10 degrees | 10 degrees |
| Coupling loss | 5.5-6.5 dB | 5-6 dB |
| 1 dB bandwidth | > 35 nm | > 30 nm |
| Centre wavelength | 1550-1580 nm | 1300-1330 nm |
| cspdk model | 6.0 dB, 1565 nm, FWHM 60.7 nm | 5.5 dB, 1315 nm, FWHM 52.0 nm |

Only the rib gratings are measured. The strip gratings share the grating section and
use the same data. cspdk cells: `grating_coupler_rectangular` (strip) and
`grating_coupler_rectangular_rib`, drawn in code to match the template.

The template also has `SOI220_cband_TE_strip_apodized_GC_focusing_v1p0`, a C-band focusing
grating used by the laser `_with_GC` / `_or_GC` variants. The SC has no page or data for it.
cspdk imports it as `grating_coupler_focusing_apodized`, with no model.

### MMIs (SC pp. 8, 9, 15, 16, 39, 40, 43, 44)

All MMIs are 6 um wide with 20 um access tapers to 1.5 um. The measured loss is the
insertion loss per MMI along one output. cspdk subtracts the 3.01 dB ideal split to get the
excess loss.

| Cell | Length | Taper gap | Measured (dB) | cspdk excess (dB) |
|------|--------|-----------|---------------|-------------------|
| C strip 1x2 | 31.8 um | 1.64 um | 3.3056 | 0.295 |
| C rib 1x2 | 32.7 um | 1.64 um | 3.0621 | 0.052 |
| O strip 1x2 | 40.1 um | 1.55 um | 3.2885 | 0.278 |
| O rib 1x2 | 40.8 um | 1.55 um | "old data", same as O strip | 0.278 |
| C strip 2x2 | 42.5 um | 0.53 um | bar 3.3673, cross 3.3073 | 0.357 / 0.297 |
| C rib 2x2 | 44.8 um | 0.53 um | bar 1.7851, cross 2.2493 (below the 3 dB split; not used) | 0.3 (generic) |
| O strip 2x2 | 53.5 um | 0.53 um | no data | 0.3 (generic) |
| O rib 2x2 | 55.0 um | 0.53 um | no data | 0.3 (generic) |

cspdk cells: `mmi1x2`, `mmi2x2` (strip) and `mmi1x2_rib`, `mmi2x2_rib`. The band comes
from the cross-section. Rib MMIs end in a 5 um straight so their ports sit in a full rib
cross-section.

### Y-Splitter (SC p. 17, p. 45)

- 40 um long, 20 um output separation, strip only
- Cut-back loss per splitter through one output:
  - C band: 4.0123 dB (450 nm waveguides) and 3.7101 dB (400 nm);
  - O band: 4.5006 dB.
- cspdk `ysplitter` (imported from the template, band from the cross-section). Its model
  uses an excess loss of 1.002 dB in the C band and 1.490 dB in the O band, flat in
  wavelength.

### Waveguide Crossings (SC pp. 11, 19, 41, 47)

| Cell | Insertion loss (dB) | Width |
|------|---------------------|-------|
| C strip | 0.1195 | 450 nm |
| C rib | 0.12467 | 450 nm |
| O strip | 0.24359 | 400 nm |
| O rib | 0.2123 | 400 nm |

cspdk cells: `crossing` (strip or rib, from the cross-section) and `crossing_rib`.

### Bends (SC pp. 10, 18, 46)

Loss per 90 degree circular bend. The cspdk bend models interpolate linearly in radius
and hold the end values beyond the table.

| Cell | Template radius | Loss (dB / 90 degrees) |
|------|-----------------|------------------------|
| C strip, W 450 nm | 5 um | 0.0136 at R 5 um, 0.0116 at R 10 um |
| C rib, W 450 nm | 25 um | 0.1006 at R 10 um, 0.0310 at R 25 um |
| O strip, W 400 nm | 5 um (SC text: 30 um) | 0.0106 at R 5 um, 0.0166 at R 10 um |
| O rib | - | no data; uses the C rib values |

cspdk cells: `bend_circular` (like the foundry bends) and `bend_euler`.

### Tapers (SC pp. 12, 20, 21)

| Taper | Length (um) | Loss (dB) |
|-------|-------------|-----------|
| Strip MM-to-SM, 10 um to 450 nm | 300 / 450 / 600 | 0.1365 / 0.1102 / 0.1424 |
| Rib MM-to-SM, 10 um to 450 nm | 300 / 450 / 600 | 0.1914 / 0.1350 / 0.1572 |
| Rib-to-strip, 5 um slab to 450 nm | 10 / 20 / 50 | 0.1192 / 0.0902 / 0.0681 |

cspdk cells:
- `taper_mm2sm` (default 450 um, strip or rib from the cross-section);
- `taper_rib_to_strip` (default 50 um);
- `taper_strip_to_ridge`.

No O-band taper data; the O-band models use the C-band values.

### MZI and Heater (SC pp. 13, 22)

- C-band strip and rib MZIs (the SC pages have no data). cspdk `mzi_heater_loop` draws them,
  with 1x2 and 2x2 MMIs, 30 um radius bends and a heater on the upper arm.
  - Strip: `length_x` = 616.3 um.
  - Rib: `length_x` = 649.5 um.
- cspdk `heater` reproduces the template `Heater` cell: a 200 um x 2 um filament with
  50 um leads and two 100 x 100 um pads at 120 um pitch. The DG recommends a 900 nm
  filament for new designs.

### Integrated Laser (SC pp. 23-25, DG section 4.2)

- Shijia Photons DC1550HP01, 1550 ± 3 nm, > 1 mW into the edge coupler
- Laser footprint 1 x 0.25 mm, trench etch 2 x 1 mm, exclusion zone 2.5 x 2 mm
- Only usable as part of a full or half design template; up to 1 laser per die,
  2 across the delivered dies
- Three variants:
  - `Integrated`: the laser feeds the circuit through an edge coupler.
  - `Integrated_with_GC`: a 2x1 MMI combines the laser with a focusing grating.
  - `Integrated_or_GC`: a focusing grating on the laser site feeds the circuit.
- cspdk cells (imported, no model):
  - `laser_full`, `laser_full_with_gc`, `laser_full_or_gc`;
  - `laser_half`, `laser_half_with_gc`, `laser_half_or_gc`.

### Integrated Photodiode (SC pp. 26-28, DG section 4.1)

- OSI Optoelectronics FCI-InGaAs-300B1, 900-1700 nm, > 0.5 A/W at 1550 nm (based on GC
  performance); active area 300 um diameter centred at (0, 0)
- Flip-chip bonded onto layer 41 routing; up to 10 PDs per die, 20 across the delivered dies
- Two variants:
  - `Plain`: pads only. cspdk `photodiode`, 2 pad ports.
  - `Integrated`: the PD sits over a C-band strip grating coupler. cspdk
    `photodiode_integrated`, 1 waveguide port and 2 pad ports.
- No models

### Graphene Components (SC pp. 29-35, DG section 4.3)

All six are C-band strip devices. Each also has a `_with_GC` template variant with grating
couplers on both waveguide ends.

| Template cell (`SOI220_cband_TE_strip_SLG_*`) | cspdk cell | Footprint | Responsivity | Bandwidth | Pads |
|------------------------------|------------|-----------|--------------|-----------|------|
| `Bolometer_metallic` | `graphene_bolometer_metallic` | 310 x 310 um | 0.1 A/W at 0 dBm | > 67 GHz | 3 |
| `Bolometer` | `graphene_bolometer` | 310 x 320 um | 0.1 A/W at 0 dBm | > 67 GHz | 3 |
| `Bolometer_no_gate` | `graphene_bolometer_no_gate` | 310 x 320 um | 0.1 A/W at 0 dBm | > 67 GHz | 2 |
| `EAM` | `graphene_eam` | 210 x 210 um | - | > 50 GHz | 2 |
| `PTE` | `graphene_pte` | 210 x 230 um | 3.5 V/W | > 50 GHz | 4 |
| `PTE_metallic` | `graphene_pte_metallic` | 210 x 230 um | 3.5 V/W | > 50 GHz | 4 |

The `_with_GC` variants are `graphene_*_with_gc` in cspdk. The pad ports are on the
graphene pad layers (165/2, 166/2), so a route on the same layer connects to them. There
are no models.

### Die Templates

- `die` (11470 x 4900 um) and `die_half` (5500 x 4900 um): the floorplan on layer 99,
  matching `Cell0_SOI_Full_Institution` and `Cell0_SOI_Half_Institution`
- `die_with_pads` reproduces `Cell0_SOI220_cband_TE_Full_Packaging_Template`:
  - 16 C-band strip gratings per side at 250 um pitch, outermost pair joined by a loopback;
  - 31 pads of 150 x 200 um per side at 300 um pitch.

---

## SiN Platform (February 2022)

### Process Overview

- Platform: Silicon Nitride (SiN) on SiO2
- SiN thickness: 300 nm (LPCVD Si3N4)
- Undercladding: 3 um thermal SiO2
- Overcladding: SiO2
- Lithography: e-beam
- Operating wavelength: O-band through C-band

### Layer Definitions

| Layer Name | GDS Layer | Description |
|------------|-----------|-------------|
| SiN_Full Etch | 1 | 300 nm full etch |
| SiN_Partial Etch | 2 | 150 nm partial etch (slab = 150 nm) |
| Heater | 11 | TiN heater |
| Via | 12 | Contact via |
| Metal | 13 | Aluminum routing |
| FloorPlan | 100 | Die boundary |

### Waveguides

#### SiN Strip Waveguide

| Parameter | Value |
|-----------|-------|
| Width (single-mode, C-band) | 1000 nm |
| Width (single-mode, O-band) | 800 nm |
| Height | 300 nm |
| Propagation loss | < 1 dB/cm |
| Effective index (TE0) | ~1.70 at 1550 nm |

### Grating Couplers

#### SiN TE Grating Coupler

| Parameter | Value |
|-----------|-------|
| Center wavelength | 1550 nm |
| Coupling loss | ~6 dB per coupler |
| 1 dB bandwidth | ~40 nm |
| Fiber angle | 8-10 degrees |

### Edge Couplers

| Parameter | Value |
|-----------|-------|
| Taper tip width | 200 nm |
| Coupling loss | < 2 dB |

### Splitters

#### SiN 1x2 MMI

| Parameter | Value |
|-----------|-------|
| Insertion loss | < 0.3 dB |
| Imbalance | < 0.2 dB |

#### SiN 2x2 MMI

| Parameter | Value |
|-----------|-------|
| Insertion loss | < 0.5 dB |

#### SiN Y-Branch

| Parameter | Value |
|-----------|-------|
| Splitting ratio | 50:50 |
| Insertion loss | < 0.3 dB |

### Ring Resonators

| Parameter | Value |
|-----------|-------|
| Radius | 50-200 um |
| Q factor | 50,000-500,000 |
| FSR (R=100 um) | ~1.5 nm |

### Directional Couplers

| Parameter | Value |
|-----------|-------|
| Gap | 300-500 nm |
| Excess loss | < 0.1 dB |

### Bends

| Minimum Radius | Loss (90-degree) |
|---------------|------------------|
| 20 um | < 0.05 dB |
| 50 um | negligible |

### Crossings

| Parameter | Value |
|-----------|-------|
| Insertion loss | < 0.15 dB |
| Crosstalk | < -30 dB |

### Thermo-Optic Phase Shifter

| Parameter | Value |
|-----------|-------|
| Heater material | TiN |
| Pi phase shift power | ~50 mW |
| Response time | ~20 us |

---

## Suspended Silicon Platform (library February 2022, MPW #7 guidelines February 2025)

Sources: `CORNERSTONE-Suspended-Si-Standard-Components-Library-Feb-2022.pdf`,
`CORNERSTONE_Suspended-Si_MPW_7-_Design_Guidelines1.pdf` and the library GDS in
`cspdk/si_sus/gds/`. The library reports no measured data for any component
("No data at the moment").

### Process Overview

- SOI: 500 nm ± 15 nm Si (100) on 3 um thermal BOX, 750 Ohm.cm substrate
- Etch 1 (layer 404, dark field): 300 nm ± 15 nm partial etch through a 200 nm SiO2 hard mask
- Etch 2: 200 nm continuation etch to the BOX wherever layer 405 does not protect, then HF release
- The HF undercuts the BOX by ~8 um in each direction; every strip waveguide is suspended in air
- After HF the Si is 450 nm ± 20 nm thick and lateral features shrink by ~70 nm
- Operating wavelength of the library: 3800 nm, TE

### Layer Definitions

| Layer | GDS | Field | Description |
|-------|-----|-------|-------------|
| Silicon Etch 1 | 404 | Dark | Drawn shapes are etched; grating couplers, suspended and rib waveguides |
| Silicon Etch 2 (rib protect) | 405 | Light | Drawn shapes are protected from the etch to BOX (rib slab) |
| Cell outline | 99 | - | 11.47 x 4.9 mm2 or 5.5 x 4.9 mm2 design area |
| Labels | 100 | Dark | Merged into layer 404 by Cornerstone |

### Bias Options

The library cells are drawn un-biased; Cornerstone recommends combining them
with un-biased user designs and selecting "CORNERSTONE to bias", which shrinks
layer 404 by 35 nm in every direction (etched features 70 nm narrower).

### Design Rules (MPW #7 Table 2 and section 5.3)

| Layer | Option | Min feature | Min gap | Max suspended width | Max support width |
|-------|--------|-------------|---------|---------------------|-------------------|
| 404 | NOT to bias | 200 nm | 250 nm | 16 um | 6 um |
| 404 | CORNERSTONE to bias | 270 nm | 180 nm | 16 um | 6 um |
| 405 | - | 200 nm | 250 nm | - | - |
| 100 | - | 250 nm | 250 nm | - | - |

- No islands < 20 um on layers 404 and 100 (lifted off during the HF release)
- Gaps < 350 nm on layers 404 and 100 must be at most 20 um long
- At least 75 um between waveguides is recommended to avoid suspended waveguides collapsing
- Quality target: straight single-mode suspended waveguide loss < 5 dB/cm (TE, 3.8 um)

### Components (3800 nm TE, etch depth 500 nm)

#### Suspendedsilicon500nm_3800nm_TE_Waveguide

- 1.5 um core between two 3.5 um etch windows (8.5 um total)
- Windows drawn as sub-wavelength slots: period 550 nm, Si fill factor 0.4545 (0.3 um slots, 0.25 um tethers)
- GDS: 500 um long, 909 slot pairs

#### Suspendedsilicon500nm_3800nm_TE_90_DegreeBend

- Suggested bend radius 40 um (GDS: 40 um at the inner core edge, 40.75 um center line)
- GDS: polar-wedge slots of 0.0075 rad at 0.01375 rad pitch (0.3 um / 0.55 um at r = 40 um), 115 slot pairs

#### Suspendedsilicon500nm_3800nm_TE_SBend

- 40 um long, 8 um offset
- GDS: cosine-like center line, 72 vertical slot pairs (0.3 um wide, 0.55 um x-pitch)

#### Suspendedsilicon500nm_3800nm_TE_Grating_Coupler

- Fiber coupling angle 19 degrees
- 300 um slotted taper from the 1.5 um waveguide to a 15 um wide grating
- Hole array: period 1.1 um x 2.3 um, fill factor 0.51 x 0.5
- Holes: 13 across the width; the PDF says 30 along the grating, the GDS has 20

### cspdk.si_sus Implementation Notes

- `xs_sus` draws the core on the abstract marker (404, 10) and the slots on (404, 0)
- `straight(500)` reproduces the foundry waveguide GDS; `grating_coupler_rectangular` imports the foundry GDS
- `bend_circular` (default radius 40.75 um) uses the foundry wedges but 114 slot pairs per 90 degrees,
  so no wedge overhangs the port plane; `bend_s` (default 40 x 8 um) uses the foundry vertical slots
- Every cell keeps its slots >= 0.125 um inside its ports, so abutting cells keep >= 0.25 um tethers
- SAX models: femwell neff/ng with an effective-medium slotted cladding, 5 dB/cm loss; the
  grating-coupler model is a placeholder

---

## Platform Comparison

| Property | SOI (220 nm) | SiN (300 nm) | Suspended Si |
|----------|-------------|-------------|-------------|
| Waveguide material | c-Si | Si3N4 | c-Si (suspended in air) |
| Core height | 220 nm | 300 nm | 450 nm (500 nm SOI after HF) |
| Cladding | 1 um SiO2 | SiO2 | Air + slotted Si side cladding |
| Loss (dB/cm) | < 4 (rib target; models use 3) | < 1 | < 5 (target, 3.8 um) |
| Min bend radius | 5 um | 20 um | 40 um (suggested) |
| Active devices | Hybrid: bonded PD, laser or graphene | No (heaters only) | No |
| Operating range | C/O-band | Vis to C-band | Mid-IR (3.8 um library) |
| Lithography | deep-UV (gratings); others not stated | e-beam | not stated |
| Key advantage | Hybrid integration | Low loss | Mid-IR transparency |
