# CORNERSTONE Standard Components Library

## SOI Platform (March 2023)

### Process Overview

- Foundry: CORNERSTONE, University of Southampton
- Platform: Silicon-on-Insulator (SOI)
- Wafer diameter: 100 mm (4-inch)
- Top silicon: 220 nm
- Buried oxide (BOX): 2 um (or 3 um option)
- Lithography: e-beam
- Cladding: 1 um SiO2 upper cladding (optional)

### Layer Definitions

| Layer Name | GDS Layer | Description |
|------------|-----------|-------------|
| Si_Full Etch | 1 | 220 nm full etch (strip waveguides) |
| Si_Partial Etch (70 nm) | 2 | 70 nm shallow etch (slab = 150 nm) |
| Si_Partial Etch (130 nm) | 3 | 130 nm etch (slab = 90 nm, rib waveguides) |
| N implant | 5 | N-type doping |
| P implant | 6 | P-type doping |
| N+ implant | 7 | Heavy N-type doping |
| P+ implant | 8 | Heavy P-type doping |
| Ge | 9 | Germanium epitaxy |
| Heater | 11 | TiN heater |
| Via | 12 | Contact via opening |
| Metal | 13 | Aluminum routing |
| FloorPlan | 100 | Die boundary |
| Text | 101 | Label |

### Design Rules

| Parameter | Value |
|-----------|-------|
| Minimum feature size (full etch) | 60 nm |
| Minimum feature size (partial etch) | 100 nm |
| Minimum waveguide width | 300 nm |
| Minimum spacing (waveguide-to-waveguide) | 200 nm |
| Grid | 1 nm |
| Minimum bend radius (strip, 220 nm) | 5 um |
| Minimum bend radius (rib) | 10 um |
| Heater minimum width | 1 um |
| Metal minimum width | 2 um |
| Metal minimum spacing | 2 um |
| Bond pad minimum size | 80 um x 80 um |

### Waveguides

#### Strip Waveguide (220 nm full etch)

| Parameter | Value |
|-----------|-------|
| Width | 450 nm (single-mode TE) / 500 nm |
| Height | 220 nm |
| Propagation loss (TE) | ~3 dB/cm |
| Effective index (450 nm width, TE0) | ~2.35 at 1550 nm |

#### Rib Waveguide (130 nm partial etch)

| Parameter | Value |
|-----------|-------|
| Width | 400-500 nm |
| Total height | 220 nm |
| Slab height | 90 nm |
| Propagation loss | ~1.5 dB/cm |

### Grating Couplers

#### SOI TE Grating Coupler

| Parameter | Value |
|-----------|-------|
| Etch type | 70 nm partial etch |
| Center wavelength | 1550 nm |
| Fiber angle | 10 degrees |
| Peak coupling loss | ~5 dB per coupler |
| 1 dB bandwidth | ~30 nm |
| Footprint | ~12 um x 15 um |

#### SOI TM Grating Coupler

| Parameter | Value |
|-----------|-------|
| Etch type | 70 nm partial etch |
| Center wavelength | 1550 nm |
| Peak coupling loss | ~6 dB per coupler |

### Edge Couplers

| Parameter | Value |
|-----------|-------|
| Taper tip width | 180 nm |
| Taper length | 150-300 um |
| Mode field diameter match | ~2.5 um (lensed fiber) |
| Coupling loss | < 3 dB |

### Splitters and Combiners

#### 1x2 MMI

| Parameter | Value |
|-----------|-------|
| Width | 6 um |
| Length | ~28 um |
| Insertion loss | < 0.3 dB |
| Imbalance | < 0.2 dB |
| Bandwidth | > 60 nm |

#### 2x2 MMI

| Parameter | Value |
|-----------|-------|
| Width | 6 um |
| Length | ~55 um |
| Insertion loss | < 0.5 dB |
| Imbalance | < 0.3 dB |

#### Y-Branch

| Parameter | Value |
|-----------|-------|
| Splitting ratio | 50:50 |
| Insertion loss | < 0.3 dB |
| Taper length | 10-20 um |

### Directional Couplers

| Parameter | Value |
|-----------|-------|
| Gap | 200 nm |
| Coupling length | Design-dependent |
| Excess loss | < 0.1 dB |
| Cross-coupling sensitivity | Wavelength-dependent |

### Ring Resonators

| Parameter | Value |
|-----------|-------|
| Type | All-pass or Add-drop |
| Radius | 5-20 um |
| Coupling gap | 100-300 nm |
| Q factor | 10,000-50,000 (typical) |
| FSR (R=10 um) | ~12 nm |
| Extinction ratio | > 15 dB |

### Mach-Zehnder Interferometers

| Parameter | Value |
|-----------|-------|
| Splitter type | MMI or Y-branch |
| Phase shifter | Thermo-optic (TiN heater) |
| Heater power for pi shift | ~25 mW |
| Switching time | ~10 us |

### Thermo-Optic Phase Shifters

| Parameter | Value |
|-----------|-------|
| Heater material | TiN |
| Heater width | 2 um |
| Heater-to-waveguide offset | 1 um laterally |
| Pi phase shift | ~25 mW |
| Switching speed | ~10 us |

### PN Junction Modulators

| Parameter | Value |
|-----------|-------|
| Type | Lateral PN junction, carrier depletion |
| Waveguide | Rib (90 nm slab) |
| VpiLpi | ~1.5 V-cm |
| Insertion loss | ~5 dB/cm |
| Bandwidth | > 20 GHz |

### Germanium Photodetectors

| Parameter | Value |
|-----------|-------|
| Responsivity | > 0.8 A/W at 1550 nm |
| Dark current | < 100 nA at -1 V |
| 3 dB bandwidth | > 20 GHz |
| Wavelength range | 1260-1620 nm |

### Waveguide Crossings

| Parameter | Value |
|-----------|-------|
| Insertion loss | < 0.2 dB |
| Crosstalk | < -30 dB |
| Type | Shaped (expanded waveguide) |

### Bends

| Type | Minimum Radius | Loss (90-degree) |
|------|---------------|------------------|
| Strip circular | 5 um | < 0.05 dB |
| Strip Euler | 3 um (effective) | < 0.05 dB |
| Rib circular | 10 um | < 0.02 dB |

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
| Cladding | SiO2 | SiO2 | Air + slotted Si side cladding |
| Loss (dB/cm) | ~3 | < 1 | < 5 (target, 3.8 um) |
| Min bend radius | 5 um | 20 um | 40 um (suggested) |
| Active devices | Yes (PN, Ge PD) | No (heaters only) | No |
| Operating range | C/O-band | Vis to C-band | Mid-IR (3.8 um library) |
| Lithography | e-beam | e-beam | not stated |
| Key advantage | Active integration | Low loss | Mid-IR transparency |
