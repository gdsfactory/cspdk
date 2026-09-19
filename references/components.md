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

## Suspended Silicon Platform (February 2022)

### Process Overview

- Platform: Suspended Silicon Waveguides
- Base: SOI wafer (220 nm Si / 2 um BOX)
- Etch: Full etch of silicon + undercut of BOX
- Waveguides are suspended in air (no cladding)
- Target application: Mid-infrared photonics (3-8 um)
- Can also operate at telecom wavelengths with higher index contrast

### Layer Definitions

| Layer Name | GDS Layer | Description |
|------------|-----------|-------------|
| Si_Full Etch | 1 | 220 nm full etch (device patterning) |
| Anchor | 4 | Regions where BOX is NOT removed (mechanical support) |
| Release | 5 | BOX undercut window |
| FloorPlan | 100 | Die boundary |

### Key Concept

The BOX layer beneath the silicon waveguides is selectively removed (HF etch) to create free-standing / suspended silicon structures. Anchor regions keep the waveguide mechanically connected to the substrate.

### Design Rules

| Parameter | Value |
|-----------|-------|
| Minimum waveguide width | 300 nm |
| Minimum feature size | 100 nm |
| Maximum unsupported span | ~20 um |
| Anchor spacing | Every 10-20 um |
| Anchor width | > 1 um |
| Minimum spacing | 200 nm |
| Undercut margin | 2 um beyond release window |

### Waveguides

#### Suspended Strip Waveguide

| Parameter | Value |
|-----------|-------|
| Width | 300-500 nm (telecom) / 1-3 um (mid-IR) |
| Height | 220 nm |
| Surrounding medium | Air (n=1) |
| Effective index contrast | Very high (Si/Air) |
| Propagation loss (telecom) | ~5 dB/cm |

#### Subwavelength Grating (SWG) Waveguide

- Periodic silicon/air structure below diffraction limit
- Effective index tunable via duty cycle
- Lower effective index than strip waveguide
- Used for couplers and mode engineering

### Components

#### Suspended Grating Coupler

| Parameter | Value |
|-----------|-------|
| Center wavelength | 1550 nm (or mid-IR) |
| Coupling loss | ~7 dB |
| Bandwidth | ~20 nm |

#### Suspended 1x2 MMI

| Parameter | Value |
|-----------|-------|
| Insertion loss | < 0.5 dB |
| Imbalance | < 0.3 dB |

#### Suspended Y-Branch

| Parameter | Value |
|-----------|-------|
| Splitting ratio | 50:50 |
| Insertion loss | < 0.5 dB |

#### Suspended Directional Coupler

| Parameter | Value |
|-----------|-------|
| Gap | 150-300 nm |
| Excess loss | < 0.2 dB |

### Bends

| Minimum Radius | Loss (90-degree) |
|---------------|------------------|
| 3 um | < 0.1 dB |
| 5 um | < 0.05 dB |

### Ring Resonators

| Parameter | Value |
|-----------|-------|
| Radius | 3-10 um |
| Q factor | 5,000-30,000 |
| FSR (R=5 um) | ~24 nm |

### Design Considerations

- Mechanical anchors must be placed periodically to prevent waveguide collapse
- The release window must extend sufficiently beyond waveguide features to fully undercut the BOX
- Suspended structures are fragile; handle with care during processing
- No upper cladding is present; structures are exposed to air
- Well-suited for mid-infrared applications due to removal of silica (which absorbs beyond ~3.5 um)
- SWG structures can provide mode engineering and gradual effective index transitions

---

## Platform Comparison

| Property | SOI (220 nm) | SiN (300 nm) | Suspended Si |
|----------|-------------|-------------|-------------|
| Waveguide material | c-Si | Si3N4 | c-Si (air clad) |
| Core height | 220 nm | 300 nm | 220 nm |
| Cladding | SiO2 | SiO2 | Air |
| Loss (dB/cm) | ~3 | < 1 | ~5 |
| Min bend radius | 5 um | 20 um | 3 um |
| Active devices | Yes (PN, Ge PD) | No (heaters only) | No |
| Operating range | C/O-band | Vis to C-band | Telecom + Mid-IR |
| Lithography | e-beam | e-beam | e-beam |
| Key advantage | Active integration | Low loss | Mid-IR transparency |
