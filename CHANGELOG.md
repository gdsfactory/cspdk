# CHANGELOG

<!-- towncrier release notes start -->

## [2.1.0](https://github.com/gdsfactory/cspdk/releases/tag/v2.1.0) - 2026-10-16

cspdk 2.1.0 checks design rules in the cells and fixes the cells that broke CORNERSTONE's rules. It also lets each flavour's DRC use its own rules, adds a submission checker for MPW Run 50 (designs due 11 November 2026), and adds a sample project for every flavour.

### Changes that affect existing designs

- **Cells reject gaps below the design-rule minimum (#406).** A gap parameter below its flavour's minimum now raises a `ValueError`.
  - A value exactly at the minimum passes.
  - `None`, meaning the cell's default geometry, isn't checked.
  - The minimums come from the CORNERSTONE design guidelines:

    | Flavour | `gap_min` (µm) | `grating_gap_min` (µm) |
    |---|---|---|
    | si220, si340, si500 | 0.2 | 0.25 |
    | sin200 | 0.2 | 0.2 |
    | sin300 | 0.25 | 0.25 |

- **Cells that broke CORNERSTONE's rules have new defaults (#387):**
  - si340 and si500 `grating_coupler_elliptical`, all variants: 350 nm unetched ridges, as si220 has had since 2.0.0.
  - si340 `grating_coupler_elliptical_so`: a new `trenches_extra_angle` parameter (default 9), set to 11 so the first trenches reach 200 nm beyond the waveguide.
  - sin300 `coupler` and `coupler_straight`: default `gap` 0.25 µm (was 0.236).
- **DRC uses each project's PDK name (#421).** cspdk no longer fixes `pdk = "cspdk"` in its DRC settings, so GDSFactory+ checks a project against the rules of the flavour set in its `pdk.name`, for example `cspdk.si340`.
- **si220 circulax heater (#386):**
  - `ThermalPhaseShifter` now uses the same phase convention as the SAX models, `exp(+1j·φ)`.
  - Its defaults now match SAX's `straight_heater_metal`: C-band strip neff 2.38, ng 4.30, dispersion, 3 dB/cm.
  - Mixed SAX and circulax circuits, such as the thermal MZI sample, now agree, and their results change.

### New

- **MPW Run 50 submission checker (#389):** `cspdk.si220.submission.check_submission(path)` checks the submission rules from the 50th-call design guideline that geometric DRC doesn't cover. A guide comes with it.
- **Sample projects for si340, si500, sin200, si_sus and ge_on_si (#407).** Every flavour now has one.
- **Docs (#405):** a models page per flavour, giving each SAX model's parameters, defaults and spectrum, plus published notebooks.

### Bug fixes

- si340 and sin300 `die` use `die_frame_with_pads` instead of the deprecated `die_with_pads`. The geometry is unchanged (#403).
- si220 `wire_corner45_straight` and sin200 `die_frame` no longer log kfactory duplicate-name errors (#412).
- The si220 sample notebook runs with GDSFactory+ 2.x (#414).

<details>
<summary>Packaging, CI and tests</summary>

- The release workflow creates the GitHub release from this file, and can be re-run safely (#388).
- CI workflow templates synced; code-security scanning (Semgrep, Trivy) now runs (#414).
- Tests:
  - every SAX model is finite, passive and reciprocal (#404);
  - regression data for every model's parameters and defaults (#409);
  - no duplicate cell names (#412);
  - a manifest of reference-file hashes (#413).

</details>


## [2.0.0](https://github.com/gdsfactory/cspdk/releases/tag/v2.0.0) - 2026-10-12

cspdk 2.0.0 merges the si220 C-band and O-band PDKs into one, matches si220 to the Cornerstone MPW Run 50 library (designs due 11 November 2026), and repairs the other six flavours against their foundry GDS. Existing designs will need changes: see **Breaking changes** below.

### Breaking changes

#### si220 is one PDK (#355)

`cspdk.si220.cband` and `cspdk.si220.oband` are removed. Use `cspdk.si220` and choose the band with the cross-section:

```python
from cspdk.si220 import PDK, cells

PDK.activate()
c = cells.mmi1x2(cross_section="strip_oband")  # strip_cband, rib_cband, strip_oband, rib_oband
```

The old cross-section names `strip` and `rib` are gone; use `strip_cband`, `rib_cband`, `strip_oband` or `rib_oband`.

#### si220 cells redrawn to the Run 50 library (#377)

Layouts that use these cells with default settings change:

- **`crossing`, `crossing_rib`:** centred on the origin (they were offset imported GDS). Placing them by absolute position, not by port, moves them.
- **`die()`:** defaults to the 11470 x 4900 um Run 50 design area (was 16000 x 3000 um).
- **`die_with_pads`:**
  - 16 gratings per side instead of 14.
  - Every optical port is renumbered, and the ports sit at x = +-4923 um.
  - Pads are 150 x 200 um, and the pad rows moved.
  - `edge_to_grating_distance` now measures to the far end of the gratings.
- **MMIs:**
  - Rib MMIs end in a 5 um straight, so their ports move 5 um outwards.
  - C strip `mmi1x2` is 31.8 um long (was 31).
  - O strip `mmi1x2` outputs are at y = +-1.525 um.
  - C strip `mmi2x2` ports are at y = +-1.015 um (was +-1).
- **`mzi_lattice`:** its outputs move with the MMIs.
- **Rectangular gratings:**
  - The fibre port `o2` moves.
  - The C-band rib period is 0.63 um.
  - The O band has 80 periods.
- **`grating_coupler_elliptical`:** the ridges are 350 nm instead of 315.
- **`taper_strip_to_ridge`, `trans_rib10/20/50`:** the port width follows the cross-section (0.45 um C, 0.40 um O) instead of a fixed 0.5 um.
- **`heater`:**
  - It is drawn to match the Run 50 Heater cell, a 248 x 182 um footprint.
  - It gains electrical ports `e1` and `e2`.
- **Text:**
  - `text_rectangular` defaults to the label layer (100/0).
  - `text_rectangular_multi_layer` defaults to `(WG, SLAB)`.

#### Renamed cross-sections in si340, si500, sin200 and sin300 (#376)

The old names were shared between flavours with different geometry. Rename them in code and schematics:

| Flavour | Old | New |
|---|---|---|
| si500 | `xs_rc`, `xs_ro` | `xs_rc500`, `xs_ro500` |
| si340 | `xs_sc`, `xs_so`, `xs_rc` | `xs_sc340`, `xs_so340`, `xs_rc340` |
| si500, si340, sin200, sin300 | `heater_metal` | `heater_metal_si500`, `_si340`, `_sin200`, `_sin300` |

#### Other flavours match their foundry GDS (#376)

Designs in si340, si500, sin200, sin300, si_sus and ge_on_si will differ from 1.4.5 in an XOR:

- the gratings, MMIs, bends, tapers, heaters and dies are redrawn to the foundry cells;
- the routing radii and separations are set to the foundry rules.

The pull request lists the changes per flavour.

#### Smaller breaking changes

- **doroutes (#347):** cspdk no longer depends on doroutes. si220's `route_astar` and `route_astar_metal` strategies are only available when doroutes is installed (`pip install doroutes`).
- **Material names (#315):** layer-stack materials now use `Si`, `SiO2`, `SiN`, `Ge`, `TiN` and `Al` (were `si`, `sin`, `ge`, `Aluminum`).
- **Schematic symbols (#314):** MZIs use the `mzi-1x2` / `mzi-2x2` symbol types instead of `mzi`.
- **Simulation results:** several model fixes change the numbers that si220 circuits produce:
  - ring FSR and Q, rib indices, the O-band models and heater physics (#362, #366, #372);
  - MMI, crossing, bend, taper, Y-splitter and grating losses, which now use the measured Run 50 data (#377).

### New

- **si220 Run 50 library (#377):**
  - New cells:
    - `ysplitter` and `grating_coupler_focusing_apodized`;
    - the graphene bolometer, EAM and PTE cells, each with a `_with_gc` variant;
    - `photodiode` and `photodiode_integrated`;
    - `laser_full` and `laser_half`, each with `_with_gc` and `_or_gc` variants;
    - `die_half`, `bend_circular`, `taper_mm2sm`, `taper_rib_to_strip` and `mzi_heater_loop`.
  - Named layers for laser, photodiode and graphene integration.
  - Models that use the measured Run 50 losses.
- **Models:**
  - SAX models for every flavour, which also work in circulax sweeps (#376).
  - A circulax heater model and schematic example (#356).
  - MZI lattice cell and models (#107).
- **si220 cross-sections:** customised cross-sections, such as `strip_oband(width=0.5)`, keep their band and strip/rib type in cells and models (#371).
- **Schematics and GDSFactory+:**
  - Logical electrical pins for SPICE export (#334).
  - Electrical connectivity on runtime PDKs (#352).
  - `die_with_pads` annotated as a die symbol (#351).
  - Routing constraints for differentiable optimisation (#215).
  - DRC settings in `pyproject.toml` (#357).
- **Sample projects:** a public sample project per flavour (si220 and sin300), in `cspdk--sample-projects/` (#368).
- **Docs:**
  - Cell pages for all seven flavours (#376).
  - An interactive layer-stack diagram (#320).
  - The Run 50 component notes in `references/components.md` (#377).

### Bug fixes

- si220 ring, coupler and MZI models (#362).
- si220 O-band models inside circuits (#366).
- Robust si220 models: couplers JIT-compile and differentiate, heater physics, rib defaults, circulax references (#372).
- Concrete model defaults so circulax DC sweeps work (#375).
- si220 model dispatch wrappers bound into the module namespace (#358).
- GDSFactory+ finds the si220 models (#368).
- Package keywords (#359) and PDF tab rendering in the docs (#318).

<details>
<summary>Packaging, CI and dependencies</summary>

- The wheel and sdist no longer contain samples, tests or reference files (#377); the build backend is hatchling (#311).
- gdsfactory ~=9.51.0 (#354), circulax >=0.2.3 (#333), gdsfactoryplus 1.8.20 (#361), zensical, pre-commit, mkdocs-material and GitHub Actions bumps (#324-#330, #337-#341, #360).
- CI: centralised workflow templates and the nyanlib generator (#336, #343, #345, #349); the full test suite runs in CI (#364); all ports are checked for Manhattan orientations (#332).

</details>


## [1.4.5](https://github.com/gdsfactory/cspdk/releases/tag/v1.4.5) - 2026-08-11

No significant changes.


## [1.4.4](https://github.com/gdsfactory/cspdk/releases/tag/v1.4.4) - 2026-06-24

No significant changes.


## [1.4.3](https://github.com/gdsfactory/cspdk/releases/tag/v1.4.3) - 2026-05-24

No significant changes.


## [1.4.2](https://github.com/gdsfactory/cspdk/releases/tag/v1.4.2) - 2026-02-03

- allow zero length straight rings [#199](https://github.com/gdsfactory/cspdk/pull/199)
- Bump actions/checkout from 5 to 6 [#190](https://github.com/gdsfactory/cspdk/pull/190)
- update changelog [#193](https://github.com/gdsfactory/cspdk/pull/193)
- remove samples from wheel [#194](https://github.com/gdsfactory/cspdk/pull/194)

## [1.4.1](https://github.com/gdsfactory/cspdk/releases/tag/v1.4.1) - 2026-01-05

- update gdsfactory 9.29.1

## [1.4.0](https://github.com/gdsfactory/cspdk/releases/tag/v1.4.0) - 2025-12-19

- add sbend router [#184](https://github.com/gdsfactory/cspdk/pull/184)
- fix route_bundle_sbend [#185](https://github.com/gdsfactory/cspdk/pull/185)
- fix typo [#183](https://github.com/gdsfactory/cspdk/pull/183)

## 1.3.0

- update gdsfactory9.22 [#177](https://github.com/gdsfactory/cspdk/pull/177)
- fix 3d stack [#166](https://github.com/gdsfactory/cspdk/pull/166)
- update gdsfactory9.22 [#177](https://github.com/gdsfactory/cspdk/pull/177)
- update gdsfactory9.21 [#175](https://github.com/gdsfactory/cspdk/pull/175)

## 1.2.2

- fix 3d stack [#166](https://github.com/gdsfactory/cspdk/pull/166)

## 1.2.1

- update gdsfactory9.16.3 [#165](https://github.com/gdsfactory/cspdk/pull/165)

## 1.2.0

- better models [#159](https://github.com/gdsfactory/cspdk/pull/159)
- fix bend models [#160](https://github.com/gdsfactory/cspdk/pull/160)

## 1.1.0

- add route samples [#150](https://github.com/gdsfactory/cspdk/pull/150)
- add missing init file [#151](https://github.com/gdsfactory/cspdk/pull/151)
- Update gdsfactory9.14 [#158](https://github.com/gdsfactory/cspdk/pull/158)

## 1.0.7

- add oband tests [#145](https://github.com/gdsfactory/cspdk/pull/145)
- fix coupler length [#144](https://github.com/gdsfactory/cspdk/pull/144)
- Fix models for Oband [#143](https://github.com/gdsfactory/cspdk/pull/143)

## 1.0.4

- use default port naming strategy for sax.models

## 1.0.3

- fix models and add tests for models [#138](https://github.com/gdsfactory/cspdk/pull/138)

## 1.0.2

- update gdsfactory 9.9.4

## 0.16.0

- update gdsfactory 9.4.0 [#110](https://github.com/gdsfactory/cspdk/pull/110)

## 0.15.0

- add heater models [#104](https://github.com/gdsfactory/cspdk/pull/104)

## 0.14.0

- fix release flow [#102](https://github.com/gdsfactory/cspdk/pull/102)
- update gdsfactory9 [#100](https://github.com/gdsfactory/cspdk/pull/100)

## 0.13.0

- add radius_loopback [#99](https://github.com/gdsfactory/cspdk/pull/99)
- update gdsfactory9 [#100](https://github.com/gdsfactory/cspdk/pull/100)

## 0.12.4

- update to gdsfactory 8.31.0

## 0.12.3

- update to gdsfactory 8.30.3 [#93](https://github.com/gdsfactory/cspdk/pull/93)

## 0.12.2

- better wire_corner [#89](https://github.com/gdsfactory/cspdk/pull/89)

## 0.12.1

- use uv in cicd
- update to gdsfactory 8.27.0

## 0.12.0

- add uv instructions [#84](https://github.com/gdsfactory/cspdk/pull/84)
- update to 8.25.1 [#82](https://github.com/gdsfactory/cspdk/pull/82)

## 0.11.5

- update changelog [#79](https://github.com/gdsfactory/cspdk/pull/79)
- update gdsfactory 8.21.0 [#78](https://github.com/gdsfactory/cspdk/pull/78)

## 0.11.4
- update gdsfactory 8.20.0 [#77](https://github.com/gdsfactory/cspdk/pull/77)

## 0.11.3
- update gdsfactory 8.15.0 [#72](https://github.com/gdsfactory/cspdk/pull/72)

## 0.11.2
- add type annotations for si220 [#68](https://github.com/gdsfactory/cspdk/pull/68)
- fix sample routing functions [#63](https://github.com/gdsfactory/cspdk/pull/63)
- Update gdsfactory 8.13.3 [#70](https://github.com/gdsfactory/cspdk/pull/70)

## 0.10.0

- add docstrings [#54](https://github.com/gdsfactory/cspdk/pull/54)
- cspdk cleanup [#51](https://github.com/gdsfactory/cspdk/pull/51)

## 0.9.2

- update gdsfactory to 8.4.0 [#50](https://github.com/gdsfactory/cspdk/pull/50)

## 0.9.1

- fix tests [#47](https://github.com/gdsfactory/cspdk/pull/47)
- update gdsfactory to 8.3.0 [#48](https://github.com/gdsfactory/cspdk/pull/48)

## 0.9.0

- Gdsfactory8 [#43](https://github.com/gdsfactory/cspdk/pull/43)
- add gplugins to pyproject [#46](https://github.com/gdsfactory/cspdk/pull/46)
- Fixes remove partials [#45](https://github.com/gdsfactory/cspdk/pull/45)
- fixing tech installation on windows [#35](https://github.com/gdsfactory/cspdk/pull/35)
- better values for cross_section [#39](https://github.com/gdsfactory/cspdk/pull/39)
- use approx version [#37](https://github.com/gdsfactory/cspdk/pull/37)
- update gdsfactory [#34](https://github.com/gdsfactory/cspdk/pull/34)

## 0.8.0

- add si500nm pdk [#32](https://github.com/gdsfactory/cspdk/pull/32)
- Add logo [#28](https://github.com/gdsfactory/cspdk/pull/28)
- better docs [#31](https://github.com/gdsfactory/cspdk/pull/31)
- make CORNERSTONE package support all the different pdks [#29](https://github.com/gdsfactory/cspdk/pull/29)
- update gplugins [#27](https://github.com/gdsfactory/cspdk/pull/27)
- Better docs [#24](https://github.com/gdsfactory/cspdk/pull/24)
- simplify routing [#22](https://github.com/gdsfactory/cspdk/pull/22)
- update gdsfactory and gplugins [#21](https://github.com/gdsfactory/cspdk/pull/21)
- Improve models [#16](https://github.com/gdsfactory/cspdk/pull/16)

## 0.7.0

- Update gdsfactory [#15](https://github.com/gdsfactory/cspdk/pull/15)
- Add routing strategies for specific cross sections [#14](https://github.com/gdsfactory/cspdk/pull/14)

## 0.5.0

- remove info from cells [#12](https://github.com/gdsfactory/cspdk/pull/12)


## 0.4.0

- Add netlist tests [#11](https://github.com/gdsfactory/cspdk/pull/11)
- more minor fixes [#9](https://github.com/gdsfactory/cspdk/pull/9)

## 0.3.0

- [improve models](https://github.com/gdsfactory/cspdk/pull/8)

## 0.2.0

- add models [#5](https://github.com/gdsfactory/cspdk/pull/5)
- Add ports to crossings [#7](https://github.com/gdsfactory/cspdk/pull/7)
