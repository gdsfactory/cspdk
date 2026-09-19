# Unified Si220 PDK Design

## Goal

Replace the duplicated `cspdk.si220.cband` and `cspdk.si220.oband` packages with one `cspdk.si220` PDK. Cell names remain shared, while optical cross-section names identify the operating band: `strip_cband`, `strip_oband`, `rib_cband`, and `rib_oband`.

## Public API

The supported entry point becomes:

```python
from cspdk.si220 import PDK, cells, tech
```

The PDK name is `cspdk.si220`. The old `cspdk.si220.cband` and `cspdk.si220.oband` packages are removed without compatibility shims. Packaging metadata exposes only the unified entry point.

Cell names remain unsuffixed, including `straight`, `bend_euler`, `mmi1x2`, `mmi2x2`, `coupler`, `ring_single`, and `grating_coupler_rectangular`. Their optical `cross_section` argument selects the band. The default is `strip_cband`, preserving the current cband behavior for callers that omit the argument.

The unified technology module registers these public optical cross-sections:

- `strip_cband`: 0.45 µm strip width
- `rib_cband`: 0.45 µm rib width
- `strip_oband`: 0.40 µm strip width
- `rib_oband`: 0.40 µm rib width

Electrical cross-sections and the layer map remain shared and unsuffixed.

## Band-Aware Cell Defaults

Band choice is derived from the resolved optical cross-section name. A small internal helper classifies the four registered cross-sections as cband or oband. A custom or otherwise unclassified cross-section uses cband defaults unless the caller supplies the relevant geometric arguments explicitly.

Cells whose existing implementations differ only by imports are consolidated directly. Cells with band-dependent defaults use `None` sentinels and resolve the appropriate value at construction time. This includes at least:

- directional-coupler length;
- rectangular grating period and wavelength-sensitive defaults;
- MMI lengths and gaps;
- heater port orientations;
- ring coupling extensions;
- any other difference identified by the migration parity tests.

Explicit caller-provided values always override inferred band defaults. `@gf.cell` remains on the public factory so resolved settings and generated names remain deterministic.

## Models and Schematics

The two passive model implementations move into one `cspdk.si220.models` package. Public model names remain aligned with shared cell names. Each model dispatches using the netlist's cross-section setting:

- cband cross-sections use the current cband numerical/interpolated models and 1.55 µm defaults;
- oband cross-sections use the current oband analytical models and 1.31 µm defaults.

The existing cband active-model support remains available from `cspdk.si220.active_models`. Where an active model is only valid for cband, it must reject an oband selection clearly rather than silently using cband parameters.

One `_schematic.py` module annotates the shared cells. It retains the union of valid logical pins and model declarations. Where old band packages used different port layouts, the unified cell's actual ports determine the annotation, and tests cover both band selections.

## Files, Assets, and Samples

The package layout becomes:

```text
cspdk/si220/
├── __init__.py
├── _schematic.py
├── active_models/
├── cells/
├── config.py
├── gds/
├── models/
├── samples/
└── tech.py
```

Assets from both old packages move into the unified `gds` directory. Same-named identical assets are deduplicated. A same-named asset with different bytes must be renamed by band and its importer updated so no data is lost.

Samples are deduplicated and updated to import `cspdk.si220`. Band-specific examples select the appropriate suffixed cross-section explicitly. Documentation generators, layer-stack generation, routing tests, project metadata, README tables, and navigation point only to the unified package.

## Tests and Migration Verification

Tests move to a unified `tests/test_si220.py` suite. Registry tests cover the default cband cell once, while parameterized parity tests construct both band variants for every band-sensitive cell.

Before deleting the old packages, migration tests capture their observable contracts. After consolidation, tests verify:

- all four cross-sections are registered with the expected widths;
- shared cells default to cband geometry;
- selecting an oband cross-section reproduces the old oband geometry and ports;
- selecting a cband cross-section reproduces the old cband geometry and ports;
- band-sensitive defaults resolve correctly and explicit overrides win;
- passive SAX models dispatch to the correct band and produce the expected port keys;
- cband and oband cells can be created in the same Python process without the old KCLayout name-cache collision;
- old band packages and entry points are absent;
- GDS, settings, netlist, schematic, electrical-pin, routing, material, sample-project, and documentation checks remain valid.

Regression reference files are regenerated under unified names that include the band when geometry differs.

## Validation and Delivery

Before every commit, `pre-commit run --all-files` must pass as required by `AGENTS.md`. The supported `make test` target and focused schematic, routing, model, documentation, and sample-project checks must pass locally. A pull request is opened from the feature branch, and any CI failures are investigated and fixed until every required GitHub check passes.

The user's existing untracked `cspdk/rings2.gsch` and `cspdk/rings2.gsch.svg` files are not modified or committed.
