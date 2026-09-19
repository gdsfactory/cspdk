# Unified Si220 Bands Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the duplicated si220 cband and oband packages with one `cspdk.si220` PDK whose shared cells select band-specific behavior through suffixed cross-sections.

**Architecture:** Move the cband package into the si220 root as the shared implementation, then parameterize every observed cband/oband geometry difference by the resolved cross-section band. Consolidate models and schematics behind the same shared cell names, migrate all repository consumers, and delete both legacy band directories.

**Tech Stack:** Python 3.12, gdsfactory 9.51, kfactory, SAX/JAX, pytest, pytest-regressions, pre-commit, Zensical, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-19-unify-si220-bands-design.md`

## Global Constraints

- `cspdk.si220.cband` and `cspdk.si220.oband` must be removed without compatibility shims.
- Shared cells retain unsuffixed names and default to cband behavior.
- Public optical cross-sections are `strip_cband`, `rib_cband`, `strip_oband`, and `rib_oband`.
- Explicit cell arguments override inferred band defaults.
- The existing untracked `cspdk/rings2.gsch` and `cspdk/rings2.gsch.svg` files must not be modified or committed.
- Run `pre-commit run --all-files` successfully before every commit.

---

### Task 1: Unified package and technology contract

**Files:**
- Create: `tests/test_si220_unified.py`
- Replace: `cspdk/si220/__init__.py`
- Move/modify: `cspdk/si220/cband/config.py` → `cspdk/si220/config.py`
- Move/modify: `cspdk/si220/cband/tech.py` → `cspdk/si220/tech.py`

**Interfaces:**
- Produces: `cspdk.si220.PDK`, `cells`, `tech`, `strip_cband()`, `rib_cband()`, `strip_oband()`, `rib_oband()`, and internal `get_band(cross_section) -> Literal["cband", "oband"]`.

- [ ] **Step 1: Write failing public-contract tests**

```python
@pytest.mark.parametrize(
    ("name", "width"),
    [("strip_cband", 0.45), ("rib_cband", 0.45),
     ("strip_oband", 0.40), ("rib_oband", 0.40)],
)
def test_band_cross_section_width(name: str, width: float) -> None:
    PDK.activate()
    assert gf.get_cross_section(name).width == width

def test_unified_pdk_name() -> None:
    assert PDK.name == "cspdk.si220"
```

- [ ] **Step 2: Run tests and verify missing unified exports/cross-sections cause failure**

Run: `UV_NO_CONFIG=1 uv run pytest tests/test_si220_unified.py -q`

- [ ] **Step 3: Move the cband package foundation into `cspdk/si220`, register four cross-sections, implement `get_band`, and make cband the default**

- [ ] **Step 4: Run the focused tests and verify they pass**

Run: `UV_NO_CONFIG=1 uv run pytest tests/test_si220_unified.py -q`

### Task 2: Shared cells with band-aware defaults

**Files:**
- Move/modify: `cspdk/si220/cband/cells/` → `cspdk/si220/cells/`
- Test: `tests/test_si220_unified.py`

**Interfaces:**
- Consumes: `tech.get_band()` and the four cross-section names.
- Produces: one unsuffixed factory per cell; `cross_section="strip_cband"` is the optical default.

- [ ] **Step 1: Add parameterized failing tests for old literal defaults**

```python
@pytest.mark.parametrize(
    ("cross_section", "coupler_length", "mmi1x2_length", "mmi2x2_length"),
    [("strip_cband", 14.5, 31.0, 42.5),
     ("strip_oband", 20.0, 40.0, 53.5)],
)
def test_band_sensitive_cell_defaults(
    cross_section: str,
    coupler_length: float,
    mmi1x2_length: float,
    mmi2x2_length: float,
) -> None:
    assert cells.coupler(cross_section=cross_section).settings["length"] == coupler_length
    assert cells.mmi1x2(cross_section=cross_section).settings["length_mmi"] == mmi1x2_length
    assert cells.mmi2x2(cross_section=cross_section).settings["length_mmi"] == mmi2x2_length
```

- [ ] **Step 2: Run focused tests and verify the oband cases fail against cband-only defaults**

- [ ] **Step 3: Consolidate the cell modules and resolve every source diff through band-aware defaults**

Use `None` for optional band-sensitive inputs, resolve after `get_band(cross_section)`, and pass concrete values into gdsfactory factories. Resolve the observed differences in coupler length; rectangular-grating period; MMI lengths and gaps; heater port orientations; and ring coupling extensions. Retain the cband `mzi_lattice` factory in the shared registry.

- [ ] **Step 4: Add and pass geometry/port parity tests for both bands**

Construct corresponding old-reference expectations with literal widths, port counts, bounding boxes, and settings; do not derive expected values through `get_band`.

- [ ] **Step 5: Verify both variants coexist in one process with distinct names and widths**

```python
def test_bands_coexist_without_cell_cache_collision() -> None:
    cband = cells.straight(cross_section="strip_cband")
    oband = cells.straight(cross_section="strip_oband")
    assert cband.name != oband.name
    assert cband.ports["o1"].width == 0.45
    assert oband.ports["o1"].width == 0.40
```

### Task 3: Unified schematics and passive/active models

**Files:**
- Move/modify: `cspdk/si220/cband/_schematic.py` → `cspdk/si220/_schematic.py`
- Move/modify: `cspdk/si220/cband/models/` → `cspdk/si220/models/`
- Move/modify: `cspdk/si220/cband/active_models/` → `cspdk/si220/active_models/`
- Incorporate/delete: `cspdk/si220/oband/models.py`
- Modify: `tests/test_si220_unified.py`
- Replace: `tests/test_schematics_si220_cband.py`, `tests/test_schematics_si220_oband.py` → `tests/test_schematics_si220.py`

**Interfaces:**
- Produces: `get_models()` keyed by shared cell names; each optical model recognizes all four band cross-section names and dispatches to 1.55 µm cband or 1.31 µm oband behavior.

- [ ] **Step 1: Add failing model-dispatch tests with literal expected port keys and distinct band responses**

```python
def test_straight_model_dispatches_by_band() -> None:
    model = PDK.models["straight"]
    cband = model(wl=1.55, length=10, cross_section="strip_cband")
    oband = model(wl=1.31, length=10, cross_section="strip_oband")
    assert set(cband) == {("o1", "o2"), ("o2", "o1")}
    assert set(oband) == {("o1", "o2"), ("o2", "o1")}
    assert not np.allclose(cband["o1", "o2"], oband["o1", "o2"])
```

- [ ] **Step 2: Run the tests and verify missing unified models cause failure**

- [ ] **Step 3: Consolidate model modules and normalize cross-section aliases before dispatch**

The model dispatcher maps `strip_cband`/`rib_cband` to cband implementations and `strip_oband`/`rib_oband` to the old oband implementations. Unsupported cross-sections raise `ValueError` naming the accepted values.

- [ ] **Step 4: Consolidate schematic annotations and run both optical and electrical pin suites**

Run: `UV_NO_CONFIG=1 uv run pytest tests/test_si220_unified.py tests/test_schematics_si220.py tests/test_electrical_pins.py -q`

### Task 4: Assets, samples, metadata, docs, and generators

**Files:**
- Move/deduplicate: `cspdk/si220/{cband,oband}/gds/` → `cspdk/si220/gds/`
- Move/deduplicate: `cspdk/si220/{cband,oband}/samples/` → `cspdk/si220/samples/`
- Modify: `pyproject.toml`, `README.md`, `Makefile`, `docs/zensical.toml`
- Replace: `.github/write_cells_si220_cband.py`, `.github/write_cells_si220_oband.py` → `.github/write_cells_si220.py`
- Modify: `.github/write_layer_stack.py`, repository imports under `tests/`, `docs/`, and `cspdk--sample-projects/`

**Interfaces:**
- Produces: one installed PDK entry point named `cspdk.si220`, one generated cell-reference page, and one unified routing/sample import path.

- [ ] **Step 1: Add failing import/routing/material tests against `cspdk.si220`**

- [ ] **Step 2: Run those tests and verify legacy paths are still required before migration**

- [ ] **Step 3: Compare checksums of both GDS trees, deduplicate identical names, and rename any unequal collision with `_cband`/`_oband`**

- [ ] **Step 4: Move and deduplicate samples, explicitly selecting oband cross-sections in oband-derived examples**

- [ ] **Step 5: Update package entry points, Make targets, docs navigation/generators, README, sample-project models, and all internal imports**

- [ ] **Step 6: Confirm no tracked source references the removed packages**

Run: `git grep -nE "cspdk\.si220\.(cband|oband)" -- cspdk tests docs README.md pyproject.toml Makefile .github cspdk--sample-projects`

Expected: no matches.

### Task 5: Unified regression suite and deletion

**Files:**
- Replace: `tests/test_si220_cband.py`, `tests/test_si220_oband.py` → `tests/test_si220.py`
- Replace: `tests/test_si220_cband/`, `tests/test_si220_oband/`, `tests/gds_ref_si220/`, `tests/gds_ref_si220_oband/` with unified references
- Delete: `cspdk/si220/cband/`, `cspdk/si220/oband/`
- Modify: all test consumers of the old packages

**Interfaces:**
- Produces: one default PDK registry regression plus explicitly named cband/oband regression cases for band-sensitive factories.

- [ ] **Step 1: Parameterize unified regression helpers over default cells and explicit band variants**

- [ ] **Step 2: Run with reference regeneration to create missing unified references**

Run: `UV_NO_CONFIG=1 uv run pytest tests/test_si220.py --update-gds-refs --force-regen -q`

Expected: initial failures only report newly created regression references.

- [ ] **Step 3: Re-run without regeneration and verify all unified regressions pass**

Run: `UV_NO_CONFIG=1 uv run pytest tests/test_si220.py -q`

- [ ] **Step 4: Delete both legacy directories and run the focused si220, schematic, routing, material, and electrical tests**

### Task 6: Full verification, commit, and PR delivery

**Files:**
- Modify only files identified by preceding tasks and any formatter output within them.

- [ ] **Step 1: Run full repository pre-commit**

Run: `UV_NO_CONFIG=1 pre-commit run --all-files`

Expected: every hook passes. If hooks modify files, inspect, stage only in-scope changes, and rerun until clean.

- [ ] **Step 2: Run the supported full test suite**

Run: `UV_NO_CONFIG=1 make test`

- [ ] **Step 3: Build documentation and sample project**

Run: `UV_NO_CONFIG=1 uv run zensical build -f docs/zensical.toml`

Run: `UV_NO_CONFIG=1 uv run pytest cspdk--sample-projects/basic/tests -q`

- [ ] **Step 4: Review the diff against the design and confirm untracked schematics remain untouched**

- [ ] **Step 5: Run pre-commit again immediately before the implementation commit, commit, and push to PR #355**

- [ ] **Step 6: Mark PR #355 ready and monitor every required GitHub check; diagnose, fix, verify, and push until all required checks pass**
