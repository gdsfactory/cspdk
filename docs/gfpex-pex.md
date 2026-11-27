# PEX rules (gfpex)

Each CORNERSTONE platform with metal ships a default [gfpex](https://github.com/doplaydo/gf-pex) techfile for
parasitic extraction:

```python
from cspdk.si220.pex.rules import DEFAULT_PEX_RULES_CONFIG  # also si340, si500, sin200, sin300
```

`si_sus` and `ge_on_si` have no metal and no PEX rules.

## What is modelled

- **Layers:** `HEATER` (39/0, TiN heater filaments) and `PAD` (41/0, Ti/Al heater contact pads), with GDS pairs taken
  from each platform's `LAYER`.
- **Connection:** there is no via. The pads touch the filaments where they overlap. A computed contact layer,
  `HEATER_PAD_CON`, on the pad drawing layer joins the two in gfpex's resistance network.
- **Stack convention (gf-pex/KPEX):** z = 0 is the top of the Si substrate. The BOX is the field oxide, the SiO2
  cladding sits above it, the filament sits on the cladding top (BOX + core + cladding), the pad sits on the
  filament, and there is air above.

The shared builder is `cspdk/_pex.py`. Each `cspdk/<platform>/pex/rules.py` holds that platform's sourced thicknesses.

## Provenance

Every number carries a tag:

| Tag | Meaning |
|---|---|
| `FOUNDRY` | Stated by a CORNERSTONE design guideline (BOX, core and cladding thicknesses) |
| `DERIVED` | Computed from other values: area capacitance 8.854·k/Σ(t/k), Rs = ρ/t, parallel-plate sidewall 8.854·k·t |
| `ASSUMED` | Borrowed from IHP SG13G2 in gf-pex (dielectric k, `side_halo`), from the literature (TiN and Al resistivity), or from cspdk's heater/pad thickness placeholders |
| `PALACE-FIT` | Pending: fringe, perimeter and side-overlap capacitance from gf-pex `benchmarks/palace_benchmark.py` |

The CORNERSTONE design guidelines give no electrical data: no sheet resistance, no dielectric constant, and no metal
thickness. Treat these rules as estimates until foundry data or Palace fits replace the `ASSUMED` and `TODO` values.

## Tests

gfpex is optional and not a dependency of cspdk. The tests skip without it:

```bash
uv pip install <gfpex wheel> networkx "kfnetlist @ git+https://github.com/gdsfactory/kfnetlist"
make test-pex
```

gfpex is private: installing it needs read access to `doplaydo/gf-pex`, git authenticated for github.com
(for example `gh auth login && gh auth setup-git`) and a Rust toolchain to build it.

The `Test PEX rules with gfpex` workflow runs the same tests on demand. It needs a `GFPEX_TOKEN` secret.
