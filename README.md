
# CORNERSTONE PDK 1.4.4

<!-- BADGES:START -->
[![Docs](https://github.com/gdsfactory/cspdk/actions/workflows/pages.yml/badge.svg)](https://github.com/gdsfactory/cspdk/actions/workflows/pages.yml)
[![Tests](https://github.com/gdsfactory/cspdk/actions/workflows/test_code.yml/badge.svg)](https://github.com/gdsfactory/cspdk/actions/workflows/test_code.yml)
[![DRC](https://github.com/gdsfactory/cspdk/raw/badges/drc.svg)](https://github.com/gdsfactory/cspdk/actions/workflows/drc.yml)
[![Model Regression](https://github.com/gdsfactory/cspdk/actions/workflows/model_regression.yml/badge.svg)](https://github.com/gdsfactory/cspdk/actions/workflows/model_regression.yml)
[![Test Coverage](https://github.com/gdsfactory/cspdk/raw/badges/coverage.svg)](https://github.com/gdsfactory/cspdk/actions/workflows/test_coverage.yml)
[![Model Coverage](https://github.com/gdsfactory/cspdk/raw/badges/model_coverage.svg)](https://github.com/gdsfactory/cspdk/actions/workflows/model_coverage.yml)
[![Issues](https://github.com/gdsfactory/cspdk/raw/badges/issues.svg)](https://github.com/gdsfactory/cspdk/issues)
[![PRs](https://github.com/gdsfactory/cspdk/raw/badges/prs.svg)](https://github.com/gdsfactory/cspdk/pulls)
<!-- BADGES:END -->


![](https://i.imgur.com/V5Ukc6j.png)

[CORNERSTONE](https://www.cornerstone.sotonfab.co.uk/) Photonics PDK.

## Supported Process Variants

| Module | Process | Waveguide | Wavelength | Heaters |
|--------|---------|-----------|------------|---------|
| `cspdk.si220.cband` | SOI 220nm | Strip + Rib | C-band (1550nm) | Yes |
| `cspdk.si220.oband` | SOI 220nm | Strip + Rib | O-band (1310nm) | Yes |
| `cspdk.si340` | SOI 340nm | Strip + Rib | C-band / O-band | Yes |
| `cspdk.si500` | SOI 500nm | Rib | C-band / O-band | Yes |
| `cspdk.sin300` | SiN 300nm | Strip | C-band / O-band | Yes |
| `cspdk.sin200` | SiN 200nm | Strip | Visible (520/638/780nm) | Yes |
| `cspdk.ge_on_si` | Ge-on-Si | Rib | Mid-IR (3800nm) | No |
| `cspdk.si_sus` | Suspended Si | Suspended | Mid-IR (3800nm) | No |

## Installation

We recommend `uv`

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Installation for users

Use python 3.11, 3.12 or 3.13. We recommend [VSCode](https://code.visualstudio.com/) as an IDE.

```
uv pip install cspdk --upgrade
```

Then you need to restart Klayout to make sure the new technology installed appears.

### Installation for contributors


Then you can install with:

```bash
git clone https://github.com/gdsfactory/cspdk.git
cd cspdk
uv venv --python 3.12
uv sync --extra docs --extra dev
```

## Documentation

- [gdsfactory docs](https://gdsfactory.github.io/gdsfactory/)

## Pre-commit

```bash
make pre-commit
```

## Release

1. Bump the version:

```bash
tbump 0.0.1
```
This triggers the release workflow that builds wheels and uploads them.

2. Create a pull request with the updated changelog since last release.
