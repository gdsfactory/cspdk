
# CORNERSTONE PDK 1.4.2

<!-- BADGES:START -->
[![Docs](https://github.com/gdsfactory/cspdk/actions/workflows/pages.yml/badge.svg)](https://github.com/gdsfactory/cspdk/actions/workflows/pages.yml)
[![Tests](https://github.com/gdsfactory/cspdk/actions/workflows/test_code.yml/badge.svg)](https://github.com/gdsfactory/cspdk/actions/workflows/test_code.yml)
[![DRC](https://github.com/gdsfactory/cspdk/actions/workflows/drc.yml/badge.svg)](https://github.com/gdsfactory/cspdk/actions/workflows/drc.yml)
[![Model Regression](https://github.com/gdsfactory/cspdk/actions/workflows/model_regression.yml/badge.svg)](https://github.com/gdsfactory/cspdk/actions/workflows/model_regression.yml)
[![Test Coverage](https://github.com/gdsfactory/cspdk/raw/badges/coverage.svg)](https://github.com/gdsfactory/cspdk/actions/workflows/test_coverage.yml)
[![Model Coverage](https://github.com/gdsfactory/cspdk/raw/badges/model_coverage.svg)](https://github.com/gdsfactory/cspdk/actions/workflows/model_coverage.yml)
[![Issues](https://github.com/gdsfactory/cspdk/raw/badges/issues.svg)](https://github.com/gdsfactory/cspdk/issues)
[![PRs](https://github.com/gdsfactory/cspdk/raw/badges/prs.svg)](https://github.com/gdsfactory/cspdk/pulls)
<!-- BADGES:END -->


![](https://i.imgur.com/V5Ukc6j.png)

[CORNERSTONE](https://www.cornerstone.sotonfab.co.uk/) Photonics PDK.

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
