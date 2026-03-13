
# CORNERSTONE PDK 1.4.2

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

1. Update the changelog in `.changelog.d/` with entries describing the changes. Make sure the changelog is clear and readable for customers.
2. Bump the version:

```bash
tbump 0.0.1
```

3. Push the tag:

```bash
git push --tags
```

4. Create a pull request with the updated changelog and version bump. This triggers the release workflow that builds wheels and uploads them.
