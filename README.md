# CORNERSTONE PDK 0.13.0

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

Use python3.10, 3.11 or 3.12. We recommend [VSCode](https://code.visualstudio.com/) as an IDE.

```
uv pip install cspdk --upgrade
```

Then you need to restart Klayout to make sure the new technology installed appears.

### Installation for contributors


Then you can install with:

```bash
git clone https://github.com/gdsfactory/cspdk.git
cd cspdk
uv venv --python 3.11
uv sync --extra docs --extra dev
```

## Documentation

- [gdsfactory docs](https://gdsfactory.github.io/gdsfactory/)
