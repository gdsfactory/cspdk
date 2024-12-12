# CORNERSTONE PDK 0.11.5

![](https://i.imgur.com/V5Ukc6j.png)

[CORNERSTONE](https://www.cornerstone.sotonfab.co.uk/) Photonics PDK.

## Installation

### Installation for users

Use python3.10 or python3.11. We recommend [VSCode](https://code.visualstudio.com/) as an IDE.

If you don't have python installed on your system you can [download anaconda](https://www.anaconda.com/download/)

Once you have python installed, open Anaconda Prompt as Administrator and then install the latest gdsfactory using pip.

![anaconda prompt](https://i.imgur.com/eKk2bbs.png)
```
pip install cspdk --upgrade
```

Then you need to restart Klayout to make sure the new technology installed appears.

You can download the [code](https://github.com/gdsfactory/cspdk/archive/refs/heads/main.zip) and examples.

### Installation for contributors

We recommend `uv`

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then you can install with:

```bash
uv venv --python 3.11
uv sync --extra docs --extra dev
```

## Documentation

- [gdsfactory docs](https://gdsfactory.github.io/gdsfactory/)
