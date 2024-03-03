# CornerStone Photonics PDK 0.0.1

CornerStone PDK

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

### Installation for developers

For developers you need to `git clone` the GitHub repository, fork it, git add, git commit, git push and merge request your changes.

```
git clone https://github.com/gdsfactory/cspdk.git
cd cspdk
pip install -e . pre-commit
pre-commit install
python install_tech.py
```

## Documentation

- [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gdsfactory/gdsfactory-photonics-training)
- [UBCpdk docs](https://gdsfactory.github.io/ubc/) and [code](https://github.com/gdsfactory/ubc)
- [gdsfactory docs](https://gdsfactory.github.io/gdsfactory/)
