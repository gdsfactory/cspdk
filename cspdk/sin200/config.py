"""Store path."""

__all__ = ["PATH"]

import pathlib

cwd = pathlib.Path.cwd()
cwd_config = cwd / "config.yml"
module = pathlib.Path(__file__).parent.absolute()
repo = module.parent.parent


class Path:
    module = module
    repo = repo
    gds = module / "gds"
    # output directory for `python -m cspdk.sin200.tech` (KLayout technology);
    # no KLayout technology files are shipped for this flavour.
    klayout = module / "klayout"
    lyp_yaml = module / "layers.yaml"


PATH = Path()
