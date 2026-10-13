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
    lyp_yaml = module / "layers.yaml"
    # Output directory of `python -m cspdk.ge_on_si.tech` (KLayout technology);
    # generated on demand, not shipped.
    klayout = module / "klayout"


PATH = Path()
