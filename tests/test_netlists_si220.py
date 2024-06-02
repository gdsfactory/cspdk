from __future__ import annotations

import gdsfactory as gf
import jsondiff
import pytest
from omegaconf import OmegaConf
from pytest_regressions.data_regression import DataRegressionFixture

from cspdk.si220 import PDK


@pytest.fixture(autouse=True)
def activate_pdk():
    PDK.activate()


cells = PDK.cells
skip_test = set()
cell_names = cells.keys() - skip_test
cell_names = [name for name in cell_names if not name.startswith("_")]


@pytest.mark.parametrize("component_type", cell_names)
def test_netlists(
    component_type: str,
    data_regression: DataRegressionFixture,
    check: bool = True,
    component_factory=cells,
) -> None:
    """Write netlists for hierarchical circuits.

    Checks that both netlists are the same jsondiff does a hierarchical diff.

    Component -> netlist -> Component -> netlist

    """
    c = component_factory[component_type]()
    n = c.get_netlist()
    if check:
        data_regression.check(n)

    c.delete()
    yaml_str = OmegaConf.to_yaml(n, sort_keys=True)
    c2 = gf.read.from_yaml(yaml_str)
    n2 = c2.get_netlist()
    n.pop("warnings", None)
    n2.pop("warnings", None)
    d = jsondiff.diff(n, n2)
    assert len(d) == 0, d


if __name__ == "__main__":
    test_netlists("die_sc", None, False)
