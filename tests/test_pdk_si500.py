import pathlib

import gdsfactory as gf
import pytest
from gdsfactory.difftest import difftest
from pytest_regressions.data_regression import DataRegressionFixture

from cspdk.si500 import PDK


@pytest.fixture(autouse=True)
def activate_pdk():
    PDK.activate()
    gf.clear_cache()


cells = PDK.cells

skip_test = {"import_gds"}

cell_names = set(cells.keys()) - set(skip_test)
cell_names = [name for name in cell_names if not name.startswith("_")]
dirpath = pathlib.Path(__file__).absolute().with_suffix(".gds").parent / "gds_ref_si220"
dirpath.mkdir(exist_ok=True, parents=True)


@pytest.mark.parametrize("component_name", cell_names)
def test_gds(component_name: str) -> None:
    """Avoid regressions in GDS geometry shapes and layers."""
    component = cells[component_name]()
    difftest(component, test_name=component_name, dirpath=dirpath)


@pytest.mark.parametrize("component_name", cell_names)
def test_settings(component_name: str, data_regression: DataRegressionFixture) -> None:
    """Avoid regressions when exporting settings."""
    component = cells[component_name]()
    data_regression.check(component.to_dict())
