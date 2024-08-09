"""Test the PDK components and settings."""

import pathlib

import gdsfactory as gf
import pytest
from gdsfactory.difftest import difftest
from pytest_regressions.data_regression import DataRegressionFixture

from cspdk.si220 import PDK


@pytest.fixture(autouse=True)
def activate_pdk() -> None:
    """Activate the PDK before running the tests."""
    PDK.activate()
    gf.clear_cache()


cells = PDK.cells

skip_test = {
    "import_gds",
    "pack_doe",
    "pack_doe_grid",
    "add_pads_top",
    "add_fiber_single_sc",
    "add_fiber_single_so",
    "add_fiber_array_sc",
    "add_fiber_array_so",
}

cell_names = set(cells.keys()) - set(skip_test)
cell_names = sorted([name for name in cell_names if not name.startswith("_")])
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
