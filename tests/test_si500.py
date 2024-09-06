"""Tests for netlists of all cells in the PDK."""

from __future__ import annotations

import pathlib

import gdsfactory as gf
import jsondiff
import pytest
from gdsfactory.difftest import difftest
from pytest_regressions.data_regression import DataRegressionFixture

from cspdk.si500 import PDK


@pytest.fixture(autouse=True)
def activate_pdk() -> None:
    """Activate the PDK and clear the cache."""
    PDK.activate()
    gf.clear_cache()


cells = PDK.cells
skip_test = {"wire_corner"}  # FIXME: why does this fail test_netlists?
cell_names = cells.keys() - skip_test
cell_names = [name for name in cell_names if not name.startswith("_")]
dirpath = pathlib.Path(__file__).absolute().with_suffix(".gds").parent / "gds_ref_si500"
dirpath.mkdir(exist_ok=True, parents=True)


def get_minimal_netlist(comp: gf.Component):
    """Get minimal netlist."""
    net = comp.get_netlist()

    def _get_instance(inst):
        return {
            "component": inst["component"],
            "settings": inst["settings"],
        }

    return {"instances": {i: _get_instance(c) for i, c in net["instances"].items()}}


def instances_without_info(net):
    """Get instances without info."""
    return {
        k: {
            "component": v.get("component", ""),
            "settings": v.get("settings", {}),
        }
        for k, v in net.get("instances", {}).items()
    }


@pytest.mark.parametrize("name", cells)
def test_cell_in_pdk(name) -> None:
    """Test cell in PDK."""
    c1 = gf.Component()
    c1.add_ref(gf.get_component(name))
    net1 = get_minimal_netlist(c1)

    c2 = gf.read.from_yaml(net1)
    net2 = get_minimal_netlist(c2)

    instances1 = instances_without_info(net1)
    instances2 = instances_without_info(net2)
    assert instances1 == instances2


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

    n.pop("connections", None)
    n.pop("warnings", None)
    yaml_str = c.write_netlist(n)
    c2 = gf.read.from_yaml(yaml_str)
    n2 = c2.get_netlist()

    d = jsondiff.diff(n, n2)
    d.pop("warnings", None)
    d.pop("connections", None)
    d.pop("ports", None)
    assert len(d) == 0, d
