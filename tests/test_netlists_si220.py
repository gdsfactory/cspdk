"""Description: Test netlists for all cells in the PDK."""

from __future__ import annotations

import gdsfactory as gf
import jsondiff
import pytest
from pytest_regressions.data_regression import DataRegressionFixture

from cspdk.si220 import PDK
from cspdk.si220.cells import wire_corner


@pytest.fixture(autouse=True)
def activate_pdk() -> None:
    """Activate PDK."""
    PDK.activate()
    gf.clear_cache()


cells = PDK.cells
skip_test = {"wire_corner"}  # FIXME: why does this fail test_netlists?
cell_names = cells.keys() - skip_test
cell_names = [name for name in cell_names if not name.startswith("_")]


def get_minimal_netlist(comp: gf.Component):
    """Get minimal netlist from a component."""
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
def test_cell_in_pdk(name):
    """Test that cell is in the PDK."""
    c1 = gf.Component()
    c1.add_ref(gf.get_component(name))
    net1 = get_minimal_netlist(c1)

    c2 = gf.read.from_yaml(net1)
    net2 = get_minimal_netlist(c2)

    instances1 = instances_without_info(net1)
    instances2 = instances_without_info(net2)
    assert instances1 == instances2


@pytest.mark.parametrize("component_type", cell_names)
def test_netlists(
    component_type: str,
    data_regression: DataRegressionFixture,
    check: bool = True,
) -> None:
    """Write netlists for hierarchical circuits.

    Checks that both netlists are the same jsondiff does a hierarchical diff.

    Component -> netlist -> Component -> netlist

    """
    c = cells[component_type]()
    n = c.get_netlist()
    if check:
        data_regression.check(n)

    n.pop("connections", None)
    n.pop("warnings", None)
    c.delete()
    yaml_str = c.write_netlist(n)
    c2 = gf.read.from_yaml(yaml_str)
    n2 = c2.get_netlist()
    d = jsondiff.diff(n, n2)
    d.pop("warnings", None)
    d.pop("ports", None)
    assert len(d) == 0, d


if __name__ == "__main__":
    c = wire_corner()
    n = c.get_netlist()
    n.pop("connections", None)
    c.delete()
    print(n)
    c2 = gf.read.from_yaml(n)
    n2 = c2.get_netlist()
    d = jsondiff.diff(n, n2)
    assert len(d) == 0, d
