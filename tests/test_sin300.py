"""Tests for netlists of all cells in the PDK."""

from __future__ import annotations

import pathlib

import gdsfactory as gf
import jsondiff
import kfactory as kf
import numpy as np
import pytest
from conftest import difftest
from pytest_regressions.data_regression import DataRegressionFixture

from cspdk.sin300 import PDK


@pytest.fixture(autouse=True)
def activate_pdk() -> None:
    """Activate the PDK and clear the cache."""
    PDK.activate()


cells = PDK.cells
skip_test = {"coupler_symmetric", "die", "die_nc", "die_no"}
cell_names = cells.keys() - skip_test
cell_names = [name for name in cell_names if not name.startswith("_")]
dirpath = (
    pathlib.Path(__file__).absolute().with_suffix(".gds").parent / "gds_ref_sin300"
)
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
def test_cell_in_pdk(name):
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
    data_regression.check(component.to_dict(with_ports=True))


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

    cis = list(c.kcl.each_cell_top_down())
    for ci in cis:
        gf.kcl.dkcells[ci].delete()

    c2 = gf.read.from_yaml(yaml_str)
    n2 = c2.get_netlist()
    d = jsondiff.diff(n, n2)
    d.pop("warnings", None)
    d.pop("connections", None)
    d.pop("ports", None)
    assert len(d) == 0, d

    cis = list(c.kcl.each_cell_top_down())
    for ci in cis:
        gf.kcl.dkcells[ci].delete()


@pytest.mark.parametrize("component_name", cell_names)
def test_optical_port_positions(component_name: str) -> None:
    """Ensure that optical ports are positioned correctly."""
    component = cells[component_name]()
    if isinstance(component, gf.ComponentAllAngle):
        new_component = gf.Component()
        kf.VInstance(component).insert_into_flat(new_component, levels=0)
        new_component.add_ports(component.ports)
        component = new_component
    for port in component.ports:
        if port.port_type == "optical":
            port_layer = port.layer
            port_width = port.width
            port_position = port.center
            port_angle = port.orientation
            # get the edges of the optical layer corresponding to the port
            cs_region = kf.kdb.Region(component.begin_shapes_rec(port_layer))
            optical_edges = cs_region.edges()

            # get a small marker around the port position
            tolerance = 0.001
            poly = kf.kdb.DBox(-tolerance, -tolerance, tolerance, tolerance)
            dbu_in_um = port.kcl.to_um(1)
            port_marker = (
                kf.kdb.DPolygon(poly).transformed(port.dcplx_trans).to_itype(dbu_in_um)
            )
            port_marker_region = kf.kdb.Region(port_marker)

            # get the physical port edge that interacts with the marker
            # assert that there is exactly one edge interacting with the marker
            # and that it has the correct length
            interacting_edges = optical_edges.interacting(port_marker_region)
            if interacting_edges.is_empty():
                raise AssertionError(
                    f"No optical edge found for port {port.name} at position {port_position} with width {port_width} and angle {port_angle}."
                )
            port_edge = next(iter(interacting_edges.each()))
            edge_length = port_edge.length() * 0.001
            if not np.isclose(edge_length, port_width, atol=1e-3):
                raise AssertionError(
                    f"Port {port.name} has width {port_width}, but the optical edge length is {edge_length}."
                )


if __name__ == "__main__":
    component_type = "mzi_no"
    c = cells[component_type]()
    n = c.get_netlist()
    n.pop("connections", None)

    yaml_str = c.write_netlist(n)
    c2 = gf.read.from_yaml(yaml_str)
    c2.show()
    n2 = c2.get_netlist()
    d = jsondiff.diff(n, n2)
    d.pop("warnings", None)
    d.pop("connections", None)
    d.pop("ports", None)
    assert len(d) == 0, d
