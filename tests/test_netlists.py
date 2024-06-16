import gdsfactory as gf
import pytest

from cspdk.si500 import activate_pdk

# from cspdk.si220 import activate_pdk


activate_pdk()
cells = list(gf.get_active_pdk().cells)

Netlist = dict


def get_minimal_netlist(comp: gf.Component) -> Netlist:
    net = comp.get_netlist()

    def _get_instance(inst):
        return {
            "component": inst["component"],
            "settings": inst["settings"],
        }

    return {"instances": {i: _get_instance(c) for i, c in net["instances"].items()}}


def check_base_cell_in_pdk(name):
    c1 = gf.Component()
    c1.add_ref(gf.get_component(name))
    net1 = get_minimal_netlist(c1)

    c2 = gf.read.from_yaml(net1)
    net2 = get_minimal_netlist(c2)

    return net1["instances"] == net2["instances"]


@pytest.mark.parametrize("name", cells)
def test_cell_in_pdk(name):
    assert check_base_cell_in_pdk(name)


# @gf.cell
# def straight_cs(length=10, npoints=2, cross_section='xs_sc'):
#    return gf.components.straight(length=length, npoints=npoints, cross_section=cross_section,)

if __name__ == "__main__":
    activate_pdk()

    name = "mzi_sc"
    c1 = gf.Component()
    c1.add_ref(gf.get_component(name))
    net1 = get_minimal_netlist(c1)
    print(net1)
    gf.show(c1)

    c2 = gf.read.from_yaml(net1)
    net2 = get_minimal_netlist(c2)
    print(net2)

    print(check_base_cell_in_pdk("straight_sc"))
