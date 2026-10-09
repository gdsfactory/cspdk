"""Foundry geometry, layer stack, routing and circuit regressions for Ge-on-Si."""

from __future__ import annotations

import pathlib
import subprocess
import sys

import gdsfactory as gf
import jax
import klayout.db as kdb
import numpy as np
import pytest
import sax

from cspdk.ge_on_si import PDK, cells, models, tech
from cspdk.ge_on_si.config import PATH

TEMPLATE = (
    pathlib.Path(__file__).parent.parent
    / "references"
    / "CORNERSTONE-GDSII-Template_All-Platforms_Jan_2024-1.gds"
)


@pytest.fixture(autouse=True)
def activate_pdk():
    """Use Ge-on-Si for every layout built in this module."""
    PDK.activate()


def _region(path, layer, cell=None):
    layout = kdb.Layout()
    layout.read(str(path))
    top = layout.cell(cell) if cell else layout.top_cell()
    region = kdb.Region(top.begin_shapes_rec(layout.find_layer(layer, 0))).merged()
    return region, layout.dbu


def _at_origin(region):
    return region.moved(-region.bbox().left, -region.bbox().bottom)


def test_straight_matches_foundry():
    """The 200 um rib waveguide is identical to the foundry cell."""
    reference, dbu = _region(PATH.gds / "Ge_on_Si_3800nm_TE_RIB_Waveguide.gds", 303)
    component = cells.straight(length=200)
    assert component.kcl.dbu == dbu
    actual = component.get_region("WG").merged()
    assert (_at_origin(actual) ^ _at_origin(reference)).is_empty()


def test_bend_circular_matches_foundry():
    """The PDK bend is the foundry r=300 um circular arc.

    The foundry polygon is drawn with ~1.4 deg chords (<= 23 nm sagitta), so the
    XOR is a set of thin slivers; anything thicker than 30 nm would be a shape
    difference. The previous euler default left ~3000 um^2 of XOR.
    """
    reference, dbu = _region(
        PATH.gds / "Ge_on_Si_3800nm_TE_RIB_90_Degree_Bend.gds", 303
    )
    component = cells.bend_circular()
    assert component.kcl.dbu == dbu
    # The foundry bend starts heading north at the origin; rotate 180 deg.
    actual = component.get_region("WG").merged().transformed(kdb.Trans.R180)
    xor = _at_origin(actual) ^ _at_origin(reference)
    assert xor.area() * dbu**2 < 10
    assert xor.sized(-15).is_empty()
    euler = cells.bend_euler().get_region("WG").merged().transformed(kdb.Trans.R180)
    assert (_at_origin(euler) ^ _at_origin(reference)).area() * dbu**2 > 1000


def test_die_matches_template():
    """The die outline and bleed strips match the foundry Ge-on-Si template cell."""
    component = cells.die()
    for layer, name in [(99, "FLOORPLAN"), (98, "BLEED")]:
        reference, _ = _region(TEMPLATE, layer, "Cell0_Ge_on_Si_Institution_Name")
        assert not reference.is_empty()
        actual = component.get_region(name).merged()
        assert (actual ^ reference).is_empty(), name


def test_bend_s_default_respects_min_radius():
    """The default S-bend stays above the 300 um minimum radius."""
    component = cells.bend_s()
    assert component.info["min_bend_radius"] >= tech.TECH.radius_rib


@pytest.mark.parametrize("router", ["route_single", "route_bundle"])
def test_routes_use_foundry_bend_without_noop_tapers(router):
    """Routing places the circular foundry bend and no 3.2 -> 3.2 um tapers."""
    component = gf.Component()
    starts, ends = [], []
    for index in range(1 if router == "route_single" else 2):
        start = component << cells.straight(length=10)
        end = component << cells.straight(length=10)
        start.dmove((0, 20 * index))
        end.dmove((2500, 1500 + 20 * index))
        starts.append(start.ports["o2"])
        ends.append(end.ports["o1"])
    if router == "route_single":
        PDK.routing_strategies[router](component, starts[0], ends[0])
    else:
        PDK.routing_strategies[router](component, starts, ends)
    names = [inst.cell.name for inst in component.insts]
    assert sum(name.startswith("bend_circular") for name in names) == 2 * len(starts)
    assert not [name for name in names if "taper" in name or "euler" in name]


def test_layer_stack_trenches():
    """The 1.8 um etch only hits 20 um trenches around 303 (plus 304/100)."""
    length = 50.0
    component = cells.straight(length=length)
    levels = tech.LAYER_STACK.layers
    area = {
        name: level.layer.get_shapes(component).area() * component.kcl.dbu**2
        for name, level in levels.items()
    }
    assert area["core"] == pytest.approx(3.2 * length)
    assert area["slab"] == pytest.approx((length + 40) * (3.2 + 40) - 3.2 * length)
    assert area["field"] == 0
    assert levels["core"].thickness == pytest.approx(3.0)
    assert levels["slab"].thickness == pytest.approx(1.2)

    with_etch = gf.Component()
    with_etch << component
    with_etch.add_polygon([(10, -1.6), (20, -1.6), (20, 1.6), (10, 1.6)], layer="WG_DF")
    core = levels["core"].layer.get_shapes(with_etch).area() * component.kcl.dbu**2
    assert core == pytest.approx(3.2 * (length - 10))

    die = cells.die()
    field = levels["field"].layer.get_shapes(die).area() * die.kcl.dbu**2
    assert field == pytest.approx(11470 * 15450)


def test_registered_models():
    """Every advertised cell model is registered."""
    assert set(PDK.models) == {
        "straight",
        "bend_euler",
        "bend_circular",
        "bend_s",
        "taper",
    }


@pytest.mark.parametrize("name", sorted(PDK.models))
@pytest.mark.parametrize("wl", [3.8, np.linspace(3.7, 3.9, 5)])
def test_registered_models_evaluate(name, wl):
    """Every model supports scalar and swept wavelengths under JIT."""
    result = jax.jit(PDK.models[name])(wl=wl)
    assert set(result) >= {("o1", "o2"), ("o2", "o1")}
    for ports, value in result.items():
        assert all(p in {"o1", "o2"} for p in ports)
        assert np.shape(value) == np.shape(wl)
        assert np.isfinite(value).all()


def test_straight_phase_and_loss():
    """Straight uses the solved index and the MPW loss bound at 3.8 um."""
    length = 1234.0
    result = models.straight(wl=3.8, length=length)
    expected = 10 ** (-models.LOSS_DB_CM * length * 1e-4 / 20) * np.exp(
        2j * np.pi * models.NEFF * length / 3.8
    )
    np.testing.assert_allclose(result["o1", "o2"], expected, atol=1e-10)
    assert models.LOSS_DB_CM <= 5
    # femwell solve of the foundry rib, see the models module docstring.
    assert models.NEFF == pytest.approx(3.953, abs=1e-3)
    assert models.NG == pytest.approx(4.144, abs=1e-3)


@pytest.mark.parametrize(
    "cell,kwargs",
    [
        ("bend_circular", {}),
        ("bend_circular", {"radius": 400, "angle": 180}),
        ("bend_euler", {}),
        ("bend_euler", {"radius": 350, "p": 0.3}),
        ("bend_s", {}),
        ("bend_s", {"size": (150.0, 10.0)}),
        ("taper", {"length": 25.0}),
    ],
)
def test_model_length_matches_cell(cell, kwargs):
    """Models derive the same centre-line length the cell reports."""
    component = getattr(cells, cell)(**kwargs)
    wl = np.array([3.75, 3.8, 3.85])
    np.testing.assert_allclose(
        PDK.models[cell](wl=wl, **kwargs)["o1", "o2"],
        models.straight(wl=wl, length=component.info["length"])["o1", "o2"],
        atol=5e-3,  # info["length"] is rounded to 1 nm
    )


def test_routed_circuit():
    """A routed straight -> bend -> straight netlist simulates with SAX."""
    component = gf.Component()
    start = component << cells.straight(length=20)
    end = component << cells.straight(length=20)
    end.drotate(90)
    end.dmove((800, 700))
    PDK.routing_strategies["route_single"](
        component, start.ports["o2"], end.ports["o1"]
    )
    component.add_port("o1", port=start.ports["o1"])
    component.add_port("o2", port=end.ports["o2"])
    netlist = component.get_netlist()
    assert any(
        inst["component"] == "bend_circular" for inst in netlist["instances"].values()
    )
    total = sum(inst.cell.info["length"] for inst in component.insts)
    circuit, _ = sax.circuit(netlist, models=PDK.models)
    wl = np.linspace(3.75, 3.85, 11)
    result = jax.jit(circuit)(wl=wl)
    np.testing.assert_allclose(
        result["o1", "o2"],
        models.straight(wl=wl, length=total)["o1", "o2"],
        atol=5e-3,  # info["length"] is rounded to 1 nm
    )


@pytest.mark.parametrize("strategy", ["inout", "optical"])  # codespell:ignore inout
def test_model_ports_in_fresh_process(strategy):
    """Import Ge-on-Si after upstream models have cached either port convention."""
    script = f"""
import numpy as np
import sax
import sax.models as sm
sax.set_port_naming_strategy({strategy!r})
sm.straight(wl=np.array([3.8]))
from cspdk.ge_on_si import PDK
for name, model in PDK.models.items():
    ports = {{p for pair in model(wl=np.array([3.8])) for p in pair}}
    assert ports == {{'o1', 'o2'}}, (name, ports)
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=120)
