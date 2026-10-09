"""Foundry geometry and circuit regressions for the Si500 repair."""

from __future__ import annotations

import inspect
import runpy
import subprocess
import sys

import gdsfactory as gf
import jax
import jax.numpy as jnp
import klayout.db as kdb
import matplotlib
import numpy as np
import pytest
import sax

from cspdk.si500 import LAYER_STACK, PDK, cells, models, tech
from cspdk.si500.config import PATH


@pytest.fixture(autouse=True)
def activate_pdk():
    """Use Si500 for every layout built in this module."""
    PDK.activate()


def _foundry_region(filename, layer):
    layout = kdb.Layout()
    layout.read(str(PATH.gds / filename))
    region = kdb.Region(
        layout.top_cell().begin_shapes_rec(layout.find_layer(layer, 0))
    ).merged()
    return region, layout.dbu


@pytest.mark.parametrize(
    ("name", "kind"),
    [("mmi1x2", "2x1"), ("mmi1x2_rc", "2x1"), ("mmi2x2", "2x2"), ("mmi2x2_rc", "2x2")],
)
def test_mmi_matches_foundry(name, kind):
    """Match the foundry MMI polygon once its 10 um port straights are removed."""
    reference, dbu = _foundry_region(f"SOI500nm_1550nm_TE_RIB_{kind}_MMI.gds", 3)
    reference &= kdb.Region(reference.bbox().enlarged(-int(10 / dbu), 0))
    component = getattr(cells, name)()
    assert component.kcl.dbu == dbu
    actual = component.get_region("WG").merged()
    for region in (actual, reference):
        region.move(-region.bbox().left, -region.bbox().bottom)
    assert (actual ^ reference).is_empty()


@pytest.mark.parametrize(
    "name", ["grating_coupler_rectangular", "grating_coupler_rectangular_rc"]
)
@pytest.mark.parametrize(("layer", "number"), [("WG", 3), ("GRA", 6)])
def test_grating_matches_foundry(name, layer, number):
    """Waveguide outline and all 60 teeth match the foundry grating in place."""
    reference, _ = _foundry_region("SOI500nm_1550nm_TE_RIB_Grating_Coupler.gds", number)
    actual = getattr(cells, name)().get_region(layer).merged()
    assert actual.count() == reference.count()
    assert (actual ^ reference).is_empty()


@pytest.mark.parametrize("name", ["bend_circular", "bend_circular_rc"])
def test_bend_matches_foundry(name):
    """The default bend is the foundry R = 25 um circular arc.

    The foundry bend has ~26 nm straight stubs at both ends, so differences
    up to 30 nm wide are tolerated.
    """
    reference, dbu = _foundry_region("SOI500nm_1550nm_TE_RIB_90_Degree_Bend.gds", 3)
    component = getattr(cells, name)()
    assert component.info["radius"] == tech.TECH.radius_rc == 25
    # foundry bend runs north from (0..w, 0) and turns east: mirror about y = x
    actual = component.get_region("WG").merged().transformed(kdb.Trans.M45)
    actual.move(int(component.ports["o1"].width / 2 / dbu), 0)
    assert (actual ^ reference).sized(-15).is_empty()
    assert component.info["length"] == pytest.approx(25 * np.pi / 2, abs=1e-3)


def test_default_bends_are_circular():
    """MZIs and routes use circular bends that never go below the 25 um radius."""
    netlist = cells.mzi_rc(delta_length=50).get_netlist()
    bends = [
        v for v in netlist["instances"].values() if v["component"].startswith("bend")
    ]
    assert bends and all(v["component"] == "bend_circular" for v in bends)
    c = gf.Component()
    a = c << cells.straight_rc()
    b = c << cells.straight_rc()
    b.dmove((200, 100))
    route = tech.route_single(c, a.ports["o2"], b.ports["o1"])
    assert route.length > 0
    names = {inst.cell.name for inst in c.insts}
    assert any(n.startswith("bend_circular") for n in names), names
    assert not any(n.startswith("bend_euler") for n in names), names


def test_route_bundle_separation():
    """Bundles keep the 5 um waveguide spacing of the design guidelines."""
    for name in ("route_bundle", "route_bundle_rc", "route_bundle_ro"):
        strategy = PDK.routing_strategies[name]
        func = getattr(strategy, "func", strategy)
        assert inspect.signature(func).parameters["separation"].default == 5.0


def test_die_matches_packaging_template():
    """Gratings, loopbacks, pads and outline match the foundry template."""
    layout = kdb.Layout()
    layout.read(str(PATH.gds / "Cell0_SOI500_Full_1550nm_Packaging_Template.gds"))
    top = layout.top_cell()
    die = cells.die()
    for layer, number, tolerance in (
        ("WG", 3, 30),  # loopback arcs are discretised differently
        ("GRA", 6, 0),
        ("PAD", 41, 0),
        ("FLOORPLAN", 99, 0),
    ):
        reference = kdb.Region(top.begin_shapes_rec(layout.find_layer(number, 0)))
        xor = die.get_region(layer).merged() ^ reference.merged()
        assert xor.sized(-tolerance).is_empty(), layer
    assert len([p for p in die.ports if p.port_type == "optical"]) == 2 * 14
    assert len([p for p in die.ports if p.port_type == "electrical"]) == 2 * 31


def test_layer_stack_grating_split():
    """The 160 nm grating etch leaves 340 nm of silicon under the teeth."""
    levels = LAYER_STACK.layers
    assert levels["core"].thickness == pytest.approx(0.5)
    assert levels["slab"].thickness == pytest.approx(0.2)
    assert levels["grating"].thickness == pytest.approx(0.34)


@pytest.mark.parametrize("name", sorted(PDK.models))
@pytest.mark.parametrize("wl", [1.55, np.linspace(1.28, 1.58, 7)])
def test_registered_models_evaluate(name, wl):
    """Every advertised model supports scalar and swept wavelengths under JIT."""
    result = jax.jit(PDK.models[name])(wl=wl)
    for ports, value in result.items():
        assert isinstance(ports, tuple) and len(ports) == 2
        assert all(p.startswith("e" if name == "wire_corner" else "o") for p in ports)
        assert np.shape(value) == np.shape(wl)
        assert np.isfinite(value).all()


def test_model_registry():
    """Band aliases are registered; O-band has waveguide models only."""
    for prefix in ("straight", "taper", "bend_euler", "bend_circular"):
        for band in ("rc", "ro"):
            assert PDK.models[f"{prefix}_{band}"] is getattr(models, f"{prefix}_{band}")
    for prefix in ("mmi1x2", "mmi2x2", "coupler", "grating_coupler_rectangular"):
        assert f"{prefix}_rc" in PDK.models
        assert f"{prefix}_ro" not in PDK.models
        with pytest.raises(ValueError, match="C-band"):
            getattr(models, prefix)(wl=1.31, cross_section="xs_ro500")


@pytest.mark.parametrize("band,wl0,neff", [("rc", 1.55, 2.990), ("ro", 1.31, 3.099)])
def test_waveguide_phase_and_loss(band, wl0, neff):
    """Straights, tapers and bends use the mode-solved index and loss in dB/cm."""
    length, loss = 123.0, 20.0
    expected = 10 ** (-loss * length * 1e-4 / 20) * np.exp(
        2j * np.pi * neff * length / wl0
    )
    for prefix in ("straight", "taper"):
        result = getattr(models, f"{prefix}_{band}")(wl=wl0, length=length, loss=loss)
        np.testing.assert_allclose(result["o1", "o2"], expected, atol=1e-6)
    arc = 25 * np.pi / 2
    bend = models.bend_circular(wl=wl0, cross_section=f"xs_{band}500")
    np.testing.assert_allclose(
        bend["o1", "o2"], np.exp(2j * np.pi * neff * arc / wl0), atol=1e-6
    )


@pytest.mark.parametrize(
    "name", ["bend_circular", "bend_euler", "bend_s", "straight", "taper"]
)
def test_path_lengths_match_layout(name):
    """Waveguide models use the same path length as the drawn cell."""
    length = getattr(cells, name)().info["length"]
    reference = models.straight(wl=1.55, length=length)["o1", "o2"]
    model = getattr(models, name)(wl=1.55)["o1", "o2"]
    assert np.angle(model / reference) == pytest.approx(0, abs=0.05)


def test_grating_peak_and_bandwidth():
    """Foundry grating: 5.5 dB at 1.56 um with a 30 nm 1 dB bandwidth."""
    wl = np.array([1.56 - 0.015, 1.56, 1.56 + 0.015])
    power = np.abs(models.grating_coupler_rectangular(wl=wl)["o1", "o2"]) ** 2
    np.testing.assert_allclose(10 * np.log10(power), [-6.5, -5.5, -6.5], atol=1e-6)


def test_mzi_simulates_and_interferes():
    """Real layout netlists produce finite, passive, wavelength-dependent outputs."""
    wl = jnp.linspace(1.52, 1.58, 121)
    component = cells.mzi_rc(delta_length=100)
    circuit, _ = sax.circuit(component.get_netlist(), models=PDK.models)
    result = jax.block_until_ready(jax.jit(circuit)(wl=wl))
    assert {p for pair in result for p in pair} == {p.name for p in component.ports}
    cross = np.abs(np.asarray(result["o1", "o3"])) ** 2
    power = cross + np.abs(np.asarray(result["o1", "o2"])) ** 2
    assert np.all((power > 0.5) & (power <= 1.0))
    assert np.ptp(cross) > 0.3
    # free spectral range from the mode-solved group index
    minima = np.flatnonzero((cross[1:-1] < cross[:-2]) & (cross[1:-1] < cross[2:]))
    fsr = np.mean(np.diff(np.asarray(wl)[minima + 1]))
    assert fsr == pytest.approx(1.55**2 / (3.881 * 100), rel=0.05)


def test_model_ports_in_fresh_process():
    """Import Si500 after upstream models have cached the in/out port naming."""
    script = """
import numpy as np
import sax
import sax.models as sm
sax.set_port_naming_strategy('inout')
for model in (sm.straight, sm.mmi1x2, sm.mmi2x2, sm.grating_coupler):
    model(wl=np.array([1.55]))
from cspdk.si500 import PDK, cells
PDK.activate()
component = cells.mzi_rc(delta_length=100)
circuit, _ = sax.circuit(component.get_netlist(), models=PDK.models)
result = circuit(wl=np.linspace(1.52, 1.58, 121))
assert {p for pair in result for p in pair} == {'o1', 'o2', 'o3'}
assert np.ptp(np.abs(result['o1', 'o3']) ** 2) > 0.3
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=120)


@pytest.mark.parametrize(
    "sample",
    [
        "circuit_simulations_rc500.py",
        "circuit_simulations_rc500_with_routing.py",
        "get_route_rc500.py",
    ],
)
def test_samples_run(sample, monkeypatch):
    """The circuit and routing samples run headless against existing cells."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    monkeypatch.setattr(gf.Component, "show", lambda self, *a, **k: None)
    runpy.run_path(str(PATH.module / "samples" / sample), run_name="__main__")
    plt.close("all")
