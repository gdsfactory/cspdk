"""Foundry geometry and circuit regressions for the Si340 repair."""

from __future__ import annotations

import inspect
import subprocess
import sys

import gdsfactory as gf
import jax
import jax.numpy as jnp
import klayout.db as kdb
import numpy as np
import pytest
import sax

from cspdk.si340 import LAYER_STACK, PDK, cells, models, tech
from cspdk.si340.config import PATH


@pytest.fixture(autouse=True)
def activate_pdk():
    """Use Si340 for every layout built in this module."""
    PDK.activate()


def _foundry_region(filename, layer):
    layout = kdb.Layout()
    layout.read(str(PATH.gds / filename))
    region = kdb.Region(
        layout.top_cell().begin_shapes_rec(layout.find_layer(layer, 0))
    ).merged()
    return region, layout.dbu


@pytest.mark.parametrize(
    ("name", "wavelength", "kind"),
    [
        ("mmi1x2", 1550, "2x1"),
        ("mmi1x2_sc", 1550, "2x1"),
        ("mmi1x2_so", 1310, "2x1"),
        ("mmi2x2", 1550, "2x2"),
        ("mmi2x2_sc", 1550, "2x2"),
        ("mmi2x2_so", 1310, "2x2"),
    ],
)
def test_mmi_matches_foundry(name, wavelength, kind):
    """Match the foundry MMI polygon once its 10 um port straights are removed."""
    reference, dbu = _foundry_region(
        f"SOI340nm_{wavelength}nm_TE_STRIP_{kind}_MMI.gds", 3
    )
    reference &= kdb.Region(reference.bbox().enlarged(-int(10 / dbu), 0))
    component = getattr(cells, name)()
    assert component.kcl.dbu == dbu
    actual = component.get_region("WG").merged()
    for region in (actual, reference):
        region.move(-region.bbox().left, -region.bbox().bottom)
    assert (actual ^ reference).is_empty()


@pytest.mark.parametrize(
    ("name", "wavelength"),
    [
        ("grating_coupler_rectangular", 1550),
        ("grating_coupler_rectangular_sc", 1550),
        ("grating_coupler_rectangular_so", 1310),
    ],
)
@pytest.mark.parametrize(("layer", "number"), [("WG", 3), ("GRA", 6)])
def test_grating_matches_foundry(name, wavelength, layer, number):
    """Waveguide outline and all 60 teeth match the foundry grating in place."""
    reference, _ = _foundry_region(
        f"SOI340nm_{wavelength}nm_TE_STRIP_Grating_Coupler.gds", number
    )
    actual = getattr(cells, name)().get_region(layer).merged()
    assert actual.count() == reference.count()
    assert (actual ^ reference).is_empty()


@pytest.mark.parametrize(
    ("name", "filename", "radius", "slab_offset"),
    [
        ("bend_circular_sc", "SOI340nm_1550nm_TE_STRIP_90_Degree_Bend.gds", 10, 0),
        ("bend_circular_so", "SOI340nm_1310nm_TE_STRIP_90_Degree_Bend.gds", 10, 0),
        ("bend_circular_rc", "SOI340nm_1550nm_TE_RIB_90_Degree_Bend.gds", 100, 5),
    ],
)
def test_bend_matches_foundry(name, filename, radius, slab_offset):
    """Default bends are the foundry circular arcs (R = 10 um strip, 100 um rib)."""
    reference, dbu = _foundry_region(filename, 3)
    component = getattr(cells, name)()
    assert component.info["radius"] == radius
    # foundry bends run north from their origin and turn east: mirror about y = x
    offset = int((slab_offset + component.ports["o1"].width / 2) / dbu)
    actual = component.get_region("WG").merged().transformed(kdb.Trans.M45)
    actual.move(offset, 0)
    assert (actual ^ reference).sized(-10).is_empty()
    if slab_offset:
        slab, _ = _foundry_region(filename, 5)
        actual_slab = component.get_region("SLAB").merged().transformed(kdb.Trans.M45)
        actual_slab.move(offset, 0)
        assert (slab - actual_slab).is_empty()


def test_rib_radius():
    """Rib waveguides use the 100 um radius of the foundry rib bend."""
    assert tech.TECH.radius_rc == 100
    xs = gf.get_cross_section("xs_rc340")
    assert xs.radius == xs.radius_min == 100


@pytest.mark.parametrize(("layer", "number"), [("WG", 3), ("SLAB", 5)])
def test_rib_to_strip_matches_foundry(layer, number):
    """The transition matches the foundry cell (which is 200.013 um long)."""
    reference, _ = _foundry_region("SOI340nm_1550nm_TE_RIB_to_STRIP.gds", number)
    component = cells.taper_rib_to_strip()
    xor = component.get_region(layer).merged() ^ reference
    assert xor.sized(-10).is_empty()
    o1, o2 = component.ports["o1"], component.ports["o2"]
    assert (o1.width, o2.width) == (0.8, 0.45)
    assert np.allclose(o2.center, (200, 0))


def test_rib_to_strip_sample_connects():
    """A strip route reaches a rib waveguide through the transition."""
    from cspdk.si340.samples.sample_routing import sample_routing_rib_to_strip

    gf.clear_cache()
    c = sample_routing_rib_to_strip()
    names = {inst.cell.name for inst in c.insts}
    assert any(n.startswith("taper_rib_to_strip") for n in names), names
    assert {p.name for p in c.ports} == {"o1", "o2"}


def test_default_bends_are_circular():
    """MZIs use the foundry circular bends."""
    for band in ("sc", "so", "rc"):
        netlist = getattr(cells, f"mzi_{band}")(delta_length=50).get_netlist()
        bends = [
            v["component"]
            for v in netlist["instances"].values()
            if v["component"].startswith("bend")
        ]
        assert bends and set(bends) == {"bend_circular"}


def test_route_bundle_separation():
    """Bundles keep the 5 um waveguide spacing of the design guidelines."""
    for name in (
        "route_bundle",
        "route_bundle_sc",
        "route_bundle_so",
        "route_bundle_rc",
    ):
        strategy = PDK.routing_strategies[name]
        func = getattr(strategy, "func", strategy)
        assert inspect.signature(func).parameters["separation"].default == 5.0


def test_layer_stack_heater_on_cladding():
    """Heaters sit on the 1 um top cladding and pads on the heaters."""
    levels = LAYER_STACK.layers
    assert levels["grating"].thickness == pytest.approx(0.2)
    assert levels["heater"].zmin == pytest.approx(0.34 + 1.0)
    assert levels["metal"].zmin == pytest.approx(
        levels["heater"].zmin + levels["heater"].thickness
    )


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
    """Band aliases are discoverable for every cross-section."""
    for prefix in (
        "straight",
        "taper",
        "bend_euler",
        "bend_circular",
        "mmi1x2",
        "mmi2x2",
        "coupler",
        "grating_coupler_rectangular",
    ):
        for band in ("sc", "so", "rc"):
            name = f"{prefix}_{band}"
            assert PDK.models[name] is getattr(models, name)
    assert "taper_rib_to_strip" in PDK.models


@pytest.mark.parametrize(
    "band,wl0,neff", [("sc", 1.55, 2.661), ("so", 1.31, 2.821), ("rc", 1.55, 3.022)]
)
def test_waveguide_phase_and_loss(band, wl0, neff):
    """Straights, tapers and bends use the mode-solved index and loss in dB/cm."""
    length, loss = 123.0, 20.0
    expected = 10 ** (-loss * length * 1e-4 / 20) * np.exp(
        2j * np.pi * neff * length / wl0
    )
    for prefix in ("straight", "taper"):
        result = getattr(models, f"{prefix}_{band}")(wl=wl0, length=length, loss=loss)
        np.testing.assert_allclose(result["o1", "o2"], expected, atol=1e-6)
    radius = models.WAVEGUIDES[f"xs_{band}340"]["radius"]
    bend = getattr(models, f"bend_circular_{band}")(wl=wl0)
    np.testing.assert_allclose(
        bend["o1", "o2"],
        np.exp(2j * np.pi * neff * radius * np.pi / 2 / wl0),
        atol=1e-6,
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
    """C-band strip grating: 6 dB at 1.56 um with a 35 nm 1 dB bandwidth."""
    wl = np.array([1.56 - 0.0175, 1.56, 1.56 + 0.0175])
    power = np.abs(models.grating_coupler_rectangular(wl=wl)["o1", "o2"]) ** 2
    np.testing.assert_allclose(10 * np.log10(power), [-7, -6, -7], atol=1e-6)
    so = models.grating_coupler_rectangular(wl=1.31, cross_section="xs_so340")
    assert 10 * np.log10(np.abs(so["o1", "o2"]) ** 2) == pytest.approx(-6)


@pytest.mark.parametrize(
    "band,wl0,ng", [("sc", 1.55, 4.281), ("so", 1.31, 4.283), ("rc", 1.55, 3.821)]
)
def test_mzi_simulates_and_interferes(band, wl0, ng):
    """Real layout netlists produce finite, passive, wavelength-dependent outputs."""
    wl = jnp.linspace(wl0 - 0.03, wl0 + 0.03, 241)
    component = getattr(cells, f"mzi_{band}")(delta_length=100)
    circuit, _ = sax.circuit(component.get_netlist(), models=PDK.models)
    result = jax.block_until_ready(jax.jit(circuit)(wl=wl))
    assert {p for pair in result for p in pair} == {p.name for p in component.ports}
    cross = np.abs(np.asarray(result["o1", "o3"])) ** 2
    power = cross + np.abs(np.asarray(result["o1", "o2"])) ** 2
    assert np.all((power > 0.5) & (power <= 1.0))
    assert np.ptp(cross) > 0.3
    minima = np.flatnonzero((cross[1:-1] < cross[:-2]) & (cross[1:-1] < cross[2:]))
    fsr = np.mean(np.diff(np.asarray(wl)[minima + 1]))
    assert fsr == pytest.approx(wl0**2 / (ng * 100), rel=0.05)


def test_rib_to_strip_circuit():
    """The rib-to-strip sample simulates end to end."""
    from cspdk.si340.samples.sample_routing import sample_routing_rib_to_strip

    gf.clear_cache()
    component = sample_routing_rib_to_strip()
    circuit, _ = sax.circuit(component.get_netlist(), models=PDK.models)
    result = circuit(wl=jnp.linspace(1.5, 1.6, 11))
    transmission = np.abs(np.asarray(result["o1", "o2"])) ** 2
    assert np.all(np.isfinite(transmission))
    assert np.all((transmission > 0.3) & (transmission <= 0.5))


def test_model_ports_in_fresh_process():
    """Import Si340 after upstream models have cached the in/out port naming."""
    script = """
import numpy as np
import sax
import sax.models as sm
sax.set_port_naming_strategy('inout')
for model in (sm.straight, sm.mmi1x2, sm.mmi2x2, sm.grating_coupler):
    model(wl=np.array([1.55]))
from cspdk.si340 import PDK, cells
PDK.activate()
component = cells.mzi_sc(delta_length=100)
circuit, _ = sax.circuit(component.get_netlist(), models=PDK.models)
result = circuit(wl=np.linspace(1.52, 1.58, 121))
assert {p for pair in result for p in pair} == {'o1', 'o2', 'o3'}
assert np.ptp(np.abs(result['o1', 'o3']) ** 2) > 0.3
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=120)
