"""Foundry geometry and circuit regressions for the SiN300 repair."""

from __future__ import annotations

import subprocess
import sys

import gdsfactory as gf
import jax
import jax.numpy as jnp
import klayout.db as kdb
import numpy as np
import pytest

from cspdk.sin300 import PDK, cells, models
from cspdk.sin300.config import PATH


@pytest.fixture(autouse=True)
def activate_pdk():
    """Use SiN300 for every layout built in this module."""
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
        ("mmi1x2_nc", 1550, "2x1"),
        ("mmi1x2_no", 1310, "2x1"),
        ("mmi2x2", 1550, "2x2"),
        ("mmi2x2_nc", 1550, "2x2"),
        ("mmi2x2_no", 1310, "2x2"),
    ],
)
def test_mmi_matches_foundry(name, wavelength, kind):
    """Compare the complete nitride polygon, including tapers and port gaps."""
    reference, dbu = _foundry_region(
        f"SiN300nm_{wavelength}nm_TE_STRIP_{kind}_MMI.gds", 203
    )
    component = getattr(cells, name)()
    assert component.kcl.dbu == dbu
    actual = component.get_region("NITRIDE").merged()
    for region in (actual, reference):
        region.move(-region.bbox().left, -region.bbox().bottom)
    assert (actual ^ reference).is_empty()


@pytest.mark.parametrize(
    ("name", "wavelength"),
    [
        ("grating_coupler_rectangular", 1550),
        ("grating_coupler_rectangular_nc", 1550),
        ("grating_coupler_rectangular_no", 1310),
    ],
)
def test_grating_teeth_match_foundry(name, wavelength):
    """Match tooth count, pitch, and width to the original 11 um foundry grating."""
    reference, dbu = _foundry_region(
        f"SiN300nm_{wavelength}nm_TE_STRIP_Grating_Coupler.gds", 204
    )
    component = getattr(cells, name)()
    actual = component.get_region("NITRIDE_ETCH").merged()
    assert component.kcl.dbu == dbu
    boxes = [
        sorted((p.bbox() for p in region.each()), key=lambda b: b.left)
        for region in (actual, reference)
    ]
    assert len(boxes[0]) == len(boxes[1])
    np.testing.assert_array_equal(
        np.diff([b.left for b in boxes[0]]),
        np.diff([b.left for b in boxes[1]]),
    )
    np.testing.assert_array_equal(
        [b.width() for b in boxes[0]], [b.width() for b in boxes[1]]
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


@pytest.mark.parametrize("strategy", ["inout", "optical"])  # codespell:ignore inout
def test_model_ports_in_fresh_process(strategy):
    """Import SiN300 after upstream models have cached either port convention."""
    script = f"""
import numpy as np
import sax
import sax.models as sm
sax.set_port_naming_strategy({strategy!r})
for model in (sm.straight, sm.mmi1x2, sm.mmi2x2, sm.grating_coupler):
    model(wl=np.array([1.55]))
from cspdk.sin300 import PDK
for name, model in PDK.models.items():
    ports = {{p for pair in model(wl=np.array([1.55])) for p in pair}}
    assert all(p.startswith('e' if name == 'wire_corner' else 'o') for p in ports), (name, ports)
assert sax.get_port_naming_strategy() == {strategy!r}
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=60)


def test_model_registry():
    """Keep band aliases discoverable without exposing unimplemented stubs."""
    for prefix in ("straight", "taper", "bend_euler", "mmi1x2", "mmi2x2", "coupler"):
        for band in ("nc", "no"):
            name = f"{prefix}_{band}"
            assert PDK.models[name] is getattr(models, name)
    assert not {"heater", "coupler_straight", "coupler_symmetric"} & PDK.models.keys()


@pytest.mark.parametrize("band,wl0,neff", [("nc", 1.55, 1.60), ("no", 1.31, 1.63)])
def test_waveguide_phase_and_loss(band, wl0, neff):
    """Preserve the loss API and use each band's index, including its taper."""
    length, loss = 123.0, 20.0
    expected = 10 ** (-loss * length * 1e-4 / 20) * np.exp(
        2j * np.pi * neff * length / wl0
    )
    for prefix in ("straight", "taper"):
        result = getattr(models, f"{prefix}_{band}")(wl=wl0, length=length, loss=loss)
        np.testing.assert_allclose(result["o1", "o2"], expected, atol=1e-10)


@pytest.mark.parametrize("shape", ["rectangular", "elliptical"])
@pytest.mark.parametrize("band,wl0", [("nc", 1.55), ("no", 1.31)])
def test_grating_peak(shape, band, wl0):
    """Both band aliases and cross-section dispatch peak at their design wavelength."""
    wl = np.array([wl0 - 0.0175, wl0, wl0 + 0.0175])
    model = getattr(models, f"grating_coupler_{shape}_{band}")
    power = np.abs(model(wl=wl)["o1", "o2"]) ** 2
    np.testing.assert_allclose(power, 10 ** (-6 / 10) * np.array([0.5, 1, 0.5]))
    dispatched = getattr(models, f"grating_coupler_{shape}")(
        wl=wl, cross_section=f"xs_{band}"
    )
    np.testing.assert_allclose(np.abs(dispatched["o1", "o2"]) ** 2, power)


@pytest.mark.parametrize("band,wl0", [("nc", 1.55), ("no", 1.31)])
def test_mzi_simulates_and_interferes(band, wl0):
    """Real layout netlists produce finite, passive, wavelength-dependent outputs."""
    import sax

    wl = jnp.linspace(wl0 - 0.03, wl0 + 0.03, 121)
    component = getattr(cells, f"mzi_{band}")(delta_length=100)
    circuit, _ = sax.circuit(component.get_netlist(), models=PDK.models)
    result = jax.block_until_ready(circuit(wl=wl))
    assert {p for pair in result for p in pair} == {p.name for p in component.ports}
    cross = np.abs(np.asarray(result["o1", "o3"])) ** 2
    power = cross + np.abs(np.asarray(result["o1", "o2"])) ** 2
    assert np.all((power > 0.5) & (power <= 1.0))
    assert np.ptp(cross) > 0.3


@pytest.mark.parametrize(
    "cross_section,layer,width",
    [("metal_routing", "PAD", 10.0), ("heater_metal", "HEATER", 4.0)],
)
def test_electrical_straight(cross_section, layer, width):
    """Metal wires have two unique electrical ports and no extra heater strip."""
    component = cells.straight(length=20, cross_section=cross_section)
    assert {p.name for p in component.ports} == {"e1", "e2"}
    assert len(component.ports) == 2
    for port in component.ports:
        assert port.port_type == "electrical"
        assert port.layer == gf.get_layer(layer)
        assert port.width == width
    assert set(component.layers) == {gf.get_layer_tuple(layer)}
    region = component.get_region(layer).merged()
    assert region.count() == 1
    assert region.area() * component.kcl.dbu**2 == pytest.approx(20 * width)


@pytest.mark.parametrize(
    "cross_section,layer", [("metal_routing", "PAD"), ("heater_metal", "HEATER")]
)
@pytest.mark.parametrize("bundle", [False, True])
def test_electrical_routing(cross_section, layer, bundle):
    """Single and bundled routes connect endpoints on only the intended metal."""
    component = gf.Component()
    starts, ends = [], []
    for index in range(2 if bundle else 1):
        start = component << cells.straight(length=20, cross_section=cross_section)
        end = component << cells.straight(length=20, cross_section=cross_section)
        start.dmove((0, 50 * index))
        end.dmove((200, 100 + 50 * index))
        starts.append(start.ports["e2"])
        ends.append(end.ports["e1"])
    if bundle:
        routes = PDK.routing_strategies["route_bundle_metal"](
            component,
            starts,
            ends,
            cross_section=cross_section,
            on_collision="error",
            raise_on_error=True,
        )
        assert len(routes) == 2
    else:
        PDK.routing_strategies["route_single_metal"](
            component, starts[0], ends[0], cross_section=cross_section
        )
    assert set(component.layers) == {gf.get_layer_tuple(layer)}
    connected = list(component.get_region(layer).merged().each())
    assert len(connected) == len(starts)
    for start, end in zip(starts, ends, strict=True):
        assert any(
            polygon.inside(kdb.DPoint(*start.center).to_itype(component.kcl.dbu))
            and polygon.inside(kdb.DPoint(*end.center).to_itype(component.kcl.dbu))
            for polygon in connected
        )
