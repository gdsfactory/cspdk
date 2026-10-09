"""Foundry geometry, layer stack, routing and circuit regressions for SiN200."""

from __future__ import annotations

import subprocess
import sys

import gdsfactory as gf
import jax
import jax.numpy as jnp
import klayout.db as kdb
import numpy as np
import pytest

from cspdk.sin200 import LAYER_STACK, PDK, cells, models, tech
from cspdk.sin200.config import PATH

BANDS = [("n780", 780, 0.78), ("n638", 638, 0.638), ("n520", 520, 0.52)]


@pytest.fixture(autouse=True)
def activate_pdk():
    """Use SiN200 for every layout built in this module."""
    PDK.activate()


def _foundry_region(filename, layer):
    layout = kdb.Layout()
    layout.read(str(PATH.gds / filename))
    region = kdb.Region(
        layout.top_cell().begin_shapes_rec(layout.find_layer(layer, 0))
    ).merged()
    return region, layout.dbu


################
# Foundry geometry
################


@pytest.mark.parametrize(
    ("name", "wavelength"),
    [
        ("grating_coupler_rectangular", 780),
        ("grating_coupler_rectangular_n780", 780),
        ("grating_coupler_rectangular_n638", 638),
        ("grating_coupler_rectangular_n520", 520),
    ],
)
@pytest.mark.parametrize(
    ("layer", "layer_name"), [(203, "NITRIDE"), (204, "NITRIDE_ETCH")]
)
def test_grating_matches_foundry(name, wavelength, layer, layer_name):
    """Slab, taper and trenches are identical to the foundry GDS, in place."""
    reference, dbu = _foundry_region(
        f"SiN200nm_{wavelength}nm_TE_STRIP_Grating_Coupler.gds", layer
    )
    component = getattr(cells, name)()
    assert component.kcl.dbu == dbu
    actual = component.get_region(layer_name).merged()
    assert (actual ^ reference).is_empty()


@pytest.mark.parametrize(
    ("band", "o2_x", "width", "fiber_angle"),
    [
        ("n780", 213.447, 0.5, 22.0),
        ("n638", 160.965, 0.36, 18.0),
        ("n520", 159.101, 0.27, 22.0),
    ],
)
def test_grating_ports(band, o2_x, width, fiber_angle):
    """o1 is the waveguide end at x=0; o2 is the fibre port at the teeth centre."""
    component = getattr(cells, f"grating_coupler_rectangular_{band}")()
    o1, o2 = component.ports["o1"], component.ports["o2"]
    assert o1.center == (0, 0)
    assert o1.orientation == 180
    assert o1.width == pytest.approx(width)
    assert o2.center == pytest.approx((o2_x, 0))
    assert o2.port_type == "vertical_te"
    assert component.info["fiber_angle"] == fiber_angle


@pytest.mark.parametrize(("name", "kind"), [("mmi1x2", "2x1"), ("mmi2x2", "2x2")])
@pytest.mark.parametrize(("band", "wavelength", "_wl"), BANDS)
def test_mmi_matches_foundry(name, kind, band, wavelength, _wl):
    """Compare the complete nitride polygon, including tapers and port gaps."""
    reference, dbu = _foundry_region(
        f"SiN200nm_{wavelength}nm_TE_STRIP_{kind}_MMI.gds", 203
    )
    component = getattr(cells, f"{name}_{band}")()
    assert component.kcl.dbu == dbu
    actual = component.get_region("NITRIDE").merged()
    for region in (actual, reference):
        region.move(-region.bbox().left, -region.bbox().bottom)
    assert (actual ^ reference).is_empty()


def test_heater_matches_foundry():
    """The heater cell is the foundry Heater.gds with a pad port on each pad."""
    component = cells.heater()
    filament, _ = _foundry_region("Heater.gds", 39)
    pad_region, _ = _foundry_region("Heater.gds", 41)
    assert (component.get_region("HEATER").merged() ^ filament).is_empty()
    assert (component.get_region("PAD").merged() ^ pad_region).is_empty()
    pads = list(pad_region.each())
    for port in component.ports:
        assert port.port_type == "electrical"
        point = kdb.DPoint(port.center[0], port.center[1] - 1).to_itype(0.001)
        assert any(p.inside(point) for p in pads)


def test_heater_cross_section_matches_foundry_filament():
    """The heater metal cross-section uses the 2 um foundry filament width."""
    assert gf.get_cross_section("heater_metal_sin200").width == 2.0


@pytest.mark.parametrize(("band", "_wavelength", "_wl"), BANDS)
def test_elliptical_period_matches_foundry(band, _wavelength, _wl):
    """Elliptical gratings use the foundry rectangular pitch."""
    period = getattr(cells, f"grating_coupler_rectangular_{band}").keywords.get(
        "period", 0.668
    )
    component = getattr(cells, f"grating_coupler_elliptical_{band}")()
    assert component.info["period"] == pytest.approx(period)


def test_die_size():
    """The die floorplan is the 11.47 x 15.45 mm2 user cell."""
    component = cells.die()
    bbox = component.get_region("FLOORPLAN").bbox().to_dtype(component.kcl.dbu)
    assert (bbox.width(), bbox.height()) == (11470, 15450)


@pytest.mark.parametrize("xs", ["xs_n780", "xs_n638", "xs_n520"])
def test_taper_width_follows_cross_section(xs):
    """Without width1, a taper starts at the cross-section width."""
    width = gf.get_cross_section(xs).width
    for component in (
        cells.taper(cross_section=xs, width2=1.0),
        getattr(cells, f"taper_{xs[3:]}")(width2=1.0),
    ):
        assert component.ports["o1"].width == width
        assert component.ports["o2"].width == 1.0
    assert cells.taper(cross_section=xs).ports["o2"].width == width


################
# Layer stack
################


def test_layer_stack_nitride_minus_etch():
    """GDS 204 is a dark-field etch: it removes nitride instead of adding it."""
    assert set(LAYER_STACK.layers) == {"nitride", "heater", "metal"}
    component = cells.grating_coupler_rectangular_n780()
    nitride = LAYER_STACK.layers["nitride"].layer.get_shapes(component).merged()
    expected = component.get_region("NITRIDE") - component.get_region("NITRIDE_ETCH")
    assert (nitride ^ expected).is_empty()
    assert nitride.area() < component.get_region("NITRIDE").area()


def test_layer_stack_heights():
    """Heaters and pads sit on the 2 um cladding over the 200 nm nitride."""
    levels = LAYER_STACK.layers
    assert levels["nitride"].zmin == 0
    assert levels["nitride"].thickness == pytest.approx(0.2)
    assert levels["heater"].zmin == pytest.approx(2.2)
    assert levels["metal"].zmin == levels["heater"].zmin


################
# Routing
################


@pytest.mark.parametrize(
    "cross_section,layer,width",
    [("metal_routing", "PAD", 10.0), ("heater_metal_sin200", "HEATER", 2.0)],
)
def test_electrical_straight(cross_section, layer, width):
    """Metal wires have two electrical ports on the intended layer."""
    component = cells.straight(length=20, cross_section=cross_section)
    assert {p.name for p in component.ports} == {"e1", "e2"}
    for port in component.ports:
        assert port.port_type == "electrical"
        assert port.layer == gf.get_layer(layer)
        assert port.width == width
    assert set(component.layers) == {gf.get_layer_tuple(layer)}


@pytest.mark.parametrize(
    "cross_section,layer", [("metal_routing", "PAD"), ("heater_metal_sin200", "HEATER")]
)
@pytest.mark.parametrize("bundle", [False, True])
@pytest.mark.parametrize("dy", [0, 100])
def test_electrical_routing(cross_section, layer, bundle, dy):
    """Single and bundled metal routes connect their endpoints, including straight runs."""
    component = gf.Component()
    starts, ends = [], []
    for index in range(2 if bundle else 1):
        start = component << cells.straight(length=20, cross_section=cross_section)
        end = component << cells.straight(length=20, cross_section=cross_section)
        start.dmove((0, 50 * index))
        end.dmove((200, dy + 50 * index))
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

    def point(port):
        return kdb.DPoint(*port.center).to_itype(component.kcl.dbu)

    for start, end in zip(starts, ends, strict=True):
        assert any(
            polygon.inside(point(start)) and polygon.inside(point(end))
            for polygon in connected
        )


@pytest.mark.parametrize(("band", "_wavelength", "_wl"), BANDS)
def test_route_bundle_sbend_fallback(band, _wavelength, _wl):
    """A small lateral offset is routed with a bend_s above the band radius."""
    component = gf.Component()
    xs = f"xs_{band}"
    start = component << cells.straight(length=10, cross_section=xs)
    end = component << cells.straight(length=10, cross_section=xs)
    end.dmove((500, 5))
    PDK.routing_strategies[f"route_bundle_{band}"](
        component, [start.ports["o2"]], [end.ports["o1"]], on_collision="error"
    )
    sbends = [i for i in component.insts if i.cell.name.startswith("bend_s")]
    assert len(sbends) == 1
    assert sbends[0].cell.info["min_bend_radius"] > gf.get_cross_section(xs).radius


def test_sample_routing_different_widths():
    """Routing between different widths inserts one taper at each end."""
    gf.clear_cache()
    from cspdk.sin200.samples.sample_routing import sample_routing_different_widths

    component = sample_routing_different_widths()
    tapers = [i for i in component.insts if i.cell.name.startswith("taper")]
    assert len(tapers) == 2


################
# Models
################


@pytest.mark.parametrize("name", sorted(PDK.models))
@pytest.mark.parametrize("wl", [0.638, np.linspace(0.5, 0.8, 7)])
def test_registered_models_evaluate(name, wl):
    """Every advertised model supports scalar and swept wavelengths under JIT."""
    result = jax.jit(PDK.models[name])(wl=wl)
    for ports, value in result.items():
        assert isinstance(ports, tuple) and len(ports) == 2
        assert all(p.startswith("e" if name == "wire_corner" else "o") for p in ports)
        assert np.shape(value) == np.shape(wl)
        assert np.isfinite(value).all()


def test_model_registry():
    """Band aliases are registered and there are no unimplemented stubs."""
    for prefix in (
        "straight",
        "taper",
        "bend_euler",
        "mmi1x2",
        "mmi2x2",
        "coupler",
        "grating_coupler_rectangular",
        "grating_coupler_elliptical",
    ):
        assert PDK.models[prefix] is getattr(models, prefix)
        for band, _, _ in BANDS:
            name = f"{prefix}_{band}"
            assert PDK.models[name] is getattr(models, name)
    with pytest.raises(ValueError, match="xs_unknown"):
        models.straight(cross_section="xs_unknown")


@pytest.mark.parametrize(
    ("band", "wl0", "neff"),
    [("n780", 0.78, 1.6609), ("n638", 0.638, 1.6951), ("n520", 0.52, 1.7381)],
)
def test_waveguide_phase_and_loss(band, wl0, neff):
    """Straight and taper use the band index and the 5 dB/cm default loss."""
    length = 123.0
    for loss, kwargs in ((5.0, {}), (20.0, {"loss": 20.0})):
        expected = 10 ** (-loss * length * 1e-4 / 20) * np.exp(
            2j * np.pi * neff * length / wl0
        )
        for prefix in ("straight", "taper"):
            alias = getattr(models, f"{prefix}_{band}")(wl=wl0, length=length, **kwargs)
            dispatched = getattr(models, prefix)(
                wl=wl0, length=length, cross_section=f"xs_{band}", **kwargs
            )
            for result in (alias, dispatched):
                np.testing.assert_allclose(result["o1", "o2"], expected, atol=1e-10)


@pytest.mark.parametrize("shape", ["rectangular", "elliptical"])
@pytest.mark.parametrize(
    ("band", "wl0", "loss", "bandwidth"),
    [
        ("n780", 0.78, 9.0, 0.064),
        ("n638", 0.638, 15.06, 0.043),
        ("n520", 0.52, 13.25, 0.028),
    ],
)
def test_grating_peak(shape, band, wl0, loss, bandwidth):
    """Grating couplers peak at the band with the foundry loss and 3 dB bandwidth."""
    wl = np.array([wl0 - bandwidth / 2, wl0, wl0 + bandwidth / 2])
    power = (
        np.abs(getattr(models, f"grating_coupler_{shape}_{band}")(wl=wl)["o1", "o2"])
        ** 2
    )
    np.testing.assert_allclose(power, 10 ** (-loss / 10) * np.array([0.5, 1, 0.5]))
    dispatched = getattr(models, f"grating_coupler_{shape}")(
        wl=wl, cross_section=f"xs_{band}"
    )
    np.testing.assert_allclose(np.abs(dispatched["o1", "o2"]) ** 2, power)


@pytest.mark.parametrize(("band", "_wavelength", "wl0"), BANDS)
def test_mzi_simulates_and_interferes(band, _wavelength, wl0):
    """Layout netlists produce finite, passive, wavelength-dependent outputs."""
    import sax

    wl = jnp.linspace(wl0 - 0.01, wl0 + 0.01, 201)
    component = getattr(cells, f"mzi_{band}")(delta_length=100)
    circuit, _ = sax.circuit(component.get_netlist(), models=PDK.models)
    result = jax.block_until_ready(jax.jit(circuit)(wl=wl))
    assert {p for pair in result for p in pair} == {p.name for p in component.ports}
    cross = np.abs(np.asarray(result["o1", "o3"])) ** 2
    power = cross + np.abs(np.asarray(result["o1", "o2"])) ** 2
    assert np.all((power > 0.5) & (power <= 1.0))
    assert np.ptp(cross) > 0.3


@pytest.mark.parametrize("strategy", ["inout", "optical"])  # codespell:ignore inout
def test_model_ports_in_fresh_process(strategy):
    """Import SiN200 after upstream models have cached either port convention."""
    script = f"""
import numpy as np
import sax
import sax.models as sm
sax.set_port_naming_strategy({strategy!r})
for model in (sm.straight, sm.mmi1x2, sm.mmi2x2, sm.grating_coupler):
    model(wl=np.array([0.638]))
from cspdk.sin200 import PDK
for name, model in PDK.models.items():
    ports = {{p for pair in model(wl=np.array([0.638])) for p in pair}}
    if name == 'wire_corner':
        expected = {{'e1', 'e2'}}
    elif name.startswith(('mmi2x2', 'coupler')):
        expected = {{'o1', 'o2', 'o3', 'o4'}}
    elif name.startswith('mmi1x2'):
        expected = {{'o1', 'o2', 'o3'}}
    else:
        expected = {{'o1', 'o2'}}
    assert ports == expected, (name, ports)
assert sax.get_port_naming_strategy() == {strategy!r}
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=120)


def test_tech_bands_match_models():
    """Model bend radii agree with the cross-section radii."""
    for band, _, _ in BANDS:
        xs = f"xs_{band}"
        assert models.BANDS[xs].radius == getattr(tech.Tech, f"radius_{band}")
