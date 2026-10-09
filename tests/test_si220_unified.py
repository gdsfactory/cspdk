"""Contract tests for the unified Cornerstone Si220 PDK."""

from pathlib import Path

import gdsfactory as gf
import numpy as np
import pytest

import cspdk.si220 as si220

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_unified_pdk_is_exported() -> None:
    """The si220 package exposes the single public PDK entry point."""
    assert si220.PDK.name == "cspdk.si220"
    assert si220.PATH.repo == REPO_ROOT


@pytest.mark.parametrize(
    ("name", "width"),
    [
        ("strip_cband", 0.45),
        ("rib_cband", 0.45),
        ("strip_oband", 0.40),
        ("rib_oband", 0.40),
    ],
)
def test_band_cross_section_width(name: str, width: float) -> None:
    """Each named band cross-section resolves to its documented width."""
    si220.PDK.activate()
    assert gf.get_cross_section(name).width == width


def test_band_detection_accepts_registered_names_factories_and_objects() -> None:
    """Band-sensitive defaults work with every supported cross-section spec form."""
    assert si220.tech.get_band("strip_oband") == "oband"
    assert si220.tech.get_band(si220.tech.strip_oband) == "oband"
    assert si220.tech.get_band(si220.tech.strip_oband()) == "oband"
    assert si220.tech.get_band(si220.tech.strip_cband()) == "cband"
    assert si220.tech.is_rib(si220.tech.rib_oband)
    assert si220.tech.is_rib(si220.tech.rib_cband())


def test_band_variants_coexist_without_cell_cache_collision() -> None:
    """Shared factories create distinct geometry for each band in one process."""
    si220.PDK.activate()
    cband = si220.cells.straight(cross_section="strip_cband")
    oband = si220.cells.straight(cross_section="strip_oband")
    assert cband.name != oband.name
    assert cband.ports["o1"].width == 0.45
    assert oband.ports["o1"].width == 0.40


@pytest.mark.parametrize(
    ("cross_section", "coupler_right", "mmi1x2_right", "mmi2x2_right"),
    [
        ("strip_cband", 24.5, 51.0, 62.5),
        ("strip_oband", 30.0, 60.0, 73.5),
    ],
)
def test_band_sensitive_cell_defaults(
    cross_section: str,
    coupler_right: float,
    mmi1x2_right: float,
    mmi2x2_right: float,
) -> None:
    """Shared cells reproduce the established geometry for each band."""
    si220.PDK.activate()
    assert si220.cells.coupler(
        cross_section=cross_section
    ).dbbox().right == pytest.approx(coupler_right)
    assert si220.cells.mmi1x2(
        cross_section=cross_section
    ).dbbox().right == pytest.approx(mmi1x2_right)
    assert si220.cells.mmi2x2(
        cross_section=cross_section
    ).dbbox().right == pytest.approx(mmi2x2_right)


def test_straight_model_dispatches_by_band() -> None:
    """One model name dispatches to the selected band's optical response."""
    model = si220.PDK.models["straight"]
    cband = model(wl=1.55, length=10.0, cross_section="strip_cband")
    oband = model(wl=1.31, length=10.0, cross_section="strip_oband")
    expected_ports = {("o1", "o2"), ("o2", "o1")}
    assert set(cband) == expected_ports
    assert set(oband) == expected_ports
    assert not np.allclose(cband["o1", "o2"], oband["o1", "o2"])


@pytest.mark.parametrize(
    "cross_section",
    [si220.tech.strip_oband, si220.tech.strip_oband()],
)
def test_model_dispatch_accepts_oband_factories_and_objects(cross_section) -> None:
    """Resolved cross-sections select the same O-band model as its name."""
    model = si220.PDK.models["straight"]
    expected = model(wl=1.31, length=10.0, cross_section="strip_oband")
    actual = model(wl=1.31, length=10.0, cross_section=cross_section)
    assert np.allclose(actual["o1", "o2"], expected["o1", "o2"])


def test_model_module_attributes_bind_dispatch_wrappers() -> None:
    """Module attributes resolve to band-dispatching wrappers, not raw models.

    Cell SAX metadata references models as ``module="cspdk.si220.models"`` plus
    ``qualname=<name>``, so consumers that resolve by module path must receive
    the dispatch wrapper.  The raw implementations imported from
    ``.waveguides``/``.couplers`` only accept the ``"strip"``/``"rib"``
    vocabulary and would reject the PDK-level cross-section names.
    """
    import cspdk.si220.models as models

    assert models.straight is si220.PDK.models["straight"]
    assert models.bend_euler is si220.PDK.models["bend_euler"]

    for cross_section in ("strip_cband", "strip_oband"):
        response = models.bend_euler(wl=1.55, length=10.0, cross_section=cross_section)
        assert ("o1", "o2") in response


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (si220.cells.mzi, {}),
        (si220.cells.mzi_lattice, {}),
        (si220.cells.straight_heater_metal, {}),
        (
            si220.cells.die_with_pads,
            {"ngratings": 2, "npads": 1, "with_loopback": False},
        ),
    ],
)
def test_composite_cells_propagate_oband_cross_section(factory, kwargs) -> None:
    """O-band selection reaches every nested optical component."""
    si220.PDK.activate()
    component = factory(cross_section="strip_oband", **kwargs)
    optical_ports = list(component.ports.filter(port_type="optical"))
    assert optical_ports
    assert all(port.width == pytest.approx(0.40) for port in optical_ports)


def test_fiber_container_propagates_oband_cross_section() -> None:
    """The routed device and grating inside a container use O-band ports."""
    si220.PDK.activate()
    component = si220.cells.add_fiber_single(
        cross_section="strip_oband", with_loopback=False
    )
    optical_ports = [
        port
        for instance in component.insts
        for port in instance.ports
        if port.port_type == "optical"
    ]
    assert optical_ports
    assert all(port.width == pytest.approx(0.40) for port in optical_ports)


@pytest.mark.parametrize(
    ("factory", "cross_section"),
    [
        (si220.cells.crossing, "strip_oband"),
        (si220.cells.crossing_rib, "rib_oband"),
    ],
)
def test_fixed_crossings_select_oband_geometry(factory, cross_section: str) -> None:
    """Fixed crossings load the 1310 nm geometry and expose O-band ports."""
    si220.PDK.activate()
    component = factory(cross_section=cross_section)
    assert all(port.width == pytest.approx(0.40) for port in component.ports)


def test_pdk_uses_dispatchable_sax_heater_model() -> None:
    """Optional active models do not replace the shared SAX model."""
    model = si220.PDK.models["straight_heater_metal"]
    result = model(wl=1.31, cross_section="strip_oband")
    assert ("o1", "o2") in result


def _simulate(component, wl):
    import jax
    import jax.numpy as jnp
    import sax

    circuit, _ = sax.circuit(component.get_netlist(), models=si220.PDK.models)
    # block before `circuit` is freed: klujax segfaults if it is released mid-solve
    return jax.block_until_ready(circuit(wl=jnp.asarray(wl)))


@pytest.mark.parametrize(
    ("cross_section", "wl"),
    [
        ("strip_cband", 1.55),
        ("rib_cband", 1.55),
        ("strip_oband", 1.31),
        ("rib_oband", 1.31),
    ],
)
def test_mzi_simulates(cross_section: str, wl: float) -> None:
    """The default coupler-based MZI simulates in every band and interferes."""
    si220.PDK.activate()
    wls = np.linspace(wl - 0.03, wl + 0.03, 121)
    s = _simulate(si220.cells.mzi(cross_section=cross_section), wls)
    cross = np.abs(np.asarray(s["o1", "o3"])) ** 2
    power = cross + np.abs(np.asarray(s["o1", "o4"])) ** 2
    assert np.all((power > 0.5) & (power <= 1.0 + 1e-6))
    # a coupler that does not split light gives a flat output: no fringes
    assert cross.max() - cross.min() > 0.3


@pytest.mark.parametrize(
    ("cross_section", "radius", "gap", "ng", "wl0"),
    [
        ("strip_cband", 10, 0.2, 4.30, 1.55),
        ("strip_cband", 25, 0.2, 4.30, 1.55),
        ("rib_cband", 25, 0.5, 3.87, 1.55),
        ("strip_oband", 25, 0.3, 4.34, 1.31),
        ("rib_oband", 25, 0.4, 3.98, 1.31),
    ],
)
def test_ring_fsr_matches_group_index(
    cross_section: str, radius: float, gap: float, ng: float, wl0: float
) -> None:
    """Simulated ring FSR matches lambda^2 / (ng * L) for the full ring length."""
    from scipy.signal import find_peaks

    si220.PDK.activate()
    ring = si220.cells.ring_single(radius=radius, cross_section=cross_section, gap=gap)
    length = 2 * np.pi * radius + 2 * 4.0 + 2 * 0.6
    wl = np.linspace(wl0 - 0.02, wl0 + 0.02, 4001)
    transmission = np.abs(np.asarray(_simulate(ring, wl)["o1", "o2"])) ** 2
    dips, _ = find_peaks(-transmission, prominence=0.01)
    fsr = np.median(np.diff(wl[dips]))
    assert fsr == pytest.approx(wl0**2 / (ng * length), rel=0.02)


def _two_in_series(cell, **settings) -> gf.Component:
    """Two copies of a two-port cell connected o2 -> o1, so SAX treats it as a circuit."""
    c = gf.Component()
    first = c << cell(**settings)
    second = c << cell(**settings)
    second.connect("o1", first.ports["o2"])
    c.add_port("o1", port=first.ports["o1"])
    c.add_port("o2", port=second.ports["o2"])
    return c


@pytest.mark.parametrize(
    "name", ["grating_coupler_rectangular", "grating_coupler_elliptical"]
)
@pytest.mark.parametrize("cross_section", ["strip_oband", "rib_oband"])
def test_oband_grating_couplers_are_centred_at_1310(
    name: str, cross_section: str
) -> None:
    """O-band grating models transmit at 1.31 um, like C-band ones at 1.55 um."""
    oband = si220.PDK.models[name](wl=np.array([1.31]), cross_section=cross_section)
    cband = si220.PDK.models[name](wl=np.array([1.55]), cross_section="strip_cband")
    assert abs(oband["o1", "o2"][0]) == pytest.approx(abs(cband["o1", "o2"][0]))


def test_dispatched_models_expose_cross_section() -> None:
    """SAX only forwards settings in the model signature, so every model needs it."""
    import inspect

    from cspdk.si220.models import get_models

    for name, model in get_models().items():
        assert si220.PDK.models[name] is model, name
        assert "cross_section" in inspect.signature(model).parameters, name


@pytest.mark.parametrize(
    ("cross_section", "wl"),
    [("strip_oband", 1.31), ("rib_oband", 1.31), ("strip_cband", 1.55)],
)
def test_straight_in_circuit_matches_direct_model(
    cross_section: str, wl: float
) -> None:
    """Inside a circuit a band's model gets its own defaults, not C-band ones."""
    si220.PDK.activate()
    circuit = _simulate(
        _two_in_series(si220.cells.straight, length=500, cross_section=cross_section),
        [wl],
    )
    direct = si220.PDK.models["straight"](
        wl=np.array([wl]), length=1000, cross_section=cross_section
    )
    assert np.allclose(circuit["o1", "o2"], direct["o1", "o2"])


def test_oband_heater_in_circuit_uses_oband_model() -> None:
    """The heater cell has no cross_section in its C-band model; O-band must still win."""
    from cspdk.si220.models import oband

    si220.PDK.activate()
    heater = si220.cells.straight_heater_metal
    length = heater(cross_section="strip_oband").info["length"]
    circuit = _simulate(_two_in_series(heater, cross_section="strip_oband"), [1.31])
    oband_heater = oband.get_models()["straight_heater_metal"]
    expected = oband_heater(wl=1.31, length=2 * length)["o1", "o2"]
    assert np.allclose(circuit["o1", "o2"][0], expected)


def test_gdsfactoryplus_finds_the_dispatching_models() -> None:
    """GDSFactory+ scans the package source for models; it must find only dispatchers.

    It registers every public ``def`` returning ``sax.SDict`` by name, so a public
    band implementation in a submodule would replace the dispatcher of the same name.
    """
    from pathlib import Path

    from gdsfactoryplus.core import find_models

    from cspdk.si220.models import get_models

    records = find_models({"cspdk.si220": Path(si220.__file__).parent})
    sources = {name: Path(record.source).as_posix() for name, record in records.items()}
    assert sources.keys() == get_models().keys()
    assert all(s.endswith("si220/models/__init__.py") for s in sources.values()), (
        sources
    )
