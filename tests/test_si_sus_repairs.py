"""Foundry geometry, layer-404 DRC and circuit regressions for the si_sus repair."""

from __future__ import annotations

import inspect

import gdsfactory as gf
import jax
import klayout.db as kdb
import numpy as np
import pytest
import sax

from cspdk.si_sus import PDK, cells, models
from cspdk.si_sus.config import PATH
from cspdk.si_sus.tech import LAYER, TECH

MIN_GAP = 180  # nm, MPW #7 Table 2, 'CORNERSTONE to bias', layer 404
MIN_FEATURE = 270  # nm


@pytest.fixture(autouse=True)
def activate_pdk():
    """Use si_sus for every layout built in this module."""
    PDK.activate()


def _foundry(name: str) -> kdb.Region:
    layout = kdb.Layout()
    layout.read(str(PATH.gds / f"Suspendedsilicon500nm_3800nm_TE_{name}.gds"))
    return _flat(layout.top_cell().begin_shapes_rec(layout.find_layer(404, 0)))


def _flat(shapes: kdb.RecursiveShapeIterator) -> kdb.Region:
    """Copy the shapes, so the region outlives its layout."""
    region = kdb.Region()
    region.insert(shapes)
    return region


def _etch(component: gf.Component) -> kdb.Region:
    return _flat(component.begin_shapes_rec(gf.get_layer(LAYER.WG)))


def _check_404(region: kdb.Region) -> None:
    """Separate slots, all gaps >= 180nm and all slots >= 270nm wide."""
    assert region.merged().count() == region.count()
    euclid = kdb.Metrics.Euclidian  # codespell:ignore
    gaps = [e.distance() for e in region.space_check(MIN_GAP, False, euclid, 30).each()]
    assert not gaps, f"layer-404 gap {min(gaps)}nm < {MIN_GAP}nm"
    widths = list(region.width_check(MIN_FEATURE, False, euclid, 30).each())
    assert not widths, f"layer-404 feature {widths[0].distance()}nm < {MIN_FEATURE}nm"


def _polar(region: kdb.Region, center: tuple[float, float]) -> np.ndarray:
    """Return (start angle, end angle, r_min, r_max) per slot, sorted by angle."""
    rows = []
    for polygon in region.each():
        pts = np.array([(p.x, p.y) for p in polygon.each_point_hull()]) / 1000
        d = pts - center
        theta = np.arctan2(d[:, 1], d[:, 0]) % (2 * np.pi)
        r = np.hypot(d[:, 0], d[:, 1])
        rows.append((theta.min(), theta.max(), r.min(), r.max()))
    return np.array(sorted(rows))


def test_straight_matches_foundry():
    """straight(500) has exactly the foundry waveguide's 909 slot pairs."""
    actual = _etch(cells.straight(length=500)).merged()
    reference = _foundry("Waveguide").merged()
    for region in (actual, reference):
        region.move(-region.bbox().left, -region.bbox().bottom)
    assert (actual ^ reference).is_empty()


@pytest.mark.parametrize("width", [1.5, 3.0])
def test_straight_windows_follow_width(width):
    """The 3.5um windows start at the core edges for any core width."""
    for c in (cells.straight(length=20, width=width), cells.taper(width1=width)):
        bbox = _etch(c).bbox()
        assert bbox.top / 1000 == pytest.approx(width / 2 + 3.5)
        assert bbox.bottom / 1000 == pytest.approx(-width / 2 - 3.5)
        inner = _etch(c) & kdb.Region(
            kdb.Box(-(10**6), -500 * width, 10**6, 500 * width)
        )
        assert inner.is_empty()


def test_bend_circular_matches_foundry_wedges():
    """Polar wedges of 0.0075 rad at 0.01375 rad pitch, like the foundry bend."""
    foundry = _polar(_foundry("90_DegreeBend"), (45.0, 0.189))
    bend = _polar(_etch(cells.bend_circular()), (0.0, TECH.radius_sus))
    for slots in (foundry, bend):
        inner = slots[slots[:, 2] < TECH.radius_sus]
        # 1nm vertex snapping is up to 1e-4 rad at the 36.5um inner radius
        np.testing.assert_allclose(inner[:, 1] - inner[:, 0], 0.0075, atol=1e-4)
        np.testing.assert_allclose(np.diff(inner[:, :2].mean(1)), 0.01375, atol=1e-4)
        np.testing.assert_allclose(inner[:, 2:] - [36.5, 40.0], 0, atol=3e-3)
    # the foundry's 115th wedge overhangs its port plane; ours stay inside
    assert len(foundry) == 230
    assert len(bend) == 228
    assert bend[:, 0].min() > 3 * np.pi / 2 and bend[:, 1].max() < 2 * np.pi


def test_bend_s_matches_foundry():
    """Same slot count, x-pitch and core-edge curve as the foundry S-bend."""
    reference = _foundry("SBend")
    reference.move(0, -4250)  # foundry o1 center line is at y=4.25
    actual = _etch(cells.bend_s())
    boxes = [sorted(p.bbox() for p in r.each()) for r in (actual, reference)]
    assert len(boxes[0]) == len(boxes[1]) == 144
    for b in boxes:
        lefts = sorted({box.left for box in b})
        np.testing.assert_allclose(np.diff(lefts), 550)
        assert {box.width() for box in b} == {300}
    # upper core edge = bottom of the upper slots, sampled at their x edges
    edges = []
    for region in (actual, reference):
        pts = []
        for polygon in region.each():
            if polygon.bbox().center().y > polygon.bbox().left / 5:
                hull = [(p.x, p.y) for p in polygon.each_point_hull()]
                for x in {x for x, _ in hull}:
                    pts.append((x, min(y for px, y in hull if px == x)))
        edges.append(np.array(sorted(pts)))
    ours, theirs = edges
    np.testing.assert_allclose(ours[:, 1], np.interp(ours[:, 0], *theirs.T), atol=15)


def test_grating_coupler_matches_foundry():
    """The grating coupler is the unmodified foundry GDS."""
    c = cells.grating_coupler_rectangular()
    assert (_etch(c).merged() ^ _foundry("GratingCoupler").merged()).is_empty()
    o1, o2 = c.ports["o1"], c.ports["o2"]
    assert o1.center == (-0.125, 0) and o1.orientation == 180 and o1.width == 1.5
    assert o2.port_type == "vertical_te"
    assert c.info["fiber_angle"] == 19


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("straight", {"length": 37.3}),
        ("bend_circular", {}),
        ("bend_circular", {"radius": TECH.radius_min_sus}),
        ("bend_circular", {"angle": -90}),
        ("bend_euler", {}),
        ("bend_s", {}),
        ("bend_s", {"size": (20.0, -1.8)}),
        ("taper", {"width1": 1.5, "width2": 6.0, "length": 20.0}),
        ("grating_coupler_rectangular", {}),
    ],
)
def test_cell_404_rules(name, kwargs):
    """Every cell meets the layer-404 gap and feature minima."""
    _check_404(_etch(getattr(cells, name)(**kwargs)))


def _junction(first: gf.Component, second: gf.Component) -> kdb.Region:
    c = gf.Component()
    a = c << first
    b = c << second
    optical = [p for p in a.ports if p.port_type == "optical"]
    b.connect("o1", optical[-1])
    return _etch(c)


LENGTHS = np.r_[np.linspace(0.3, 3.0, 28), [9.99, 10.0, 37.3, 123.456, 500.0]]


@pytest.mark.parametrize("length", LENGTHS)
def test_junction_gaps(length):
    """Abutting cells keep >= 180nm between their slots for any straight length."""
    s = cells.straight(length=float(length))
    taper = cells.taper(length=float(length) + 1, width2=3.0)
    pairs = [
        (s, s),
        (s, cells.straight(length=float(length) + 0.137)),
        (s, taper),
        (taper, cells.straight(length=float(length), width=3.0)),
        (cells.grating_coupler_rectangular(), s),
        (cells.bend_circular(), cells.bend_circular()),
    ]
    for other in (cells.bend_circular(), cells.bend_euler(), cells.bend_s()):
        pairs += [(s, other), (other, s)]
    for first, second in pairs:
        _check_404(_junction(first, second))


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("straight", {"length": 30.0}),
        ("taper", {"length": 30.0, "width2": 8.0}),
        ("bend_s", {}),
    ],
)
def test_no_unslotted_window(name, kwargs):
    """Inside the 3.5um windows, only tether-sized Si (< 0.7um) is left."""
    c = getattr(cells, name)(**kwargs)
    core = kdb.Region(c.begin_shapes_rec(gf.get_layer(LAYER.WG_MARK))).merged()
    xmax = c.ports["o2"].center[0] * 1000
    clip = kdb.Region(kdb.Box(0, -(10**6), int(xmax), 10**6))
    windows = (core.sized(3300) & clip) - core.sized(50)
    leftover = (windows - _etch(c)).merged()
    assert not leftover.is_empty()
    assert max(p.bbox().width() for p in leftover.each()) < 700


def test_bend_euler_checks_local_radius():
    """A nominal-20um euler bend dips to ~14um and violates radius_min."""
    with pytest.raises(ValueError, match="radius_min"):
        cells.bend_euler(radius=TECH.radius_min_sus)


def test_taper_rejects_unsupported_width():
    """Suspended cores wider than 16um cannot be released."""
    with pytest.raises(ValueError, match="16"):
        cells.taper(width2=17.0)


def test_routing_defaults():
    """Routes use circular bends and the 75um suspended-waveguide spacing."""
    route_bundle = PDK.routing_strategies["route_bundle"]
    route_single = PDK.routing_strategies["route_single"]
    assert inspect.signature(route_bundle).parameters["separation"].default == 75
    for f in (route_bundle, route_single):
        assert inspect.signature(f).parameters["bend"].default == "bend_circular"
    assert PDK.layer_transitions == {LAYER.WG_MARK: cells.taper}


def test_route_bundle_and_auto_taper_layouts():
    """Bundles (with long-straight tapers) and width auto-tapers stay DRC clean."""
    c = gf.Component()
    starts, ends = [], []
    for i in range(2):
        a = c << cells.straight(length=20)
        b = c << cells.straight(length=20)
        a.dmove((0, 100 * i))
        b.dmove((700, 500 + 100 * i))
        starts.append(a.ports["o2"])
        ends.append(b.ports["o1"])
    routes = PDK.routing_strategies["route_bundle"](c, starts, ends)
    assert len(routes) == 2
    _check_404(_etch(c))

    c = gf.Component()
    wide = c << cells.straight(length=20, width=3.0)
    narrow = c << cells.straight(length=20)
    narrow.dmove((300, 200))
    gf.routing.route_single(
        c,
        wide.ports["o2"],
        narrow.ports["o1"],
        cross_section="xs_sus",
        bend="bend_circular",
        auto_taper=True,
    )
    assert any(inst.cell.name.startswith("taper") for inst in c.insts)
    _check_404(_etch(c))


@pytest.mark.parametrize("name", sorted(PDK.models))
@pytest.mark.parametrize("wl", [3.8, np.linspace(3.7, 3.9, 7)])
def test_registered_models_evaluate(name, wl):
    """Every model supports scalar and swept wavelengths under JIT."""
    result = jax.jit(PDK.models[name])(wl=wl)
    assert {p for pair in result for p in pair} == {"o1", "o2"}
    for value in result.values():
        assert np.shape(value) == np.shape(wl)
        assert np.isfinite(value).all()


def test_model_registry():
    """Every cell with a schematic model has a registered SAX model."""
    assert set(PDK.models) == {
        "straight",
        "taper",
        "bend_circular",
        "bend_euler",
        "bend_s",
        "grating_coupler_rectangular",
    }


def test_waveguide_phase_and_loss():
    """Straights use the mode-solved index and the 5 dB/cm loss target."""
    length = 1234.0
    s21 = models.straight(wl=models.WL0, length=length)["o1", "o2"]
    expected = 10 ** (-5 * length * 1e-4 / 20) * np.exp(
        2j * np.pi * models.NEFF * length / models.WL0
    )
    np.testing.assert_allclose(s21, expected, atol=1e-10)
    bend = models.bend_circular(wl=3.8)["o1", "o2"]
    straight = models.straight(wl=3.8, length=np.pi / 2 * TECH.radius_sus)["o1", "o2"]
    np.testing.assert_allclose(bend, straight)


def test_routed_circuit():
    """A routed grating-to-grating link simulates from its layout netlist."""
    c = gf.Component()
    g1 = c << cells.grating_coupler_rectangular()
    g2 = c << cells.grating_coupler_rectangular()
    g1.dmirror_x()
    g2.drotate(90)
    g2.dmove((600, 400))
    route = PDK.routing_strategies["route_single"](c, g1.ports["o1"], g2.ports["o1"])
    c.add_port("o1", port=g1.ports["o2"])
    c.add_port("o2", port=g2.ports["o2"])
    names = {inst.cell.name.split("_")[0] for inst in route.instances}
    assert "bend" in names

    _check_404(_etch(c))

    netlist = c.get_netlist()
    components = {i["component"] for i in netlist["instances"].values()}
    assert "bend_circular" in components
    circuit, _ = sax.circuit(netlist, models=PDK.models)
    wl = np.linspace(3.7, 3.9, 21)
    s21 = np.abs(np.asarray(circuit(wl=wl)["o1", "o2"])) ** 2

    length = sum(
        inst.cell.info["length"]
        for inst in c.insts
        if not inst.cell.name.startswith("grating")
    )
    gc = np.abs(np.asarray(models.grating_coupler_rectangular(wl=wl)["o1", "o2"])) ** 2
    expected = gc**2 * 10 ** (-5 * length * 1e-4 / 10)
    np.testing.assert_allclose(s21, expected, rtol=1e-5)
    assert np.argmax(s21) == 10
