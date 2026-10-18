"""Tests of the EVENT-ONLY CORNERSTONE e-beam flavour (cspdk.si220_ebeam_event).

Regression data (GDS, settings, netlists) as for the other flavours, plus the
event rules: post-processing, DRC, parameter checks, die and models.
"""

from __future__ import annotations

import pathlib

import gdsfactory as gf
import jsondiff
import kfactory as kf
import numpy as np
import pytest
from conftest import difftest
from pytest_regressions.data_regression import DataRegressionFixture

from cspdk.si220_ebeam_event import PDK, TECH, cells, check, models, post_process
from cspdk.si220_ebeam_event.mask import exposure_region, layer_region
from cspdk.si220_ebeam_event.tech import LAYER


@pytest.fixture(autouse=True)
def activate_pdk() -> None:
    """Activate the PDK."""
    PDK.activate()


cell_names = sorted(name for name in PDK.cells if not name.startswith("_"))
dirpath = pathlib.Path(__file__).absolute().parent / "gds_ref_si220_ebeam_event"
dirpath.mkdir(exist_ok=True, parents=True)

UM2 = 1e-6  # dbu^2 (1 nm grid) to um^2


def _area(region) -> float:
    return region.area() * UM2


def _rules(component: gf.Component) -> set[str]:
    return {v.rule for v in check(component)}


################
# Regression
################


@pytest.mark.parametrize("component_name", cell_names)
def test_gds(component_name: str) -> None:
    """Avoid regressions in GDS geometry shapes and layers."""
    component = PDK.cells[component_name]()
    difftest(component, test_name=component_name, dirpath=dirpath)


@pytest.mark.parametrize("component_name", cell_names)
def test_settings(component_name: str, data_regression: DataRegressionFixture) -> None:
    """Avoid regressions when exporting settings."""
    component = PDK.cells[component_name]()
    data_regression.check(component.to_dict(with_ports=True))


@pytest.mark.parametrize("component_type", cell_names)
def test_netlists(component_type: str, data_regression: DataRegressionFixture) -> None:
    """Netlist regression and Component -> netlist -> Component round trip."""
    c = PDK.cells[component_type]()
    n = c.get_netlist()
    data_regression.check(n)

    n.pop("connections", None)
    n.pop("warnings", None)
    yaml_str = c.write_netlist(n)
    for ci in list(c.kcl.each_cell_top_down()):
        gf.kcl.dkcells[ci].delete()

    c2 = gf.read.from_yaml(yaml_str)
    d = jsondiff.diff(n, c2.get_netlist())
    for key in ("warnings", "connections", "ports"):
        d.pop(key, None)
    assert len(d) == 0, d
    for ci in list(c2.kcl.each_cell_top_down()):
        gf.kcl.dkcells[ci].delete()


@pytest.mark.parametrize("component_name", cell_names)
def test_optical_port_positions(component_name: str) -> None:
    """Every optical port sits on an edge of its layer as wide as the port."""
    component = PDK.cells[component_name]()
    for port in component.ports:
        if port.port_type != "optical":
            continue
        edges = kf.kdb.Region(component.begin_shapes_rec(port.layer)).edges()
        tolerance = 0.001
        marker = (
            kf.kdb.DPolygon(kf.kdb.DBox(-tolerance, -tolerance, tolerance, tolerance))
            .transformed(port.dcplx_trans)
            .to_itype(port.kcl.dbu)
        )
        touching = edges.interacting(kf.kdb.Region(marker))
        assert not touching.is_empty(), port.name
        edge = next(iter(touching.each()))
        assert np.isclose(edge.length() * port.kcl.dbu, port.width, atol=1e-3)


################
# Cells
################


@pytest.mark.parametrize("component_name", cell_names)
def test_default_cells_pass_drc(component_name: str) -> None:
    """Every cell builds with its defaults and passes the event rules."""
    component = PDK.cells[component_name]()
    assert check(component) == []


@pytest.mark.parametrize(
    ("cell", "kwargs"),
    [
        ("bend_circular", {"radius": 24.9}),
        ("bend_euler", {"radius": 24.9}),
        # radius 30 is above 25, but the Euler curve is tighter (~21 um) in its middle
        ("bend_euler", {"radius": 30.0}),
        ("bend_s", {"size": (20.0, 5.0)}),
        ("coupler_ring", {"radius": 20.0}),
        ("ring_single", {"radius": 20.0}),
        ("coupler", {"gap": 0.19}),
        ("coupler_ring", {"gap": 0.19}),
        ("ring_single", {"gap": 0.19}),
        ("grating_coupler", {"fill_factor": 0.25}),
        ("grating_coupler", {"fill_factor": 0.75}),
    ],
)
def test_cells_reject_rule_violations(cell: str, kwargs: dict) -> None:
    """Radius < 25 um, gap < 0.2 um and teeth/gaps < 0.2 um raise ValueError."""
    with pytest.raises(ValueError, match="minimum"):
        gf.get_component(cell, **kwargs)


def test_minimum_values_are_accepted() -> None:
    """The rule minimums themselves pass."""
    cells.bend_circular(radius=TECH.radius_min)
    cells.coupler(gap=TECH.min_space)
    cells.ring_single(gap=TECH.min_space, radius=TECH.radius_min)


def test_grating_coupler_matches_student_rib_grating() -> None:
    """10 um wide, 350 um taper, 60 teeth of 335 nm at 670 nm on GRA (6/0)."""
    gc = cells.grating_coupler()
    teeth = layer_region(gc, LAYER.GRA)
    assert teeth.count() == 60
    boxes = sorted((p.bbox() for p in teeth.each()), key=lambda b: b.left)
    assert {b.width() for b in boxes} == {335}
    assert {b.height() for b in boxes} == {10_000}
    assert boxes[0].left == 350_000  # first tooth at the end of the taper
    assert {b2.left - b1.left for b1, b2 in zip(boxes, boxes[1:])} == {670}
    wg = layer_region(gc, LAYER.WG)
    assert wg.count() == 1  # solid core under the teeth
    assert wg.bbox().height() == 10_000
    assert gc.ports["o1"].width == pytest.approx(TECH.width_rib)


def test_grating_coupler_sweep_rows() -> None:
    """One loopback per (period, fill factor), stacked at the I/O pitch."""
    sweep = cells.grating_coupler_sweep(periods=(0.66, 0.68), fill_factors=(0.5,))
    assert sweep.info["rows"] == [[0.66, 0.5], [0.68, 0.5]]
    assert layer_region(sweep, LAYER.GRA).count() == 2 * 2 * 60
    assert sweep.ysize == pytest.approx(TECH.io_pitch + 10)


################
# Post-processing
################


def test_post_process_trenches_on_a_straight() -> None:
    """A straight gets a 5 um trench on every side, merged into 4/0."""
    s = cells.straight(length=10)
    exposure = layer_region(post_process(s), LAYER.EXPOSURE)
    assert exposure.count() == 1
    assert exposure.bbox().to_dtype(0.001) == kf.kdb.DBox(-5, -5.225, 15, 5.225)
    assert _area(exposure) == pytest.approx(20 * 10.45 - 10 * 0.45)
    # the core itself is not exposed
    assert (exposure & layer_region(s, LAYER.WG)).is_empty()


def test_post_process_merges_gra_and_exposure() -> None:
    """Drawn 6/0 and 4/0 shapes are merged into 4/0; 3/1 is not in the output."""
    c = gf.Component()
    c << cells.straight(length=10)
    c.add_polygon([(100, 0), (110, 0), (110, 10), (100, 10)], layer=LAYER.EXPOSURE)
    c.add_polygon([(200, 0), (201, 0), (201, 10), (200, 10)], layer=LAYER.GRA)
    c.add_polygon([(14, -1), (30, -1), (30, 1), (14, 1)], layer=LAYER.EXPOSURE)
    out = post_process(c)
    exposure = layer_region(out, LAYER.EXPOSURE)
    trench = 20 * 10.45 - 10 * 0.45
    overlap = 1 * 2  # the 4/0 patch inside the trench from x = 14 to 15
    assert _area(exposure) == pytest.approx(trench + 100 + 10 + 32 - overlap)
    assert exposure.count() == 3  # trench + patch merged, 4/0 square, 6/0 bar
    assert layer_region(out, LAYER.TRENCH_TMP).is_empty()
    assert layer_region(out, LAYER.GRA).count() == 1  # other layers are kept
    assert _area(exposure_region(c)) == pytest.approx(_area(exposure))


def test_grating_teeth_inside_exposure() -> None:
    """The grating teeth (6/0) end up in the exposure (4/0)."""
    gc = cells.grating_coupler()
    exposure = exposure_region(gc)
    assert (layer_region(gc, LAYER.GRA) - exposure).is_empty()


################
# DRC
################


def _two_lines(gap: float, width: float = 0.45, layer=LAYER.WG) -> gf.Component:
    c = gf.Component()
    c.add_polygon([(0, 0), (10, 0), (10, width), (0, width)], layer=layer)
    y = width + gap
    c.add_polygon([(0, y), (10, y), (10, y + width), (0, y + width)], layer=layer)
    return c


@pytest.mark.parametrize(
    ("layer", "name"),
    [(LAYER.WG, "WG"), (LAYER.EXPOSURE, "EXPOSURE"), (LAYER.GRA, "GRA")],
)
def test_drc_catches_150nm_gap_and_feature(layer, name: str) -> None:
    """A 150 nm gap or 150 nm feature fails on 3/0, 4/0 and 6/0; 200 nm passes."""
    assert _rules(_two_lines(gap=0.15, layer=layer)) == {f"{name}.S"}
    assert _rules(_two_lines(gap=1.0, width=0.15, layer=layer)) == {f"{name}.W"}
    assert _rules(_two_lines(gap=0.2, width=0.2, layer=layer)) == set()


def test_drc_catches_150nm_ring_gap() -> None:
    """A ring drawn with a 150 nm gap (bypassing the cell check) is flagged."""
    c = gf.Component()
    c << gf.c.ring_single(
        gap=0.15,
        radius=25,
        bend=gf.c.bend_circular,
        coupler_ring=gf.c.coupler_ring,
        cross_section="rib",
    )
    violations = check(c)
    assert [v.rule for v in violations] == ["WG.S"]
    assert "gap < 200 nm" in str(violations[0])


def test_drc_catches_exposure_area() -> None:
    """More than 2,000,000 um^2 of post-processed exposure fails."""
    c = gf.Component()
    c.add_polygon([(0, 0), (1500, 0), (1500, 1500), (0, 1500)], layer=LAYER.EXPOSURE)
    (violation,) = check(c)
    assert violation.rule == "EXPOSURE.A"
    assert violation.value == pytest.approx(2.25e6)

    ok = gf.Component()
    ok.add_polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)], layer=LAYER.EXPOSURE)
    assert check(ok) == []


def test_drc_counts_trenches_in_exposure_area() -> None:
    """The area rule applies after post-processing: WG trenches count."""
    square = gf.Component()
    square.add_polygon(
        [(0, 100), (1900, 100), (1900, 1100), (0, 1100)], layer=LAYER.EXPOSURE
    )
    assert check(square) == []  # 1.9 mm^2
    c = gf.Component()
    c << square
    # 20 mm of straight waveguide adds ~0.2 mm^2 of trench
    c << cells.straight(length=20_000)
    assert _rules(c) == {"EXPOSURE.A"}


def test_drc_flags_drawn_trench_tmp() -> None:
    """3/1 is reserved for post-processing."""
    c = gf.Component()
    c.add_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], layer=LAYER.TRENCH_TMP)
    assert _rules(c) == {"TRENCH_TMP.R"}


################
# Die
################


def test_die_size_and_placeholder_io() -> None:
    """6 x 3 mm floorplan; inputs west facing east, outputs east facing west."""
    d = cells.die()
    floorplan = layer_region(d, LAYER.FLOORPLAN)
    assert floorplan.bbox().to_dtype(0.001) == kf.kdb.DBox(-3000, -1500, 3000, 1500)
    assert d.xsize == pytest.approx(6000) and d.ysize == pytest.approx(3000)

    n = 8
    inputs = [d.ports[f"o{i}"] for i in range(1, n + 1)]
    outputs = [d.ports[f"o{i}"] for i in range(n + 1, 2 * n + 1)][::-1]
    assert len(d.ports) == 2 * n
    assert {p.orientation for p in inputs} == {0}
    assert {p.orientation for p in outputs} == {180}
    for group in (inputs, outputs):
        ys = [p.center[1] for p in group]
        assert np.allclose(np.diff(ys), TECH.io_pitch)
        assert len({p.center[0] for p in group}) == 1
    # input and output rows are aligned and face each other
    assert [p.center[1] for p in inputs] == [p.center[1] for p in outputs]
    assert inputs[0].center[0] < outputs[0].center[0]

    # fiber positions: inputs at least 500 um from the outputs
    fibers_in = {inst.ports["o2"].center[0] for inst in d.insts if inst.xmin < 0}
    fibers_out = {inst.ports["o2"].center[0] for inst in d.insts if inst.xmin > 0}
    assert len(fibers_in) == len(fibers_out) == 1
    separation = fibers_out.pop() - fibers_in.pop()
    assert separation == pytest.approx(d.info["io_separation"])
    assert separation >= TECH.io_min_separation
    for inst in d.insts:
        assert -3000 <= inst.xmin and inst.xmax <= 3000


def test_die_rejects_bad_io() -> None:
    """Too many gratings, a tighter pitch or too little separation raise."""
    with pytest.raises(ValueError, match="do not fit"):
        cells.die(n_io=13)
    with pytest.raises(ValueError, match="minimum"):
        cells.die(io_pitch=200)
    with pytest.raises(ValueError, match="minimum"):
        cells.die(size=(700, 3000))


################
# Models
################


def test_models_cover_the_cells() -> None:
    """Every cell with optical behaviour has a model of the same name."""
    assert set(PDK.models) == {
        "straight",
        "bend_circular",
        "bend_euler",
        "bend_s",
        "coupler",
        "coupler_ring",
        "grating_coupler",
    }


def test_straight_model_uses_event_numbers() -> None:
    """2.5 dB/cm loss and group index 3.85 at 1.55 um."""
    length = 10_000.0  # 1 cm
    s21 = models.straight(wl=1.55, length=length)["o1", "o2"]
    assert -20 * np.log10(abs(s21)) == pytest.approx(2.5)
    dwl = 1e-4
    phase = [
        np.angle(models.straight(wl=wl, length=1.0)["o1", "o2"])
        for wl in (1.55 - dwl, 1.55 + dwl)
    ]
    # ng = -(wl^2 / 2 pi) d(phase)/d(wl) per unit length
    ng = -(1.55**2) / (2 * np.pi) * (phase[1] - phase[0]) / (2 * dwl)
    assert ng == pytest.approx(3.85, rel=1e-3)


def test_bend_models_match_cell_lengths() -> None:
    """Bend model lengths follow the cells' geometry."""
    for name in ("bend_circular", "bend_euler", "bend_s"):
        cell = PDK.cells[name]()
        loss = 1.0e4  # dB/cm, large so the length dominates |S21|
        s21 = getattr(models, name)(loss=loss)["o1", "o2"]
        length_um = -20 * np.log10(abs(s21)) / loss * 1e4
        assert length_um == pytest.approx(cell.info["length"], rel=1e-3), name


def test_grating_model_peak() -> None:
    """6 dB peak loss at 1.565 um (student-batch rib grating)."""
    s21 = models.grating_coupler(wl=1.565)["o1", "o2"]
    assert -20 * np.log10(abs(s21)) == pytest.approx(6.0)
