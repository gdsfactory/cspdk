"""Tests of cspdk.si220_ebeam_event."""

from __future__ import annotations

import pathlib

import gdsfactory as gf
import jsondiff
import kfactory as kf
import numpy as np
import pytest
from conftest import difftest
from pytest_regressions.data_regression import DataRegressionFixture

from cspdk.si220_ebeam_event import (
    PDK,
    TECH,
    cells,
    check,
    models,
    post_process,
    tapeout,
)
from cspdk.si220_ebeam_event.mask import exposure_region
from cspdk.si220_ebeam_event.tech import LAYER, route_bundle, route_single


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
        ("straight", {"width": 0.19}),
        ("bend_circular", {"width": 0.19}),
        ("bend_euler", {"width": 0.19}),
    ],
)
def test_cells_reject_rule_violations(cell: str, kwargs: dict) -> None:
    """Radius < 25 um, gap or width < 0.2 um and teeth/gaps < 0.2 um raise ValueError."""
    with pytest.raises(ValueError, match="minimum"):
        gf.get_component(cell, **kwargs)


def test_grating_coupler_matches_student_rib_grating() -> None:
    """10 um wide, 350 um taper, 60 teeth of 335 nm at 670 nm on GRA (6/0)."""
    gc = cells.grating_coupler()
    teeth = gc.get_region(LAYER.GRA, merge=True)
    assert teeth.count() == 60
    boxes = sorted((p.bbox() for p in teeth.each()), key=lambda b: b.left)
    assert {b.width() for b in boxes} == {335}
    assert {b.height() for b in boxes} == {10_000}
    assert boxes[0].left == 350_000  # first tooth at the end of the taper
    assert {b2.left - b1.left for b1, b2 in zip(boxes, boxes[1:])} == {670}
    wg = gc.get_region(LAYER.WG, merge=True)
    assert wg.count() == 1  # solid core under the teeth
    assert wg.bbox().height() == 10_000
    assert gc.ports["o1"].width == pytest.approx(TECH.width_rib)


def test_grating_coupler_sweep_rows() -> None:
    """One loopback per (period, fill factor), stacked at the I/O pitch."""
    sweep = cells.grating_coupler_sweep(periods=(0.66, 0.68), fill_factors=(0.5,))
    assert sweep.info["rows"] == [[0.66, 0.5], [0.68, 0.5]]
    assert sweep.get_region(LAYER.GRA, merge=True).count() == 2 * 2 * 60
    assert sweep.ysize == pytest.approx(TECH.io_pitch + 10)
    # each row is one continuous core from grating to grating
    assert sweep.get_region(LAYER.WG, merge=True).count() == 2


################
# Routing
################


def test_routing_strategies_route_with_bends() -> None:
    """route_single and route_bundle turn corners with 25 um bends and pass DRC."""
    c = gf.Component()
    a = c << cells.straight()
    b = c << cells.straight()
    b.rotate(90)
    b.move((200, 200))
    route_single(c, a.ports["o2"], b.ports["o1"])

    left = [c << cells.straight() for _ in range(2)]
    right = [c << cells.straight() for _ in range(2)]
    for i, (s1, s2) in enumerate(zip(left, right)):
        s1.move((0, 400 + 10 * i))
        s2.rotate(90)
        s2.move((310 - 10 * i, 700))  # the lower route takes the outer corner
    route_bundle(c, [s.ports["o2"] for s in left], [s.ports["o1"] for s in right])
    # a failed route draws markers on routing_error_marker instead of raising
    assert c.get_region(LAYER.routing_error_marker, merge=True).is_empty()
    assert check(c) == []


################
# Post-processing
################


def test_post_process_trenches_on_a_straight() -> None:
    """A straight gets a 5 um trench on every side, merged into 4/0."""
    s = cells.straight(length=10)
    exposure = post_process(s).get_region(LAYER.EXPOSURE, merge=True)
    assert exposure.count() == 1
    assert exposure.bbox().to_dtype(0.001) == kf.kdb.DBox(-5, -5.225, 15, 5.225)
    assert _area(exposure) == pytest.approx(20 * 10.45 - 10 * 0.45)
    # the core itself is not exposed
    assert (exposure & s.get_region(LAYER.WG, merge=True)).is_empty()


def test_post_process_merges_gra_and_exposure() -> None:
    """Drawn 6/0 and 4/0 shapes are merged into 4/0; 3/1 is not in the output."""
    c = gf.Component()
    c << cells.straight(length=10)
    c.add_polygon([(100, 0), (110, 0), (110, 10), (100, 10)], layer=LAYER.EXPOSURE)
    c.add_polygon([(200, 0), (201, 0), (201, 10), (200, 10)], layer=LAYER.GRA)
    c.add_polygon([(14, -1), (30, -1), (30, 1), (14, 1)], layer=LAYER.EXPOSURE)
    out = post_process(c)
    exposure = out.get_region(LAYER.EXPOSURE, merge=True)
    trench = 20 * 10.45 - 10 * 0.45
    overlap = 1 * 2  # the 4/0 patch inside the trench from x = 14 to 15
    assert _area(exposure) == pytest.approx(trench + 100 + 10 + 32 - overlap)
    assert exposure.count() == 3  # trench + patch merged, 4/0 square, 6/0 bar
    assert out.get_region(LAYER.TRENCH_TMP, merge=True).is_empty()
    assert out.get_region(LAYER.GRA, merge=True).count() == 1  # other layers are kept
    assert _area(exposure_region(c)) == pytest.approx(_area(exposure))


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
    ("layer", "name", "merged_gap", "merged_feature"),
    [
        # cores are unexposed, so a gap between cores is an exposed feature
        (LAYER.WG, "WG", "MERGED.W", "MERGED.S"),
        (LAYER.EXPOSURE, "EXPOSURE", "MERGED.S", "MERGED.W"),
        (LAYER.GRA, "GRA", "MERGED.S", "MERGED.W"),
    ],
)
def test_drc_catches_150nm_gap_and_feature(
    layer, name: str, merged_gap: str, merged_feature: str
) -> None:
    """A 150 nm gap or feature fails on 3/0, 4/0, 6/0 and the merged 4/0; 200 nm passes."""
    assert _rules(_two_lines(gap=0.15, layer=layer)) == {f"{name}.S", merged_gap}
    assert _rules(_two_lines(gap=1.0, width=0.15, layer=layer)) == {
        f"{name}.W",
        merged_feature,
    }
    assert _rules(_two_lines(gap=0.2, width=0.2, layer=layer)) == set()


def _straight_with_patch(distance: float) -> gf.Component:
    """A straight, plus a drawn 4/0 patch ``distance`` um above its trench."""
    c = gf.Component()
    c << cells.straight(length=10)
    y = TECH.width_rib / 2 + TECH.trench_width + distance
    c.add_polygon([(0, y), (10, y), (10, y + 1), (0, y + 1)], layer=LAYER.EXPOSURE)
    return c


def test_drc_catches_sliver_on_merged_exposure() -> None:
    """A drawn 4/0 patch 150 nm from a trench leaves a sliver of Si on the merged 4/0."""
    assert _rules(_straight_with_patch(0.15)) == {"MERGED.S"}
    assert _rules(_straight_with_patch(0.2)) == set()


@pytest.mark.parametrize(
    ("trench_gap", "exposed_parts"),
    [(0.05, 1), (0.15, 1), (0.2, 1), (0.25, 2)],
)
def test_post_process_fills_slivers_between_trenches(
    trench_gap: float, exposed_parts: int
) -> None:
    """Gaps between two trenches narrower than 210 nm are etched; wider ones stay."""
    c = gf.Component()
    c << cells.straight(length=10)
    other = c << cells.straight(length=10)
    other.movey(TECH.width_rib + 2 * TECH.trench_width + trench_gap)
    assert exposure_region(c).count() == exposed_parts
    assert check(c) == []


@pytest.mark.parametrize(
    ("cell", "kwargs"),
    [
        ("bend_circular", {"radius": 25.0}),
        ("ring_single", {"gap": 0.2}),
        ("ring_single", {"gap": 0.5, "radius": 40.0}),
        ("ring_single", {"gap": 0.4, "radius": 28.0}),
        ("ring_single", {"gap": 0.6, "radius": 30.0}),
        ("coupler_ring", {"gap": 0.2}),
        ("coupler", {"gap": 0.2}),
        ("bend_euler", {"radius": 50.0}),
        ("bend_s", {"size": (100.0, 20.0)}),
    ],
)
def test_cells_pass_merged_rules(cell: str, kwargs: dict) -> None:
    """Rule minimums and other non-default geometry pass, merged 4/0 included."""
    assert check(gf.get_component(cell, **kwargs)) == []


def test_drc_catches_diagonal_corner_gap() -> None:
    """Two cores whose corners are 150 nm apart on a diagonal fail; 200 nm passes."""

    def corners(gap: float) -> gf.Component:
        c = gf.Component()
        c.add_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], layer=LAYER.WG)
        d = 1 + gap / 2**0.5
        c.add_polygon([(d, d), (d + 1, d), (d + 1, d + 1), (d, d + 1)], layer=LAYER.WG)
        return c

    assert _rules(corners(0.15)) == {"WG.S"}
    assert _rules(corners(0.21)) == set()


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
    assert [v.rule for v in violations] == ["WG.S", "MERGED.W"]
    assert "gap < 200 nm" in str(violations[0])


def test_drc_catches_exposure_area() -> None:
    """More than 2,000,000 um^2 of post-processed exposure fails."""
    c = gf.Component()
    c.add_polygon([(0, 0), (1500, 0), (1500, 1500), (0, 1500)], layer=LAYER.EXPOSURE)
    (violation,) = check(c)
    assert violation.rule == "EXPOSURE.A"
    assert violation.value == pytest.approx(2.25e6)


def test_drc_counts_trenches_in_exposure_area() -> None:
    """The area rule applies after post-processing: WG trenches count."""
    square = gf.Component()
    square.add_polygon(
        [(0, 100), (1900, 100), (1900, 1100), (0, 1100)], layer=LAYER.EXPOSURE
    )
    assert check(square) == []  # 1.9 mm^2
    c = gf.Component()
    c << square
    # 4 x 5 mm of straight waveguide adds ~0.2 mm^2 of trench
    for i in range(4):
        wg = c << cells.straight(length=5_000)
        wg.movey(1200 + 100 * i)
    assert _rules(c) == {"EXPOSURE.A"}


def test_drc_catches_design_larger_than_6x3mm() -> None:
    """The design, trenches included, fits in 6 mm east-west by 3 mm north-south."""
    assert check(cells.straight(length=5_990)) == []  # 6 mm with its 5 um trenches
    assert _rules(cells.straight(length=5_991)) == {"DESIGN.SIZE"}

    tall = gf.Component()
    tall.add_polygon([(0, 0), (1, 0), (1, 3001), (0, 3001)], layer=LAYER.GRA)
    assert _rules(tall) == {"DESIGN.SIZE"}
    # 3 x 6 mm (rotated) does not fit: the I/O runs east-west
    turned = gf.Component()
    turned.add_polygon([(0, 0), (1, 0), (1, 5000), (0, 5000)], layer=LAYER.GRA)
    assert _rules(turned) == {"DESIGN.SIZE"}


@pytest.mark.parametrize(
    ("layer", "rule"),
    [
        (LAYER.TRENCH_TMP, "TRENCH_TMP.R"),  # reserved for post-processing
        (LAYER.routing_error_marker, "ROUTING.E"),  # left by a failed route
        ((5, 0), "LAYER.U"),  # not written, e.g. a core on the wrong layer
    ],
)
def test_drc_flags_shapes_that_are_not_written(layer, rule: str) -> None:
    """Shapes the foundry would drop or that mark a failed route are reported."""
    c = gf.Component()
    c.add_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], layer=layer)
    assert _rules(c) == {rule}


def test_drc_catches_gap_at_a_sharp_tip() -> None:
    """A gap between separate shapes is measured whatever the edge angles."""
    c = gf.Component()
    c.add_polygon([(0, -5), (1, -5), (1, 5), (0, 5)], layer=LAYER.GRA)
    # a 1 um wide wedge whose tip is 150 nm from the bar
    c.add_polygon([(1.15, 0), (6.15, -0.5), (6.15, 0.5)], layer=LAYER.GRA)
    assert "GRA.S" in _rules(c)


def test_drc_catches_exposure_outside_floorplan() -> None:
    """With a FLOORPLAN drawn, the exposure must lie inside it."""
    c = gf.Component()
    c.add_polygon([(0, 0), (100, 0), (100, 100), (0, 100)], layer=LAYER.FLOORPLAN)
    wg = c << cells.straight(length=50)
    wg.move((25, 50))
    assert check(c) == []
    wg.movex(60)
    assert _rules(c) == {"DESIGN.SIZE"}


################
# Tapeout
################


def test_tapeout_passes_die_and_writes_exposure(tmp_path, capsys) -> None:
    """A clean GDS passes (exit 0) and its post-processed 4/0 is written."""
    gds = cells.die().write_gds(tmp_path / "die.gds")
    assert tapeout.main([str(gds), "-o", str(tmp_path / "out")]) == 0
    output = capsys.readouterr().out
    assert output.startswith("PASS")
    assert "of 2,000,000" in output

    exposed = gf.import_gds(tmp_path / "out" / "die_exposure.gds")
    assert _area(exposed.get_region(LAYER.EXPOSURE, merge=True)) == pytest.approx(
        _area(exposure_region(cells.die())), rel=1e-6
    )


def test_tapeout_fails_on_violations(tmp_path, capsys) -> None:
    """A GDS that breaks a rule fails (exit 1) and lists the rule."""
    gds = _two_lines(gap=0.15).write_gds(tmp_path / "gap.gds")
    assert tapeout.main([str(gds)]) == 1
    output = capsys.readouterr().out
    assert output.startswith("FAIL")
    assert "WG.S" in output


def test_tapeout_reports_unreadable_files_and_continues(tmp_path, capsys) -> None:
    """An unreadable GDS fails without stopping the other files."""
    bad = tmp_path / "bad.gds"
    bad.write_bytes(b"not a gds")
    good = cells.straight().write_gds(tmp_path / "good.gds")
    assert tapeout.main([str(bad), str(good)]) == 1
    output = capsys.readouterr().out
    assert f"FAIL {bad}: cannot read" in output
    assert f"PASS {good}" in output


def test_tapeout_rejects_clashing_output_names(tmp_path) -> None:
    """With -o, two inputs named alike would overwrite each other's output."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    gds = [cells.straight().write_gds(tmp_path / d / "x.gds") for d in "ab"]
    with pytest.raises(SystemExit):
        tapeout.main([*map(str, gds), "-o", str(tmp_path / "out")])


################
# Die
################


def test_die_size_and_placeholder_io() -> None:
    """6 x 3 mm floorplan; inputs west facing east, outputs east facing west."""
    d = cells.die()
    floorplan = d.get_region(LAYER.FLOORPLAN, merge=True)
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


@pytest.mark.parametrize("name", sorted(models.get_models()))
def test_models_follow_wavelength_arrays(name: str) -> None:
    """Every S-parameter has one value per wavelength."""
    sdict = models.get_models()[name](wl=np.array([1.53, 1.55, 1.57]))
    for value in sdict.values():
        assert np.shape(value) == (3,), name


def test_grating_model_peak() -> None:
    """6 dB peak loss at 1.565 um (student-batch rib grating)."""
    s21 = models.grating_coupler(wl=1.565)["o1", "o2"]
    assert -20 * np.log10(abs(s21)) == pytest.approx(6.0)
