"""Tests for the MPW Run 50 submission check (cspdk.si220.submission)."""

from __future__ import annotations

from pathlib import Path

import gdsfactory as gf
import pytest

from cspdk.si220 import PDK, cells
from cspdk.si220.submission import DESIGNATED_LAYERS, check_submission, main


@pytest.fixture(autouse=True)
def _activate() -> None:
    PDK.activate()


def _design(tmp_path: Path, name: str, die=cells.die, extra=None) -> Path:
    """A heater MZI and a grating coupler on a die, written to a GDS file."""
    gf.clear_cache()  # tests reuse top-cell names
    c = gf.Component(name)
    c << die()
    mzi = c << cells.mzi_heater_loop()
    mzi.dmove((-500, 0))
    gc = c << cells.grating_coupler_rectangular()
    gc.dmove((1000, 0))
    if extra:
        extra(c)
    path = tmp_path / f"{name}.gds"
    c.write_gds(path)
    return path


@pytest.mark.parametrize("die", [cells.die, cells.die_half, cells.die_with_pads])
def test_valid_designs_pass(tmp_path: Path, die) -> None:
    """A design on the full die, half die or packaging template passes."""
    assert check_submission(_design(tmp_path, "Cell0_Southampton", die)) == []


def test_top_cell_name(tmp_path: Path) -> None:
    """The top cell must be named Cell0_<Name of Institution>."""
    (problem,) = check_submission(_design(tmp_path, "my_design"))
    assert "Cell0_<Name of Institution>" in problem


def test_missing_cell_outline(tmp_path: Path) -> None:
    """A design needs the 99/0 cell outline."""
    (problem,) = check_submission(_design(tmp_path, "Cell0_X", die=gf.Component))
    assert "no cell outline" in problem


def test_wrong_design_area(tmp_path: Path) -> None:
    """The outline must be the full or half design area."""
    path = _design(tmp_path, "Cell0_X", die=lambda: cells.die(size=(10000, 4900)))
    (problem,) = check_submission(path)
    assert "10000 x 4900 um" in problem


def test_undesignated_layer_and_geometry_outside(tmp_path: Path) -> None:
    """Layer 60 is no longer designated, and nothing may leave the outline."""

    def extra(c: gf.Component) -> None:
        c.add_polygon([(0, 0), (5, 0), (5, 5)], layer=(60, 0))
        c.add_polygon([(6000, 0), (6005, 0), (6005, 5)], layer=(3, 0))

    problems = check_submission(_design(tmp_path, "Cell0_X", extra=extra))
    assert problems == [
        "layer 60/0 is not a designated MPW-50 layer",
        "layer 3/0 extends outside the cell outline",
    ]


def test_file_format(tmp_path: Path) -> None:
    """Only GDSII and OASIS files are accepted."""
    path = tmp_path / "design.dxf"
    path.write_text("")
    (problem,) = check_submission(path)
    assert "GDSII" in problem


def test_pdk_cells_use_designated_layers_only() -> None:
    """Every si220 cell can go into a submission without layer problems."""
    for name in PDK.cells:
        c = gf.get_component(name)
        for index in c.kcl.layer_indexes():
            info = c.kcl.get_info(index)
            if not c.bbox(index).empty():
                assert (info.layer, info.datatype) in DESIGNATED_LAYERS, name


def test_command_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The command line exits 1 when any file has a problem."""
    good = _design(tmp_path, "Cell0_Southampton")
    bad = _design(tmp_path, "my_design")
    assert main([str(good)]) == 0
    assert main([str(good), str(bad)]) == 1
    assert "OK" in capsys.readouterr().out


def test_off_grid_points(tmp_path: Path) -> None:
    """With a 0.5 nm database unit, points must still sit on the 1 nm grid."""
    import klayout.db as kdb

    layout = kdb.Layout()
    layout.dbu = 0.0005
    top = layout.create_cell("Cell0_Grid")
    top.shapes(layout.layer(99, 0)).insert(kdb.DBox(-5735, -2450, 5735, 2450))
    top.shapes(layout.layer(3, 0)).insert(kdb.DBox(0, 0, 10.0005, 0.5))
    path = tmp_path / "grid.gds"
    layout.write(str(path))
    assert check_submission(path) == ["layer 3/0 has points off the 1 nm grid"]


def _outline_layout(dbu: float = 0.001) -> tuple:
    """A layout whose top cell holds only the full design-area outline."""
    import klayout.db as kdb

    layout = kdb.Layout()
    layout.dbu = dbu
    top = layout.create_cell("Cell0_Test")
    top.shapes(layout.layer(99, 0)).insert(kdb.DBox(-5735, -2450, 5735, 2450))
    return layout, top


def test_unreadable_file_is_a_problem(tmp_path: Path) -> None:
    """A broken file is reported, and the command line carries on to the next."""
    broken = tmp_path / "broken.gds"
    broken.write_bytes(b"not a gds file")
    (problem,) = check_submission(broken)
    assert "could not read the layout" in problem
    assert main([str(broken), str(_design(tmp_path, "Cell0_Southampton"))]) == 1


def test_text_labels_on_grid_and_inside(tmp_path: Path) -> None:
    """Text labels are checked like shapes: on the grid and inside the outline."""
    import klayout.db as kdb

    layout, top = _outline_layout(dbu=0.0005)
    labels = top.shapes(layout.layer(100, 0))
    labels.insert(kdb.DText("off grid", 10.0005, 0))
    labels.insert(kdb.DText("outside", 6000, 0))
    path = tmp_path / "labels.gds"
    layout.write(str(path))
    assert check_submission(path) == [
        "layer 100/0 has points off the 1 nm grid",
        "layer 100/0 extends outside the cell outline",
    ]


def test_coarser_database_unit_is_on_grid(tmp_path: Path) -> None:
    """A 2 nm database unit keeps every coordinate on the 1 nm grid."""
    import klayout.db as kdb

    layout, top = _outline_layout(dbu=0.002)
    top.shapes(layout.layer(3, 0)).insert(kdb.DBox(0, 0, 10.002, 0.5))
    path = tmp_path / "dbu2nm.gds"
    layout.write(str(path))
    assert check_submission(path) == []


def test_cell_names_are_escaped(tmp_path: Path) -> None:
    """Cell names are quoted with repr(), so untrusted names can't inject control codes.

    KLayout's own writers already replace control characters, so this checks the
    quoting; files from other tools are covered by the same repr().
    """
    import klayout.db as kdb

    layout = kdb.Layout()
    layout.create_cell("Cell0_A\x1b[2J")
    layout.create_cell("Cell0_B")
    path = tmp_path / "two_tops.oas"
    layout.write(str(path))
    (problem,) = check_submission(path)
    assert "'Cell0_B'" in problem
    assert "\x1b" not in problem
