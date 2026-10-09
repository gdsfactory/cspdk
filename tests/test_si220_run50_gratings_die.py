"""Run-50 grating couplers and dies against the Cornerstone template and rules.

The template cells have their waveguide port at the origin facing east; the cspdk
gratings face west (gdsfactory convention), so the template grating cells are
rotated by 180 degrees about the origin before the XOR. The dies are compared as is.
"""

from __future__ import annotations

import gdsfactory as gf
import klayout.db as kdb
import numpy as np
import pytest
from _run50 import template_region

from cspdk import si220

LAYERS = [(3, 0), (4, 0), (5, 0), (6, 0), (39, 0), (41, 0), (99, 0), (100, 0)]
R180 = kdb.Trans(2, False, 0, 0)


@pytest.fixture(autouse=True)
def activate_pdk() -> None:
    """Activate the si220 PDK."""
    si220.PDK.activate()


def _region(component: gf.Component, layer: tuple[int, int]) -> kdb.Region:
    index = component.kcl.layout.layer(*layer)
    return kdb.Region(component.begin_shapes_rec(index))


GRATINGS = [
    ("SOI220_cband_TE_strip_GC", "strip_cband"),
    ("SOI220_cband_TE_rib_GC", "rib_cband"),
    ("SOI220_oband_TE_strip_GC", "strip_oband"),
    ("SOI220_oband_TE_rib_GC", "rib_oband"),
]


@pytest.mark.parametrize(("name", "cross_section"), GRATINGS)
def test_grating_coupler_matches_template(name: str, cross_section: str) -> None:
    """The foundry-named cell and the parametric cell equal the template cell."""
    for component in (
        getattr(si220.cells, name)(),
        si220.cells.grating_coupler_rectangular(cross_section=cross_section),
    ):
        for layer in LAYERS:
            ours = _region(component, layer)
            theirs = template_region(name, layer).transformed(R180)
            assert (ours ^ theirs).is_empty(), (name, layer)
        assert component.ports["o1"].center == (0, 0)
        assert component.ports["o1"].orientation == 180


def test_rib_partial_is_the_cband_rib_grating() -> None:
    """The rib C-band grating has the 0.63 um period of the foundry cell."""
    component = si220.cells.grating_coupler_rectangular_rib()
    assert component.info["period"] == 0.63
    assert component.info["n_periods"] == 60


@pytest.mark.parametrize(
    ("name", "size"),
    [
        ("Cell0_SOI_Full_Institution", (11470, 4900)),
        ("Cell0_SOI_Half_Institution", (5500, 4900)),
    ],
)
def test_die_outlines_match_template(name: str, size: tuple[float, float]) -> None:
    """The full and half design-area outlines equal the template ones."""
    component = getattr(si220.cells, name)()
    for layer in LAYERS:
        ours = _region(component, layer)
        theirs = template_region(name, layer)
        assert (ours ^ theirs).is_empty(), layer
    assert (component.xsize, component.ysize) == size


def test_die_defaults_to_the_full_design_area() -> None:
    """``die`` is the full design area, ``die_half`` the half one."""
    assert si220.cells.die().dbbox().width() == 11470
    assert si220.cells.die().dbbox().height() == 4900
    assert si220.cells.die_half().dbbox().width() == 5500


def test_packaging_template_matches() -> None:
    """Gratings, pads and outline match exactly; the loopback arcs to within 1 nm.

    The template's loopback arcs are discretised differently (370 and 374 segments
    on the two edges of a 180 degree arc), so their XOR is a sub-nanometre sliver.
    """
    name = "Cell0_SOI220_cband_TE_Full_Packaging_Template"
    component = si220.cells.Cell0_SOI220_cband_TE_Full_Packaging_Template()
    for layer in LAYERS:
        xor = _region(component, layer) ^ template_region(name, layer)
        if layer == (3, 0):
            assert xor.sized(-1).is_empty()
            assert xor.area() * 1e-6 < 0.5
        else:
            assert xor.is_empty(), layer


def test_packaging_template_ports() -> None:
    """14 open gratings per side at x = +-4923 um and 31 pads per side."""
    component = si220.cells.die_with_pads()
    optical = sorted(
        (p.center for p in component.ports.filter(port_type="optical")),
        key=lambda c: (c[0], c[1]),
    )
    ys = [-1625 + 250 * i for i in range(14)]
    assert optical == [(-4923, y) for y in ys] + [(4923, y) for y in ys]
    electrical = list(component.ports.filter(port_type="electrical"))
    assert len(electrical) == 62
    xs = sorted({p.center[0] for p in electrical})
    assert xs == [-4615 + 300 * i for i in range(31)]


######################
# Cornerstone rules (Table 6 and the v3.1 pre-DRC deck), in pure klayout.db
######################

# klayout's default metrics is euclidean, like the deck's.


def _width(region: kdb.Region, d: int) -> kdb.EdgePairs:
    return region.width_check(d - 1, ignore_angle=45)


def _space(region: kdb.Region, d: int) -> kdb.EdgePairs:
    return region.space_check(d - 1, ignore_angle=45)


def drc(component: gf.Component) -> dict[str, int]:
    """Count violations of the deck rules on the layers cspdk si220 draws."""
    wg = _region(component, (3, 0)).merged()
    slab = _region(component, (5, 0)).merged()
    gra = _region(component, (6, 0)).merged()
    pad = _region(component, (41, 0)).merged()
    label = _region(component, (100, 0)).merged()
    # Gaps in GDS 6 narrower than 350 nm may be at most 20 um long.
    narrow = gra.space_check(350).polygons()
    long_narrow = narrow.with_area(int(0.25 * 20.001 * 1e6), None, False) & (
        narrow.with_perimeter(int(2 * (0.35 + 20.001) * 1e3), None, False)
    )
    overlap = gra.not_outside(wg) & gra.inside(wg.sized(200))
    counts = {
        "GDS6 width < 200 nm": _width(gra, 200).count(),
        "GDS6 gap < 250 nm": _space(gra, 250).count(),
        "GDS6 gap < 350 nm longer than 20 um": long_narrow.count(),
        "GDS6 overlaps GDS3 by < 200 nm": overlap.count(),
        "GDS3 width < 350 nm": _width(wg, 350).count(),
        "GDS3 gap < 200 nm": _space(wg, 200).count(),
        "GDS5 width < 250 nm": _width(slab, 250).count(),
        "GDS5 gap < 250 nm": _space(slab, 250).count(),
        "GDS41 width < 2 um": _width(pad, 2000).count(),
        "GDS41 gap < 10 um": _space(pad, 10000).count(),
        "GDS100 width < 250 nm": _width(label, 250).count(),
        "GDS100 gap < 250 nm": _space(label, 250).count(),
    }
    return {rule: n for rule, n in counts.items() if n}


@pytest.mark.parametrize(
    "name",
    [
        "grating_coupler_rectangular",
        "grating_coupler_rectangular_rib",
        "grating_coupler_elliptical",
        "SOI220_cband_TE_strip_GC",
        "SOI220_cband_TE_rib_GC",
        "SOI220_oband_TE_strip_GC",
        "SOI220_oband_TE_rib_GC",
        "text_rectangular",
        "text_rectangular_multi_layer",
        "die",
        "die_half",
        "die_with_pads",
    ],
)
def test_cell_passes_cornerstone_rules(name: str) -> None:
    """The default cell passes the deck rules on the layers it draws."""
    assert drc(si220.PDK.get_component(name)) == {}


@pytest.mark.parametrize("cross_section", ["strip_oband", "rib_oband", "rib_cband"])
def test_elliptical_grating_passes_rules_in_every_band(cross_section: str) -> None:
    """The ridge width keeps the elliptical grating within the deck rules."""
    component = si220.cells.grating_coupler_elliptical(cross_section=cross_section)
    assert drc(component) == {}


def test_rules_catch_the_old_geometry() -> None:
    """The checks see the violations the deck reported before the Run-50 update."""
    flush_teeth = si220.cells.grating_coupler_rectangular(grating_overhang=0)
    assert "GDS6 overlaps GDS3 by < 200 nm" in drc(flush_teeth)
    narrow_ridges = si220.cells.grating_coupler_elliptical(grating_line_width=0.315)
    assert "GDS6 gap < 350 nm longer than 20 um" in drc(narrow_ridges)
    pad_text = si220.cells.text_rectangular(layer="PAD")
    assert "GDS41 gap < 10 um" in drc(pad_text)


######################
# Models
######################


@pytest.mark.parametrize(
    ("name", "wl0", "loss", "one_db_width"),
    [
        ("SOI220_cband_TE_strip_GC", 1.565, 6.0, 0.035),
        ("SOI220_cband_TE_rib_GC", 1.565, 6.0, 0.035),
        ("SOI220_oband_TE_strip_GC", 1.315, 5.5, 0.030),
        ("SOI220_oband_TE_rib_GC", 1.315, 5.5, 0.030),
    ],
)
def test_foundry_grating_models(
    name: str, wl0: float, loss: float, one_db_width: float
) -> None:
    """Measured Run-50 data: loss at the centre and the 1 dB bandwidth."""
    assert name in si220.PDK.cells
    model = si220.PDK.models[name]
    assert model is getattr(si220.models, name)
    peak = model(wl=wl0)["o1", "o2"]
    assert -20 * np.log10(abs(peak)) == pytest.approx(loss)
    edge = model(wl=wl0 + one_db_width / 2)["o1", "o2"]
    assert -20 * np.log10(abs(edge)) == pytest.approx(loss + 1, abs=1e-6)
