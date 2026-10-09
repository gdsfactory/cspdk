"""Si220 cells remade from the Cornerstone MPW Run-50 template, and their models."""

from __future__ import annotations

import warnings

import gdsfactory as gf
import jax.numpy as jnp
import kfactory as kf
import numpy as np
import pytest
from _run50 import template

import cspdk.si220 as si220
from cspdk.si220 import cells
from cspdk.si220.models import foundry

# Cells of the Run-50 template remade by code, each by a cell of the same name.
FOUNDRY_CELLS = [
    *(
        f"SOI220_{band}_TE_{kind}_{suffix}"
        for band in ("cband", "oband")
        for kind in ("strip", "rib")
        for suffix in ("1x2_MMI", "2x2_MMI", "crossing")
    ),
    "SOI220_cband_TE_strip_90deg_bend",
    "SOI220_oband_TE_strip_90deg_bend",
    "SOI220_cband_TE_rib_90deg_bend",
    "SOI220_cband_TE_strip_MM2SM_taper",
    "SOI220_cband_TE_rib_MM2SM_taper",
    "SOI220_cband_TE_rib2strip_taper",
    "Heater",
]

# The parametric cells at the settings each foundry cell uses.
PARAMETRIC = [
    *(
        pytest.param(
            factory,
            {"cross_section": f"{kind}_{band}"},
            f"SOI220_{band}_TE_{kind}_{suffix}",
            id=f"{factory}-{kind}_{band}",
        )
        for band in ("cband", "oband")
        for kind in ("strip", "rib")
        for factory, suffix in (
            ("mmi1x2", "1x2_MMI"),
            ("mmi2x2", "2x2_MMI"),
            ("crossing", "crossing"),
        )
    ),
    pytest.param(
        "bend_circular",
        {"radius": 5, "npoints": 31, "allow_min_radius_violation": True},
        "SOI220_cband_TE_strip_90deg_bend",
        id="bend_circular-strip_cband",
    ),
    pytest.param(
        "bend_circular",
        {"radius": 25, "npoints": 156, "cross_section": "rib_cband"},
        "SOI220_cband_TE_rib_90deg_bend",
        id="bend_circular-rib_cband",
    ),
    pytest.param("taper_mm2sm", {}, "SOI220_cband_TE_strip_MM2SM_taper", id="mm2sm"),
    pytest.param(
        "taper_mm2sm",
        {"cross_section": "rib_cband"},
        "SOI220_cband_TE_rib_MM2SM_taper",
        id="mm2sm-rib",
    ),
    pytest.param(
        "taper_rib_to_strip", {}, "SOI220_cband_TE_rib2strip_taper", id="rib2strip"
    ),
    pytest.param("heater", {}, "Heater", id="heater"),
]


@pytest.fixture(autouse=True)
def _activate() -> None:
    si220.PDK.activate()


def _template_regions(cell: str) -> dict[tuple[int, int], kf.kdb.Region]:
    layout = template()
    top = layout.cell(cell)
    regions = {}
    for index in layout.layer_indexes():
        region = kf.kdb.Region(top.begin_shapes_rec(index))
        if not region.is_empty():
            info = layout.get_info(index)
            regions[info.layer, info.datatype] = region.merged()
    return regions


def _regions(component: gf.Component) -> dict[tuple[int, int], kf.kdb.Region]:
    regions = {}
    for index in component.kcl.layer_indexes():
        region = kf.kdb.Region(component.begin_shapes_rec(index))
        if not region.is_empty():
            info = component.kcl.get_info(index)
            regions[info.layer, info.datatype] = region.merged()
    return regions


def _xor(component: gf.Component, cell: str) -> dict[tuple[int, int], int]:
    """Number of XOR polygons per layer between ``component`` and a template cell."""
    ours, theirs = _regions(component), _template_regions(cell)
    empty = kf.kdb.Region()
    return {
        layer: (ours.get(layer, empty) ^ theirs.get(layer, empty)).count()
        for layer in ours.keys() | theirs.keys()
    }


def _same(a: gf.Component, b: gf.Component) -> bool:
    """Whether two components have the same geometry on every layer."""
    ours, theirs = _regions(a), _regions(b)
    return ours.keys() == theirs.keys() and all(
        (ours[layer] ^ theirs[layer]).is_empty() for layer in ours
    )


@pytest.mark.parametrize("name", FOUNDRY_CELLS)
def test_foundry_cell_matches_template(name: str) -> None:
    """Each foundry-named cell is the template cell of that name, XOR == 0."""
    xor = _xor(si220.PDK.cells[name](), name)
    assert not any(xor.values()), xor


@pytest.mark.parametrize(("factory", "settings", "template"), PARAMETRIC)
def test_parametric_cell_matches_template(
    factory: str, settings: dict, template: str
) -> None:
    """The parametric cells reproduce the template cells, XOR == 0."""
    xor = _xor(gf.get_component(factory, **settings), template)
    assert not any(xor.values()), xor


@pytest.mark.parametrize(
    ("name", "settings"),
    [
        ("SOI220_cband_TE_strip_MZI", {"cross_section": "strip_cband"}),
        ("SOI220_cband_TE_rib_MZI", {"cross_section": "rib_cband", "length_x": 649.5}),
    ],
)
def test_foundry_mzi_matches_template_interferometer(name: str, settings: dict) -> None:
    """The foundry MZI is the template MZI without its grating couplers and routes.

    The template also holds three grating couplers, routed to the MMIs inside the
    loop of the arms. Outside the box between the MMIs' inner ports the geometry
    matches, up to 1 nm slivers where the template's arcs meet straights, except
    that the slab of a rib straight runs 5 um past its ends (a bbox slab), beside
    the bends, in slivers under 0.4 um wide.
    """
    component = si220.PDK.cells[name]()
    assert _same(component, gf.get_component("mzi_heater_loop", **settings))
    ours, theirs = _regions(component), _template_regions(name)
    for layer in ((39, 0), (41, 0)):
        assert (ours[layer] ^ theirs[layer]).is_empty(), layer

    x_in = component.ports["o1"].x
    x_out = component.ports["o2"].x
    gc_region = kf.kdb.Region(
        kf.kdb.DBox(x_in, -125, x_out, 75).to_itype(component.kcl.dbu)
    )
    for layer in ours:
        if layer in ((39, 0), (41, 0)):
            continue
        extra = ours[layer] - theirs[layer]
        tolerance = 200 if layer == (5, 0) else 1
        assert extra.sized(-tolerance).is_empty(), layer
        rest = (theirs[layer] - ours[layer]) - gc_region
        assert rest.sized(-1).is_empty(), layer
    # The template's remaining shapes: the grating couplers, inside that box.
    assert theirs[6, 0].inside(gc_region).count() == theirs[6, 0].count()


def test_mzi_arms_and_heater() -> None:
    """Key dimensions of the foundry MZI: 160 um arm difference, heater on top."""
    component = cells.mzi_heater_loop()
    assert component.info["delta_length"] == 160.0
    # The filament is 2 um wide, centred on the upper arm at y = 96.57.
    filament = _regions(component)[39, 0].bbox()
    assert filament.bottom * component.kcl.dbu == pytest.approx(96.57 - 1)
    assert component.ports["e1"].port_type == "electrical"


def test_crossing_rib_is_deprecated() -> None:
    """crossing_rib warns and returns the rib crossing."""
    gf.clear_cache()
    with pytest.warns(DeprecationWarning, match="crossing_rib is deprecated"):
        old = cells.crossing_rib(cross_section="rib_oband")
    new = cells.crossing(cross_section="rib_oband")
    assert _same(old, new)
    assert [(p.name, p.center, p.orientation, p.width) for p in old.ports] == [
        (p.name, p.center, p.orientation, p.width) for p in new.ports
    ]


def test_repo_cells_raise_no_deprecation_warning() -> None:
    """Internal cells do not go through deprecated names."""
    gf.clear_cache()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        for name in ("crossing", "mzi_lattice", "mzi_heater_loop", "heater"):
            si220.PDK.cells[name]()


@pytest.mark.parametrize(
    ("cross_section", "span"),
    [
        ("strip_cband", 9.24),
        ("strip_oband", 8.47),
        ("rib_cband", 19.24),
        ("rib_oband", 19.24),
    ],
)
def test_crossing_ports_compatible(cross_section: str, span: float) -> None:
    """Ports keep the old names, order and orientations, at the arm ends.

    The strip crossings keep the old port spacing (the old fixed cells were these
    foundry crossings, placed off the origin); rib ports now sit on the arm ends
    of the 19.24 um rib crossing instead of 4.62 um from its centre.
    """
    component = cells.crossing(cross_section=cross_section)
    width = gf.get_cross_section(cross_section).width
    ports = {p.name: p for p in component.ports}
    assert list(ports) == ["o1", "o2", "o3", "o4"]
    assert [ports[n].orientation for n in ports] == [180, 90, 0, 270]
    assert all(p.width == pytest.approx(width) for p in ports.values())
    assert ports["o3"].x - ports["o1"].x == pytest.approx(span)
    assert ports["o2"].y - ports["o4"].y == pytest.approx(span)
    assert ports["o1"].y == ports["o3"].y == ports["o2"].x == ports["o4"].x == 0


def test_heater_ports() -> None:
    """The heater has an electrical port on top of each pad (the old one had none)."""
    component = cells.heater()
    ports = {p.name: p for p in component.ports}
    assert set(ports) == {"e1", "e2"}
    pads = _regions(component)[41, 0]
    for port in ports.values():
        assert port.port_type == "electrical"
        assert port.orientation == 90
        assert port.width == pytest.approx(100)
        assert port.center[1] == pytest.approx(181)
        point = kf.kdb.DPoint(port.center[0], port.center[1] - 1)
        assert pads.interacting(
            kf.kdb.Region(kf.kdb.DBox(point, point).enlarged(0.01).to_itype(0.001))
        ).count()


def test_heater_mzi_pads() -> None:
    """The heater draws the MZI's larger pads straight on the leads."""
    component = cells.heater(pad_size=(200, 100), pad_pitch=250, pad_taper_length=0)
    assert component.dbbox().left == pytest.approx(-125)
    assert component.dbbox().top == pytest.approx(151)


@pytest.mark.parametrize("name", ["trans_rib10", "trans_rib20", "trans_rib50"])
def test_trans_rib_slab_tip_meets_minimum_feature_size(name: str) -> None:
    """The slab tip is as wide as the core, above the 250 nm minimum (GDS5)."""
    component = si220.PDK.cells[name]()
    slab = _regions(component)[5, 0]
    # As the Cornerstone deck checks it: 45 degree angle limit.
    assert slab.width_check(250, False, kf.kdb.Metrics.Euclidian, 45).is_empty()


def test_mmi_dimensions() -> None:
    """Run-50 MMI lengths and port offsets (the old ones were 31.0/40.0 um long)."""
    assert cells.mmi1x2().ports["o2"].x == pytest.approx(51.8)
    assert cells.mmi1x2().ports["o2"].y == pytest.approx(1.57)
    oband = cells.mmi1x2(cross_section="strip_oband")
    assert oband.ports["o2"].x == pytest.approx(60.1)
    assert oband.ports["o2"].y == pytest.approx(1.525)
    assert cells.mmi2x2().ports["o3"].y == pytest.approx(1.015)
    rib = cells.mmi1x2(cross_section="rib_cband")
    assert rib.ports["o1"].x == pytest.approx(-25)


################
# Models
################


def _db(value) -> float:
    return float(-20 * np.log10(np.abs(np.asarray(value)).ravel()[0]))


@pytest.mark.parametrize(
    ("name", "wl", "loss"),
    [
        ("SOI220_cband_TE_strip_crossing", 1.55, 0.1195),
        ("SOI220_cband_TE_rib_crossing", 1.55, 0.12467),
        ("SOI220_oband_TE_strip_crossing", 1.31, 0.24359),
        ("SOI220_oband_TE_rib_crossing", 1.31, 0.2123),
    ],
)
def test_crossing_models_use_measured_loss(name: str, wl: float, loss: float) -> None:
    """Crossing models carry the measured insertion loss and no crosstalk."""
    model = si220.PDK.models[name]
    s = model(wl=jnp.array([wl]))
    assert _db(s["o1", "o3"]) == pytest.approx(loss, abs=1e-6)
    assert ("o1", "o2") not in s
    generic = si220.PDK.models["crossing"](
        wl=jnp.array([wl]), cross_section=model.__kwdefaults__["cross_section"]
    )
    assert np.allclose(generic["o1", "o3"], s["o1", "o3"])


@pytest.mark.parametrize(
    ("name", "wl", "measured"),
    [
        ("SOI220_cband_TE_strip_1x2_MMI", 1.55, 3.3056),
        ("SOI220_cband_TE_rib_1x2_MMI", 1.55, 3.0621),
        ("SOI220_oband_TE_strip_1x2_MMI", 1.31, 3.2885),
    ],
)
def test_mmi1x2_models_use_measured_loss(name: str, wl: float, measured: float) -> None:
    """At the band centre, each output is the measured per-MMI insertion loss."""
    s = si220.PDK.models[name](wl=jnp.array([wl]))
    assert _db(s["o1", "o2"]) == pytest.approx(measured, abs=1e-4)


def test_mmi2x2_model_uses_measured_bar_and_cross_loss() -> None:
    """C-band strip 2x2 MMI: bar 3.3673 dB, cross 3.3073 dB (p. 16)."""
    s = si220.PDK.models["SOI220_cband_TE_strip_2x2_MMI"](wl=jnp.array([1.55]))
    assert _db(s["o1", "o4"]) == pytest.approx(3.3673, abs=1e-4)
    assert _db(s["o1", "o3"]) == pytest.approx(3.3073, abs=1e-4)


def test_mmi_loss_can_be_overridden() -> None:
    """``loss_dB`` still sets the excess loss explicitly."""
    s = si220.PDK.models["mmi1x2"](wl=jnp.array([1.55]), loss_dB=1.0)
    assert _db(s["o1", "o2"]) == pytest.approx(1.0 + 10 * np.log10(2), abs=1e-4)


@pytest.mark.parametrize(
    ("name", "loss"),
    [
        ("SOI220_cband_TE_strip_90deg_bend", 0.013603),
        ("SOI220_oband_TE_strip_90deg_bend", 0.010597),
        ("SOI220_cband_TE_rib_90deg_bend", 0.030998),
        ("SOI220_cband_TE_strip_MM2SM_taper", 0.11019),
        ("SOI220_cband_TE_rib_MM2SM_taper", 0.13498),
        ("SOI220_cband_TE_rib2strip_taper", 0.068069),
    ],
)
def test_two_port_models_use_measured_loss(name: str, loss: float) -> None:
    """Bends and tapers lose the measured amount (their waveguide is lossless here)."""
    model = si220.PDK.models[name]
    wl = model.__kwdefaults__["wl"]
    assert _db(model(wl=jnp.array([wl]))["o1", "o2"]) == pytest.approx(loss, abs=1e-6)


def test_rib_to_strip_loss_follows_taper_length() -> None:
    """The rib-to-strip loss is interpolated in the measured taper lengths."""
    model = si220.PDK.models["taper_rib_to_strip"]
    for length, loss in foundry.RIB_TO_STRIP_LOSS_DB:
        s = model(wl=jnp.array([1.55]), length=length)
        assert _db(s["o1", "o2"]) == pytest.approx(loss, abs=1e-6)


def test_mzi_model_interferes_and_tunes() -> None:
    """The MZI model conserves power at most, has fringes, and the heater shifts them."""
    model = si220.PDK.models["SOI220_cband_TE_strip_MZI"]
    wl = jnp.linspace(1.54, 1.56, 401)
    s = model(wl=wl)
    total = np.abs(s["o1", "o2"]) ** 2 + np.abs(s["o1", "o3"]) ** 2
    assert np.all(total <= 1)
    lower = np.abs(np.asarray(s["o1", "o2"])) ** 2
    assert lower.max() - lower.min() > 0.5
    tuned = np.abs(np.asarray(model(wl=wl, voltage=0.5)["o1", "o2"])) ** 2
    assert not np.allclose(lower, tuned)


@pytest.mark.parametrize(
    "name", ["SOI220_cband_TE_strip_MZI", "SOI220_cband_TE_rib_MZI"]
)
def test_mzi_model_matches_its_netlist(name: str) -> None:
    """The flat MZI model equals a SAX simulation of the cell's netlist."""
    import jax
    import sax

    netlist = si220.PDK.cells[name]().get_netlist()
    circuit, _ = sax.circuit(netlist, models=si220.PDK.models)
    wl = jnp.linspace(1.545, 1.555, 11)
    simulated = jax.block_until_ready(circuit(wl=wl))
    flat = si220.PDK.models[name](wl=wl)
    for port in ("o2", "o3"):
        assert np.allclose(simulated["o1", port], flat["o1", port], atol=1e-6)
