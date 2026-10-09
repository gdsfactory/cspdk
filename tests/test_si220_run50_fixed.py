"""Run 50 (October 2026) foundry layers and fixed cells of the si220 PDK."""

from __future__ import annotations

import math
import tomllib

import gdsfactory as gf
import jax.numpy as jnp
import klayout.db as kdb
import numpy as np
import pytest
from _run50 import template, template_region

from cspdk.si220 import LAYER, LAYER_VIEWS, PDK, cells, models
from cspdk.si220.config import PATH

YSPLITTERS = [
    "SOI220_cband_TE_strip_1x2_Ysplitter",
    "SOI220_oband_TE_strip_1x2_Ysplitter",
]
GRAPHENE = [
    f"SOI220_cband_TE_strip_SLG_{device}{suffix}"
    for device in (
        "Bolometer",
        "Bolometer_metallic",
        "Bolometer_no_gate",
        "EAM",
        "PTE",
        "PTE_metallic",
    )
    for suffix in ("", "_with_GC")
]
PHOTODIODES = ["SOI220_PD_FCI300B1_Integrated", "SOI220_PD_FCI300B1_Plain"]
LASERS = [
    f"SOI220_{size}_LD_DC1550HP01_Integrated{suffix}"
    for size in ("Full", "Half")
    for suffix in ("", "_or_GC", "_with_GC")
]
RUN50_CELLS = [
    *YSPLITTERS,
    "SOI220_cband_TE_strip_apodized_GC_focusing_v1p0",
    *GRAPHENE,
    *PHOTODIODES,
    *LASERS,
]

# (cell, number of waveguide ports, number of fibre ports, number of pad ports)
PORT_COUNTS = {
    **dict.fromkeys(YSPLITTERS, (3, 0, 0)),
    "SOI220_cband_TE_strip_apodized_GC_focusing_v1p0": (1, 1, 0),
    **{name: (0, 2, 0) if "_with_GC" in name else (2, 0, 0) for name in GRAPHENE},
    "SOI220_PD_FCI300B1_Integrated": (1, 0, 2),
    "SOI220_PD_FCI300B1_Plain": (0, 0, 2),
    **{
        name: (1, 0, 0) if name.endswith("Integrated") else (1, 1, 0) for name in LASERS
    },
}


@pytest.fixture(autouse=True)
def activate_pdk() -> None:
    """Use si220 for every layout built in this module."""
    PDK.activate()


def _template_layers(cell: kdb.Cell) -> set[tuple[int, int]]:
    layout = cell.layout()
    return {
        (layout.get_info(index).layer, layout.get_info(index).datatype)
        for index in layout.layer_indexes()
        if not cell.bbox_per_layer(index).empty()
    }


def _component_region(component: gf.Component, layer: tuple[int, int]) -> kdb.Region:
    return component.get_region(layer).merged()


def _point(x: float, y: float) -> kdb.Point:
    return kdb.DPoint(x, y).to_itype(0.001)


def _marker(x: float, y: float) -> kdb.Region:
    """A 2 nm box around (x, y)."""
    return kdb.Region(kdb.Box(_point(x, y), _point(x, y)).enlarged(1))


def _inside(region: kdb.Region, x: float, y: float) -> bool:
    return any(polygon.inside(_point(x, y)) for polygon in region.each())


################
# Layers
################


def test_every_template_layer_is_named() -> None:
    """Every layer drawn in the Run 50 template has a name and a layer view."""
    used = set()
    for cell in template().each_cell():
        used |= _template_layers(cell)
    named = {(layer.layer, layer.datatype) for layer in LAYER}
    viewed = {tuple(view.layer) for view in LAYER_VIEWS.layer_views.values()}
    assert used - named == set()
    assert used - viewed == set()


@pytest.mark.parametrize(
    ("name", "layer"),
    [
        ("LASER_CONTACT", (43, 2)),
        ("LASER_TRENCH", (47, 2)),
        ("PD_AREA", (140, 1)),
        ("LASER_AREA", (143, 1)),
        ("LASER_WG", (144, 1)),
        ("PD_ACTIVE", (147, 1)),
        ("PD_EXCLUSION", (150, 1)),
        ("LASER_EXCLUSION", (151, 1)),
        ("GRAPHENE1", (161, 2)),
        ("GRAPHENE2", (162, 2)),
        ("GRAPHENE_CONTACT1", (163, 2)),
        ("GRAPHENE_CONTACT2", (164, 2)),
        ("GRAPHENE_PAD1", (165, 2)),
        ("GRAPHENE_PAD2", (166, 2)),
        ("GRAPHENE_EXCLUSION", (169, 1)),
        ("GRA_EBL", (60, 0)),
    ],
)
def test_layer_names(name: str, layer: tuple[int, int]) -> None:
    """The Run 50 integration layers use the guideline's layer numbers."""
    assert tuple(getattr(LAYER, name)) == layer
    assert tuple(LAYER_VIEWS.layer_views[name].layer) == layer


def test_lyp_lists_every_layer_view() -> None:
    """The KLayout layer properties are generated from layers.yaml."""
    lyp = (PATH.klayout / "layers.lyp").read_text()
    for name, view in LAYER_VIEWS.layer_views.items():
        layer, datatype = view.layer
        assert f"<source>{layer}/{datatype}@1</source>" in lyp
        label = f"{name} {layer}/{datatype}" if view.layer_in_name else name
        assert f"<name>{label}</name>" in lyp


################
# Fixed cells
################


def test_run50_cells_are_in_the_pdk() -> None:
    """Each fixed cell is a public PDK cell named exactly as its foundry cell."""
    for name in RUN50_CELLS:
        assert template().cell(name) is not None, name
        assert name in PDK.cells
        assert getattr(cells, name)().name == name


@pytest.mark.parametrize("name", RUN50_CELLS)
def test_cell_matches_template(name: str) -> None:
    """The cell is the template cell: the XOR on every layer is empty."""
    component = getattr(cells, name)()
    assert component.kcl.dbu == template().dbu
    layers = _template_layers(template().cell(name))
    layers |= {
        (info.layer, info.datatype)
        for info in (component.kcl.get_info(i) for i in component.kcl.layer_indexes())
        if not component.bbox(component.kcl.layer(info)).empty()
    }
    for layer in sorted(layers):
        xor = _component_region(component, layer) ^ template_region(name, layer)
        assert xor.is_empty(), (name, layer)


@pytest.mark.parametrize("name", RUN50_CELLS)
def test_port_counts_and_names(name: str) -> None:
    """Waveguide ports, then fibre ports, are o1, o2, ...; pad ports are e1, ...

    Within each kind the ports are numbered clockwise from the west.
    """
    component = getattr(cells, name)()
    optical = [p for p in component.ports if p.port_type == "optical"]
    fiber = [p for p in component.ports if p.port_type == "vertical_te"]
    electrical = [p for p in component.ports if p.port_type == "electrical"]
    assert (len(optical), len(fiber), len(electrical)) == PORT_COUNTS[name]
    assert len(component.ports) == sum(PORT_COUNTS[name])

    def names(ports):
        return [p.name for p in gf.port.sort_ports_clockwise(ports)]

    assert names(optical) + names(fiber) == [
        f"o{i}" for i in range(1, len(optical) + len(fiber) + 1)
    ]
    assert names(electrical) == [f"e{i}" for i in range(1, len(electrical) + 1)]


@pytest.mark.parametrize("name", RUN50_CELLS)
def test_waveguide_ports_sit_on_waveguide_ends(name: str) -> None:
    """Each waveguide port is the centre of a waveguide end of the port's width."""
    component = getattr(cells, name)()
    band = "oband" if "_oband_" in name else "cband"
    waveguide = template_region(name, (3, 0))
    edges = waveguide.edges()
    for port in component.ports:
        if port.port_type != "optical":
            continue
        assert port.info["cross_section"] == f"strip_{band}"
        assert port.width == gf.get_cross_section(f"strip_{band}").width
        assert (port.layer_info.layer, port.layer_info.datatype) == (3, 0)
        x, y = port.center
        angle = math.radians(port.orientation)
        dx, dy = round(math.cos(angle)), round(math.sin(angle))
        # The waveguide end: an edge across the port, centred on it, of its width.
        end = [e.to_dtype(0.001) for e in edges.interacting(_marker(x, y))]
        assert len(end) == 1, (name, port.name, end)
        edge = end[0]
        assert edge.length() == pytest.approx(port.width, abs=1e-6)
        assert ((edge.p1.x + edge.p2.x) / 2, (edge.p1.y + edge.p2.y) / 2) == (
            pytest.approx(x, abs=1e-6),
            pytest.approx(y, abs=1e-6),
        )
        assert edge.dx() * dx + edge.dy() * dy == 0  # across the port direction
        # Silicon just behind the port, none just in front of it.
        assert _inside(waveguide, x - dx * 0.001, y - dy * 0.001), (name, port.name)
        assert not _inside(waveguide, x + dx * 0.001, y + dy * 0.001), (name, port.name)


@pytest.mark.parametrize("name", RUN50_CELLS)
def test_fiber_ports_sit_on_grating_centres(name: str) -> None:
    """A fibre port is at the centre of a grating's teeth, facing away from it."""
    component = getattr(cells, name)()
    teeth = template_region(name, (6, 0))
    gratings = list(teeth.sized(2000).merged().each())
    for port in component.ports:
        if port.port_type != "vertical_te":
            continue
        point = _point(*port.center)
        grating = [g for g in gratings if g.inside(point)]
        assert len(grating) == 1, (name, port.name)
        box = (teeth & kdb.Region(grating[0])).bbox().to_dtype(0.001)
        assert port.center == (
            pytest.approx(box.center().x, abs=0.001),
            pytest.approx(box.center().y, abs=0.001),
        )
        assert (port.layer_info.layer, port.layer_info.datatype) == (6, 0)


@pytest.mark.parametrize("name", PHOTODIODES)
def test_pad_ports_sit_on_pad_edges(name: str) -> None:
    """e1 (cathode) and e2 (anode) span the outer edge of their PAD pad."""
    component = getattr(cells, name)()
    pads = template_region(name, (41, 0))
    edges = pads.edges()
    for port in component.ports.filter(port_type="electrical"):
        assert (port.layer_info.layer, port.layer_info.datatype) == (41, 0)
        x, y = port.center
        edge = [e.to_dtype(0.001) for e in edges.interacting(_marker(x, y))]
        assert len(edge) == 1
        assert edge[0].length() == pytest.approx(port.width)
        angle = math.radians(port.orientation)
        assert _inside(pads, x - math.cos(angle), y - math.sin(angle))
        assert not _inside(pads, x + math.cos(angle), y + math.sin(angle))
    assert component.ports["e1"].center[1] > component.ports["e2"].center[1]


def test_no_model_cells_are_listed_in_pyproject() -> None:
    """GDSFactory+ expects no model only for the cells listed in pyproject.toml."""
    with (PATH.repo / "pyproject.toml").open("rb") as f:
        expected = set(
            tomllib.load(f)["tool"]["gdsfactoryplus"]["pdk"]["cells_no_model_expected"]
        )
    for name in RUN50_CELLS:
        assert (name in PDK.models) != (name in expected), name


################
# Y-splitter models
################


@pytest.mark.parametrize(
    ("name", "wl", "loss_per_unit_dB"),
    [
        ("SOI220_cband_TE_strip_1x2_Ysplitter", 1.55, 4.0123),
        ("SOI220_oband_TE_strip_1x2_Ysplitter", 1.31, 4.5006),
    ],
)
def test_ysplitter_model_matches_measured_loss(
    name: str, wl: float, loss_per_unit_dB: float
) -> None:
    """Each output carries the measured cut-back loss per splitter, flat in wl."""
    model = PDK.models[name]
    assert model is getattr(models, name)
    s = model(wl=jnp.array([wl - 0.05, wl, wl + 0.05]))
    for out in ("o2", "o3"):
        loss = -10 * np.log10(np.abs(np.asarray(s["o1", out])) ** 2)
        assert loss == pytest.approx(np.full(3, loss_per_unit_dB), abs=1e-4)
        assert np.allclose(s["o1", out], s[out, "o1"])
    assert np.allclose(s["o1", "o2"], s["o1", "o3"])
    assert set(model().keys()) == {
        ("o1", "o2"),
        ("o1", "o3"),
        ("o2", "o1"),
        ("o3", "o1"),
    }


@pytest.mark.parametrize("name", YSPLITTERS)
def test_ysplitter_schematic_links_its_model(name: str) -> None:
    """The splitter schematic points GDSFactory+ to the splitter's SAX model."""
    factory = getattr(cells, name)
    component = factory()
    schematic = component.kcl.factories[name].get_schematic()
    (entry,) = schematic.info["models"]
    assert entry["name"] == name
    assert entry["module"] == "cspdk.si220.models"
    assert entry["port_order"] == [p.name for p in component.ports]
