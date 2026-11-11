"""Each sample-project ``.gsch`` describes the same circuit as its ``.pic.yml``.

The ``.gsch`` is what the GDSFactory+ schematic editor opens; the ``.pic.yml``
stays as the plain-gdsfactory source the layout was checked against. Both were
converted and layout-verified with ``scripts/convert_sample_schematics.py``.
These checks keep the pair in step without needing GDSFactory+:

* the same instances, components, butt connections, routed ports and ports;
* every ``model`` names the flavour PDK's own cell by its dotted Python name, so
  GDSFactory+ can't pick another flavour's or gdsfactory's same-named cell;
* every instance carries the layout placement taken from the ``.pic.yml``.
"""

from __future__ import annotations

import importlib
import json
import re
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest
import yaml

SAMPLES = Path(__file__).resolve().parents[1] / "cspdk--sample-projects"
GSCH_FILES = sorted(SAMPLES.glob("*/mycspdk/*.gsch"))
# Simulation samples: their sources and test ports have no layout, so there is
# no .pic.yml to compare against.
SIMULATION_ONLY = {"mzi_heater_sweep"}
PLACEMENT_FIELDS = (
    "irl_x",
    "irl_y",
    "irl_orientation",
    "irl_flip",
    "bbox_xmin",
    "bbox_ymin",
    "bbox_width",
    "bbox_height",
)


def _short_name(model: str) -> str:
    """``cspdk.si220.cells.mzis.mzi`` and ``mycspdk/mzi_heater.gsch`` -> bare cell name."""
    return model.removesuffix(".gsch").rsplit("/", 1)[-1].rsplit(".", 1)[-1]


def _devices(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Instances of a .gsch, keyed by instance name."""
    return {dev.get("name", key): dev for key, dev in doc.items() if dev.get("model")}


def _terminal(end: dict[str, str]) -> str:
    """``instance,port`` of a polyline end, in either the Mosaic or Livewire shape."""
    instance = end.get("component_id") or end.get("device")
    return f"{instance},{end.get('port_name') or end.get('port')}"


def _instance_names(name: str, inst: dict[str, Any]) -> list[str]:
    """An instance's name, or ``name<col.row>`` for each element of an array."""
    if not (array := inst.get("array")):
        return [name]
    return [
        f"{name}<{col}.{row}>"
        for col in range(array.get("columns", 1))
        for row in range(array.get("rows", 1))
    ]


def _gsch_circuit(doc: dict[str, Any]) -> dict[str, Any]:
    routed = {
        _terminal(end)
        for dev in doc.values()
        if dev.get("type") == "polyline"
        for end in dev["terminals"].values()
    }
    ports = {
        dev["name"]: f"{attached['component_id']},{attached['port_name']}"
        for dev in doc.values()
        if dev.get("type") == "port" and (attached := dev.get("attached_port"))
    }
    by_net: dict[str, set[str]] = defaultdict(set)
    devices = _devices(doc)
    for name, dev in devices.items():
        for port, net in (dev.get("nets") or {}).items():
            if not net.startswith("port_net_"):
                by_net[net].add(f"{name},{port}")
    pairs = {tuple(sorted(pins)) for pins in by_net.values() if len(pins) > 1}
    return {
        "instances": {name: _short_name(dev["model"]) for name, dev in devices.items()},
        # Routed pairs share a net too; only the rest are butt connections.
        "connections": sorted(pair for pair in pairs if not set(pair) <= routed),
        "routed": sorted(routed),
        "ports": ports,
    }


def _expand(link: str) -> list[str]:
    """``c1,o3-4`` -> ``c1,o3`` and ``c1,o4``; other links unchanged."""
    instance, port = link.split(",")
    if match := re.fullmatch(r"([A-Za-z_]+)(\d+)-(\d+)", port):
        prefix, first, last = match[1], int(match[2]), int(match[3])
        return [f"{instance},{prefix}{i}" for i in range(first, last + 1)]
    return [link]


def _pic_yml_circuit(doc: dict[str, Any]) -> dict[str, Any]:
    connections = doc.get("connections") or {}
    # Routes are compared by the ports they route: with sort_ports or port
    # ranges, gdsfactory decides the pairing.
    routed = {
        port
        for route in (doc.get("routes") or {}).values()
        for pair in (route.get("links") or {}).items()
        for end in pair
        for port in _expand(end)
    }
    return {
        "instances": {
            element: inst["component"]
            for name, inst in doc["instances"].items()
            for element in _instance_names(name, inst)
        },
        "connections": sorted({tuple(sorted(pair)) for pair in connections.items()}),
        "routed": sorted(routed),
        "ports": dict(doc.get("ports") or {}),
    }


def _flavour_pdk(gsch: Path) -> Any:
    pyproject = tomllib.loads((gsch.parents[1] / "pyproject.toml").read_text())
    return importlib.import_module(
        pyproject["tool"]["gdsfactoryplus"]["pdk"]["name"]
    ).PDK


@pytest.fixture(
    params=GSCH_FILES, ids=lambda p: f"{p.parents[1].name.split('--')[0]}/{p.stem}"
)
def gsch(request: pytest.FixtureRequest) -> Path:
    """One sample-project .gsch file."""
    return request.param


def test_samples_have_schematics() -> None:
    """The glob finds the sample schematics, so the checks below aren't vacuous."""
    assert GSCH_FILES, f"no .gsch files under {SAMPLES}"


def test_gsch_matches_pic_yml(gsch: Path) -> None:
    """Instances, components, connections and ports agree with the .pic.yml."""
    if gsch.stem in SIMULATION_ONLY:
        pytest.skip("simulation sample: no .pic.yml")
    pic_yml = gsch.with_name(f"{gsch.stem}.pic.yml")
    assert pic_yml.is_file(), f"{gsch.name} has no {pic_yml.name}"
    assert _gsch_circuit(json.loads(gsch.read_text())) == _pic_yml_circuit(
        yaml.safe_load(pic_yml.read_text())
    )


def test_gsch_models_are_flavour_cells(gsch: Path) -> None:
    """Every model is the flavour's own cell (or a project cell), never a namesake."""
    cells = _flavour_pdk(gsch).cells
    project = gsch.parents[1]
    for name, dev in _devices(json.loads(gsch.read_text())).items():
        model = dev["model"]
        if model.startswith("mycspdk/"):
            assert (project / model).is_file(), f"{name}: {model} does not exist"
        elif model.startswith("mycspdk."):
            module = model.rsplit(".", 1)[0].replace(".", "/")
            assert (project / f"{module}.py").is_file(), (
                f"{name}: {model} does not exist"
            )
        else:
            module, _, attribute = model.rpartition(".")
            cell = getattr(importlib.import_module(module), attribute, None)
            assert any(cell is c for c in cells.values()), (
                f"{name}: {model} is not a cell of this flavour"
            )


def test_gsch_instances_are_placed(gsch: Path) -> None:
    """Each instance keeps the .pic.yml layout placement (irl_* and cell bbox)."""
    for name, dev in _devices(json.loads(gsch.read_text())).items():
        missing = [field for field in PLACEMENT_FIELDS if field not in dev]
        assert not missing, (
            f"{name} lacks {missing}; reconvert with scripts/convert_sample_schematics.py"
        )


def test_gsch_instances_list_their_ports(gsch: Path) -> None:
    """Each instance lists its ports, the editor's only source for drawing pins.

    Without the list the editor ignores the stored placement and draws a
    default-size box with no pins.
    """
    for name, dev in _devices(json.loads(gsch.read_text())).items():
        ports = {port["name"] for port in dev.get("ports") or []}
        assert ports, f"{name} lists no ports"
        assert set(dev.get("nets") or {}) <= ports, (
            f"{name}: nets use ports it doesn't list"
        )


@pytest.mark.parametrize("stem", sorted(SIMULATION_ONLY))
def test_simulation_sample_is_wired(stem: str) -> None:
    """A simulation sample has one optical source, swept sources, and no open pins.

    GDSFactory+'s circulax simulation needs every instance pin on a net, one
    ``emitter`` port for the light, and ``dc`` sweeps as ``start;stop;points``.
    """
    (gsch,) = (path for path in GSCH_FILES if path.stem == stem)
    doc = json.loads(gsch.read_text())
    for name, dev in _devices(doc).items():
        open_pins = {port["name"] for port in dev["ports"]} - set(dev.get("nets") or {})
        assert not open_pins, f"{name} leaves {sorted(open_pins)} unconnected"
    emitters = [
        d
        for d in doc.values()
        if d.get("type") == "port" and d.get("variant") == "emitter"
    ]
    assert len(emitters) == 1
    sources = [d for d in doc.values() if d.get("type") in ("isource", "vsource")]
    assert sources
    for source in sources:
        start, stop, points = source["props"]["dc"].split(";")
        assert float(start) < float(stop) and int(points) > 1
