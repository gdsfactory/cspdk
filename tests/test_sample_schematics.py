"""Each sample-project ``.gsch`` describes the same circuit as its ``.pic.yml``.

The ``.gsch`` is what the GDSFactory+ schematic editor opens; the ``.pic.yml``
stays as the plain-gdsfactory source the layout was checked against. Both were
converted and layout-verified with ``scripts/convert_sample_schematics.py``.
These checks keep the pair in step without needing GDSFactory+:

* the same instances, components, connections (including route links) and ports;
* every ``model`` names the flavour PDK's own cell by its dotted Python name, so
  GDSFactory+ can't pick another flavour's or gdsfactory's same-named cell;
* every instance carries the layout placement taken from the ``.pic.yml``.
"""

from __future__ import annotations

import importlib
import json
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest
import yaml

SAMPLES = Path(__file__).resolve().parents[1] / "cspdk--sample-projects"
GSCH_FILES = sorted(SAMPLES.glob("*/mycspdk/*.gsch"))
PLACEMENT_FIELDS = (
    "irl_x",
    "irl_y",
    "irl_orientation",
    "irl_flip",
    "bbox_xmin",
    "bbox_ymin",
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
    by_net: dict[str, set[str]] = defaultdict(set)
    ports = {}
    for key, dev in doc.items():
        if dev.get("type") == "port" and (attached := dev.get("attached_port")):
            ports[dev["name"]] = f"{attached['component_id']},{attached['port_name']}"
        elif dev.get("type") == "polyline":
            by_net[f"route:{key}"] = {
                _terminal(end) for end in dev["terminals"].values()
            }
    devices = _devices(doc)
    for name, dev in devices.items():
        for port, net in (dev.get("nets") or {}).items():
            if not net.startswith("port_net_"):
                by_net[net].add(f"{name},{port}")
    return {
        "instances": {name: _short_name(dev["model"]) for name, dev in devices.items()},
        # A routed pair appears both as the route's terminals and as a net.
        "connections": sorted(
            {tuple(sorted(pins)) for pins in by_net.values() if len(pins) > 1}
        ),
        "ports": ports,
    }


def _pic_yml_circuit(doc: dict[str, Any]) -> dict[str, Any]:
    links = dict(doc.get("connections") or {})
    for route in (doc.get("routes") or {}).values():
        links.update(route.get("links") or {})
    return {
        "instances": {
            element: inst["component"]
            for name, inst in doc["instances"].items()
            for element in _instance_names(name, inst)
        },
        "connections": sorted({tuple(sorted(pair)) for pair in links.items()}),
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
