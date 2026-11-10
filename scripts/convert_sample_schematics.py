"""Convert sample-project ``.pic.yml`` schematics to ``.gsch`` and check the layout.

Run it with the sample project's own environment, so the flavour PDK and
gdsfactoryplus resolve exactly as GDSFactory+ resolves them::

    uv run --project cspdk--sample-projects/cspdk.si220--public--project \
        python scripts/convert_sample_schematics.py \
        cspdk--sample-projects/cspdk.si220--public--project mymzi mzi2 --verify

The GDSFactory+ converter (``convertPicYamlToGsch``) leaves two gaps that make
the ``.gsch`` build a different layout from its ``.pic.yml``:

* Bare or wrongly qualified ``model`` ids. A bare ``mzi`` resolves through the
  global kfactory registry to gdsfactory's generic ``mzi``, and every cspdk
  flavour registers a ``straight``, so the last one imported wins. Each model
  is rewritten to the dotted name of the active flavour PDK's own cell, which
  GDSFactory+ resolves first.
* No layout placement. Without ``irl_*`` fields GDSFactory+ places each
  instance at its schematic grid position, so butt connections fail and get
  dropped. The ``irl_*`` and cell-bbox fields are taken from the ``.pic.yml``
  layout, so every instance, and every element of an instance array, lands
  where the ``.pic.yml`` puts it.
* No ports. The schematic editor only uses that placement when the instance
  also lists its ports, and draws pins from that list; each instance gets its
  cell's ports in the editor's own format.
* Unresolved routes. Port ranges (``o3-4``) stay unexpanded. Routes are
  rewritten as one polyline per route the ``.pic.yml`` made, between the
  ports gdsfactory actually routed.

``--verify`` builds each ``.gsch`` with GDSFactory+ (``buildNyancirGds``) and
XORs it against the ``.pic.yml`` layout on every layer.

Route settings other than cross-section, width, radius, bend type and
waypoints (e.g. ``sort_ports``, ``start_straight_length``, bundle spacing)
don't survive the conversion; ``--verify`` reports those samples as different.

The ``gfp`` binary ships inside the GDSFactory+ VS Code extension; set
``GFP_BIN`` if it isn't found automatically.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pkgutil
import socket
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import gdsfactory as gf
import klayout.db as kdb
import yaml

PACKAGE = "mycspdk"


def find_gfp() -> Path:
    """Locate the gfp binary from GFP_BIN or the newest installed extension."""
    if env := os.environ.get("GFP_BIN"):
        return Path(env)
    roots = [
        Path.home() / ".vscode-server/extensions",
        Path.home() / ".vscode/extensions",
    ]
    found = sorted(
        (
            p
            for root in roots
            if root.is_dir()
            for p in root.glob("gdsfactory.gdsfactoryplus-*/bin/gfp")
        ),
        key=lambda p: p.stat().st_mtime,
    )
    if not found:
        raise SystemExit("gfp binary not found; set GFP_BIN")
    return found[-1]


class GfpServer:
    """`gfp serve` over stdio, framed like the VS Code extension's RPC client."""

    def __init__(self, gfp: Path, project: Path) -> None:
        """Start `gfp serve` for ``project``."""
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.proc = subprocess.Popen(  # noqa: S603
            [
                gfp,
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--project-root",
                project,
            ],
            cwd=project,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self.next_id = 0

    def _read(self) -> dict[str, Any]:
        length = None
        while (line := self.proc.stdout.readline().decode().strip()) or length is None:
            if not line and self.proc.poll() is not None:
                raise RuntimeError("gfp serve exited")
            name, _, value = line.partition(":")
            if name.strip().lower() == "content-length":
                length = int(value)
        return json.loads(self.proc.stdout.read(length))

    def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send one request and return its result; raise on an RPC error."""
        self.next_id += 1
        body = json.dumps(
            {
                "jsonrpc": "2.0",
                "type": "request",
                "id": self.next_id,
                "method": method,
                "params": params,
            }
        ).encode()
        self.proc.stdin.write(b"Content-Length: %d\r\n\r\n%s" % (len(body), body))
        self.proc.stdin.flush()
        while (msg := self._read()).get("id") != self.next_id:
            pass
        if "error" in msg:
            raise RuntimeError(
                f"{method}: {msg['error'].get('message') or msg['error'].get('msg')}"
            )
        return msg["result"]

    def close(self) -> None:
        """Stop the server."""
        self.proc.terminate()
        self.proc.wait(10)


def activate_pdk(project: Path) -> tuple[str, Any]:
    """Activate the project's flavour PDK with its .pic.yml cells.

    A ``.pic.yml`` may also use a Python cell from the project package (e.g.
    ``ports8`` from ``mycspdk/ports8.py``), as GDSFactory+ allows; those are
    imported and registered by name.
    """
    pdk_name = tomllib.loads((project / "pyproject.toml").read_text())["tool"][
        "gdsfactoryplus"
    ]["pdk"]["name"]
    flavour = importlib.import_module(pdk_name)
    for info in pkgutil.walk_packages(flavour.__path__, prefix=f"{pdk_name}."):
        importlib.import_module(info.name)
    pdk = flavour.PDK
    pdk.activate()
    sys.path.insert(0, str(project))
    for netlist in (project / PACKAGE).glob("*.pic.yml"):
        for inst in (
            yaml.safe_load(netlist.read_text()).get("instances") or {}
        ).values():
            name = inst["component"]
            if name not in pdk.cells and (project / PACKAGE / f"{name}.py").is_file():
                pdk.cells[name] = getattr(
                    importlib.import_module(f"{PACKAGE}.{name}"), name
                )
    pdk.register_cells_yaml(dirpath=project / PACKAGE, update=True)
    return pdk_name, pdk


def qualified_model(pdk_name: str, pdk: Any, name: str) -> str | None:
    """Dotted name of the module attribute that *is* the PDK cell ``name``."""
    cell = pdk.cells.get(name)
    if not cell:
        return None
    modules = [
        m
        for m, mod in list(sys.modules.items())
        if (m == pdk_name or m.startswith((f"{pdk_name}.", f"{PACKAGE}.")))
        and getattr(mod, name, None) is cell
    ]
    if not modules:
        return None
    # The defining module is the deepest one; packages only re-export.
    return f"{max(modules, key=lambda m: m.count('.'))}.{name}"


def placements(
    component: Any, names: set[str]
) -> dict[str, tuple[Any, kdb.DCplxTrans]]:
    """Cell and transform of each named instance; array elements as ``name<col.row>``.

    Only the .pic.yml's own instances count: the component also holds the
    straights, bends and tapers its routes added.
    """
    placed = {}
    for inst in component.insts:
        if inst.name not in names:
            continue
        trans = inst.dcplx_trans
        array = inst.instance
        if not array.is_regular_array():
            placed[inst.name] = (inst.cell, trans)
            continue
        for col in range(array.na):
            for row in range(array.nb):
                shift = kdb.DCplxTrans(array.da * col + array.db * row)
                placed[f"{inst.name}<{col}.{row}>"] = (inst.cell, shift * trans)
    return placed


def routes_as_polylines(
    component: Any,
    bundles: dict[str, Any],
    placed: dict[str, tuple[Any, kdb.DCplxTrans]],
    nets: dict[tuple[str, str], str],
) -> dict[str, dict[str, Any]] | None:
    """One polyline per route the .pic.yml actually made, between its real ports.

    The endpoints come from each route's backbone, so ``o1-4`` port ranges and
    ``sort_ports`` resolve to the pairs gdsfactory routed. Returns None when a
    routing strategy doesn't return its backbone (the A* strategies).

    Pinning the backbone as ``vertices`` doesn't reproduce the layout: with
    ``route_bundle`` waypoints, or doroutes' corner router, the bends differ
    from the original route.
    """
    pins = [
        (name, port.name, trans * kdb.DPoint(*port.center))
        for name, (cell, trans) in placed.items()
        for port in cell.ports
    ]

    def pin_at(point: kdb.DPoint) -> tuple[str, str]:
        name, port, _ = min(pins, key=lambda pin: pin[2].distance(point))
        return name, port

    dbu = component.kcl.dbu
    polylines = {}
    routes = component.routes if isinstance(component.routes, dict) else {}
    for index, (route_name, route) in enumerate(routes.items()):
        if not getattr(route, "backbone", None):
            return None
        bundle = bundles[
            max((b for b in bundles if route_name.startswith(f"{b}-")), key=len)
        ]
        ends = route.backbone[0], route.backbone[-1]
        start, finish = (pin_at(kdb.DPoint(p.x * dbu, p.y * dbu)) for p in ends)
        key = f"route_{index}"
        polyline = {
            "type": "polyline",
            "name": key,
            "routing_strategy": bundle.get("routing_strategy", "route_bundle"),
            "terminals": {
                "start": {"component_id": start[0], "port_name": start[1]},
                "finish": {"component_id": finish[0], "port_name": finish[1]},
            },
        }
        if cross_section := (bundle.get("settings") or {}).get("cross_section"):
            polyline["cross_section"] = cross_section
        if net := nets.get(start) or nets.get(finish):
            polyline["net"] = net
        polylines[key] = polyline
    return polylines


def device_ports(cell: Any, pdk: Any) -> list[dict[str, Any]]:
    """The pins the schematic editor draws for an instance of ``cell``.

    Without a ``ports`` list the editor ignores the stored placement and draws
    a default-size box with no pins. Same convention as the editor writes
    (gdsfactoryplus UI): offsets from the cell's bbox centre in the cell's own
    frame with y pointing down, and GDS angles.
    """
    centre = cell.dbbox().center()
    records = []
    for port in cell.ports:
        x, y = port.center
        layer = port.layer_info
        record = {
            "name": port.name,
            "irl_rx": round(x - centre.x, 6),
            "irl_ry": round(centre.y - y, 6),
            "orientation": round(port.orientation % 360.0, 6),
            "port_type": port.port_type,
            "width": round(port.width, 6),
            "layer": [layer.layer, layer.datatype],
            "layer_name": getattr(port.layer, "name", str(port.layer)),
            # The editor's own fallback bend radius.
            "bendRadii": [5],
        }
        # Routed or modified cross-sections get derived names the PDK doesn't
        # register (e.g. ``metal_routing_5d1ef7b9``); the editor can't look those up.
        if (cross_section := port.info.get("cross_section")) in pdk.cross_sections:
            record["cross_section"] = cross_section
            record["bendRadii"] = [pdk.get_cross_section(cross_section).radius or 5]
        records.append(record)
    return records


def fix_gsch(gsch: Path, stems: set[str], pdk_name: str, pdk: Any) -> list[str]:
    """Qualify models, and write .pic.yml placement and routes into one .gsch."""
    netlist = gsch.with_name(f"{gsch.stem}.pic.yml")
    component = gf.read.from_yaml(netlist)
    source = yaml.safe_load(netlist.read_text())
    bundles = source.get("routes") or {}
    placed = placements(component, set(source["instances"]))
    doc = json.loads(gsch.read_text())
    problems = []
    nets = {}
    for key, dev in doc.items():
        if not dev.get("model"):
            continue
        short = dev["model"].removesuffix(".gsch").rsplit("/", 1)[-1].rsplit(".", 1)[-1]
        if short in stems:
            dev["model"] = f"{PACKAGE}/{short}.gsch"
        elif model := qualified_model(pdk_name, pdk, short):
            dev["model"] = model
        else:
            problems.append(f"unresolved model {dev['model']!r}")
        name = dev.get("name", key)
        if name not in placed:
            problems.append(f"{name!r} has no .pic.yml instance")
            continue
        cell, trans = placed[name]
        # The converter keeps unexpanded port ranges (``o3-4``) as pins.
        ports = {port.name for port in cell.ports}
        dev["nets"] = {p: n for p, n in (dev.get("nets") or {}).items() if p in ports}
        nets |= {(name, port): net for port, net in dev["nets"].items()}
        local = cell.dbbox()
        world = local.transformed(trans)
        # gdsfactoryplus.convert._placement: bbox centre at (irl_x, -irl_y),
        # y-down orientation, x-axis mirror.
        dev.update(
            irl_x=round(world.center().x, 6),
            irl_y=round(-world.center().y, 6),
            irl_orientation=round((360.0 - trans.angle) % 360.0, 6),
            irl_flip=bool(trans.is_mirror()),
            bbox_xmin=round(local.left, 6),
            bbox_ymin=round(local.bottom, 6),
            bbox_width=round(local.width(), 6),
            bbox_height=round(local.height(), 6),
            ports=device_ports(cell, pdk),
        )
    # A* strategies return no backbone; their routes stay as converted.
    if (polylines := routes_as_polylines(component, bundles, placed, nets)) is not None:
        doc = {k: v for k, v in doc.items() if v.get("type") != "polyline"} | polylines
    gsch.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")  # stable diffs
    return problems


def xor_area(a: Path, b: Path) -> dict[str, float]:
    """XOR area (um^2) per layer between the top cells of two GDS files."""
    la, lb = kdb.Layout(), kdb.Layout()
    la.read(str(a))
    lb.read(str(b))
    ta, tb = la.top_cells()[0], lb.top_cells()[0]

    def region(layout: kdb.Layout, top: kdb.Cell, info: tuple[int, int]) -> kdb.Region:
        idx = layout.find_layer(*info)
        if idx is None:
            return kdb.Region()
        return kdb.Region(top.begin_shapes_rec(idx))

    diffs = {}
    for info in {(i.layer, i.datatype) for lay in (la, lb) for i in lay.layer_infos()}:
        ra, rb = region(la, ta, info), region(lb, tb, info)
        area = (ra ^ rb).merged().sized(-1).sized(1).area() * la.dbu**2
        if area > 0.01:
            diffs[f"{info[0]}/{info[1]}"] = round(area, 2)
    return diffs


def main() -> int:
    """Convert, fix and optionally verify the requested schematics."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("project", type=Path, help="sample project directory")
    parser.add_argument(
        "stems", nargs="*", help="schematics to convert (default: every .pic.yml)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="build each .gsch and XOR it against its .pic.yml",
    )
    args = parser.parse_args()
    project = args.project.resolve()
    sources = project / PACKAGE
    stems = {p.name.removesuffix(".pic.yml") for p in sources.glob("*.pic.yml")}
    todo = sorted(args.stems or stems)

    pdk_name, pdk = activate_pdk(project)
    server = GfpServer(find_gfp(), project)
    failed = False
    try:
        for stem in todo:
            try:
                result = server.call(
                    "convertPicYamlToGsch", {"path": str(sources / f"{stem}.pic.yml")}
                )
                gsch = Path(result["output_path"])
                problems = fix_gsch(gsch, stems, pdk_name, pdk)
                if args.verify and not problems:
                    result = server.call("buildNyancirGds", {"nyancir_path": str(gsch)})
                    built = Path(result["output_path"])
                    reference = built.with_name(f"{stem}.pic_yml.gds")
                    gf.get_component(stem).write_gds(reference)
                    if diffs := xor_area(reference, built):
                        problems.append(
                            f"layout differs from .pic.yml (XOR um^2 per layer: {diffs})"
                        )
            except RuntimeError as e:
                problems = [str(e)[:300]]
            failed |= bool(problems)
            print(
                f"{'FAIL' if problems else 'OK  '} {stem}"
                + "".join(f"\n     {p}" for p in problems)
            )
    finally:
        server.close()
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
