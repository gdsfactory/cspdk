"""Convert sample-project ``.pic.yml`` schematics to ``.gsch`` and check the layout.

Run it with the sample project's own environment, so the flavour PDK and
gdsfactoryplus resolve exactly as GDSFactory+ resolves them::

    uv run --project cspdk--sample-projects/cspdk.si220--public--project \
        python scripts/convert_sample_schematics.py \
        cspdk--sample-projects/cspdk.si220--public--project mymzi mzi2 --verify

The GDSFactory+ converter (``convertPicYamlToGsch``) leaves these gaps, which
make the ``.gsch`` build a different layout from its ``.pic.yml`` or show up
wrong in the editor:

* Bare or wrongly qualified ``model`` ids. A bare ``mzi`` resolves through the
  global kfactory registry to gdsfactory's generic ``mzi``, and every cspdk
  flavour registers a ``straight``, so the last one imported wins. Each model
  is rewritten to the dotted name of the active flavour PDK's own cell, which
  GDSFactory+ resolves first, and each instance gets the editor's device type
  for it (``mzi-2x2``, ``bend``, ...).
* No layout placement. Without ``irl_*`` fields GDSFactory+ places each
  instance at its schematic grid position, so butt connections fail and get
  dropped. The ``irl_*`` and cell-bbox fields are taken from the ``.pic.yml``
  layout, so every instance, and every element of an instance array, lands
  where the ``.pic.yml`` puts it.
* No ports. The schematic editor only uses that placement when the instance
  also lists its ports, and draws pins from that list; each instance gets its
  cell's ports in the editor's own format.
* No route paths. Port ranges (``o3-4``) stay unexpanded and routes have no
  path for the editor to draw. Each route the ``.pic.yml`` made becomes an
  editor route along its routed path, between the ports gdsfactory actually
  routed, with its cross-section, bend type and S-bends.

``--verify`` builds each ``.gsch`` with GDSFactory+ (``buildNyancirGds``) and
XORs it against the ``.pic.yml`` layout on every layer.

GDSFactory+ routes along the stored path itself, so what its router does
differently stays different: it adds no auto-tapers and draws metal corners
as wire corners. A* routes have no stored path and are routed again, without
their bundle settings. ``--verify`` reports those samples as different.

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
import time
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


def manhattan_path(
    points: list[kdb.DPoint],
) -> tuple[list[kdb.DPoint], dict[str, dict[str, Any]]]:
    """A routed backbone as editor corners, with S-bends for its diagonal steps.

    gdsfactory routes a small lateral offset as one ``bend_s``, which shows up
    as a diagonal step in the backbone. The editor's format draws it as a Z
    (two corners at the step's midpoint) with that jog marked as an S-bend.
    """
    path = [points[0]]
    sbends = {}
    for a, b in zip(points, points[1:]):
        if a.x != b.x and a.y != b.y:
            middle = (a.x + b.x) / 2
            # Segment index of the Z's middle jog in the editor's numbering.
            sbends[str(len(path))] = {"factory": "bend_s"}
            path += [kdb.DPoint(middle, a.y), kdb.DPoint(middle, b.y)]
        path.append(b)
    return path, sbends


def routes_as_polylines(
    component: Any,
    bundles: dict[str, Any],
    placed: dict[str, tuple[Any, kdb.DCplxTrans]],
    nets: dict[tuple[str, str], str],
    pdk: Any,
) -> dict[str, dict[str, Any]]:
    """Each route the .pic.yml made, as an editor route.

    Like routes the editor saves: the routed backbone as ``vertices`` (y down),
    with the cross-section's name, width, layer and bend radius, and no routing
    strategy, so GDSFactory+ routes along those corners. The endpoints come
    from the backbone, so ``o1-4`` port ranges and ``sort_ports`` resolve to
    the pairs gdsfactory routed.

    A* strategies return no backbone. Those routes keep their strategy (and
    get routed again by GDSFactory+), with endpoints from the route's name.
    """
    pins = [
        (name, port, trans * kdb.DPoint(*port.center))
        for name, (cell, trans) in placed.items()
        for port in cell.ports
    ]
    ports = {(name, port.name): port for name, port, _ in pins}

    def pin_at(point: kdb.DPoint) -> tuple[str, str]:
        name, port, _ = min(pins, key=lambda pin: pin[2].distance(point))
        return name, port.name

    dbu = component.kcl.dbu
    polylines = {}
    routes = component.routes if isinstance(component.routes, dict) else {}
    for index, (route_name, route) in enumerate(routes.items()):
        bundle_name = max(
            (b for b in bundles if route_name.startswith(f"{b}-")), key=len
        )
        bundle = bundles[bundle_name]
        bundle_cross_section = (bundle.get("settings") or {}).get("cross_section")
        polyline: dict[str, Any] = {"type": "polyline", "name": f"route_{index}"}
        if backbone := getattr(route, "backbone", None):
            points = [kdb.DPoint(p.x * dbu, p.y * dbu) for p in backbone]
            start, finish = pin_at(points[0]), pin_at(points[-1])
            path, sbends = manhattan_path(points)
            polyline["vertices"] = [[round(p.x, 6), round(-p.y, 6)] for p in path]
            # Route with the bends the original route used: flavour strategies
            # such as route_bundle_sc bend with bend_circular, the default
            # corner router with bend_euler.
            for inst in route.instances:
                factory = inst.cell.function_name or ""
                if factory.startswith("bend_s"):
                    for sbend in sbends.values():
                        sbend["factory"] = factory
                elif (
                    "bend" in factory
                    and factory != "bend_euler"
                    and factory in pdk.cells
                ):
                    polyline["bend_type"] = factory
            cross_section = bundle_cross_section or ports[start].info.get(
                "cross_section"
            )
            if cross_section in pdk.cross_sections:
                xs = pdk.get_cross_section(cross_section)
                polyline |= {
                    "cross_section": cross_section,
                    "width": xs.width,
                    "layer": getattr(xs.layer, "name", str(xs.layer)),
                }
                if xs.radius:
                    polyline["radius"] = xs.radius
                for sbend in sbends.values():
                    sbend["cross_section"] = cross_section
            if sbends:
                polyline["sbend_segments"] = sbends
        else:
            # from_yaml names each route "<bundle>-<inst>,<port>-<inst>,<port>".
            first, second = route_name.removeprefix(f"{bundle_name}-").split("-")
            start, finish = tuple(first.split(",")), tuple(second.split(","))
            polyline["routing_strategy"] = bundle["routing_strategy"]
            # The A* strategies take a cross-section but no width or radius.
            if bundle_cross_section:
                polyline["cross_section"] = bundle_cross_section
        polyline["terminals"] = {
            "start": {"component_id": start[0], "port_name": start[1]},
            "finish": {"component_id": finish[0], "port_name": finish[1]},
        }
        if net := nets.get(start) or nets.get(finish):
            polyline["net"] = net
        polylines[polyline["name"]] = polyline
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


PORT_NATURE = {"optical": "photonic", "electrical": "electric"}
RESERVED_TYPES = {"wire", "port", "text", "polyline", "group"}


def fix_gsch(
    gsch: Path,
    stems: set[str],
    pdk_name: str,
    pdk: Any,
    component_type: Any,
) -> list[str]:
    """Qualify models, and write .pic.yml placement and routes into one .gsch.

    ``component_type(model, props)`` returns the editor's device type for a
    cell (``mzi-2x2``, ``bend``, ...), or None to keep the converter's.
    """
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
        dev.setdefault("transform", [1, 0, 0, 1, 0, 0])
        device_type = component_type(dev["model"], dev.get("props") or {})
        # GDSFactory+ reports straight_metal as "wire", which the build would
        # then skip as a schematic wire; keep instances off the reserved types.
        if device_type and device_type not in RESERVED_TYPES:
            dev["type"] = device_type
    port_types = {
        (dev.get("name", key), port["name"]): port["port_type"]
        for key, dev in doc.items()
        for port in dev.get("ports") or []
    }
    for dev in doc.values():
        if dev.get("type") == "wire":
            dev.setdefault("variant", "hv")
        elif dev.get("type") == "port" and (attached := dev.get("attached_port")):
            port_type = port_types.get(
                (attached["component_id"], attached["port_name"])
            )
            dev["nature"] = PORT_NATURE.get(port_type, port_type)
    polylines = routes_as_polylines(component, bundles, placed, nets, pdk)
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
    types: dict[str, str | None] = {}

    def component_type(model: str, props: dict[str, Any]) -> str | None:
        """The editor's device type, from the same getBBox call the editor makes."""
        key = json.dumps([model, props], sort_keys=True)
        if key not in types:
            try:
                bbox = server.call(
                    "getBBox", {"name": model, "params": props, "texture_key": None}
                )
                types[key] = bbox.get("component_type")
            except RuntimeError:
                types[key] = None  # project subcircuits and cells aren't indexed
        return types[key]

    failed = False
    try:
        # The factory index fills in the background after the server starts.
        for _ in range(120):
            if server.call("listFactories", {}).get("factories"):
                break
            time.sleep(5)
        for stem in todo:
            try:
                result = server.call(
                    "convertPicYamlToGsch", {"path": str(sources / f"{stem}.pic.yml")}
                )
                gsch = Path(result["output_path"])
                problems = fix_gsch(gsch, stems, pdk_name, pdk, component_type)
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
