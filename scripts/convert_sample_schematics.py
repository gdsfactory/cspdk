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
  layout, so every instance lands where the ``.pic.yml`` puts it.

``--verify`` builds each ``.gsch`` with GDSFactory+ (``buildNyancirGds``) and
XORs it against the ``.pic.yml`` layout on every layer.

Route settings other than cross-section, width, radius, bend type and
waypoints, instance arrays and ``o1-4``-style port ranges don't survive the
conversion yet; ``--verify`` reports those samples as different.

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
    modules = [
        m
        for m, mod in list(sys.modules.items())
        if (
            m == pdk_name or m.startswith(f"{pdk_name}.") or m.startswith(f"{PACKAGE}.")
        )
        and getattr(mod, name, None) is cell
    ]
    if not (cell and modules):
        return None
    # The defining module is the deepest one; packages only re-export.
    return f"{max(modules, key=lambda m: m.count('.'))}.{name}"


def fix_gsch(gsch: Path, stems: set[str], pdk_name: str, pdk: Any) -> list[str]:
    """Qualify models and write .pic.yml layout placement into one .gsch."""
    layout = {inst.name: inst for inst in gf.get_component(gsch.stem).insts}
    doc = json.loads(gsch.read_text())
    problems = []
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
        inst = layout.get(name)
        if inst is None:
            problems.append(f"{name!r} has no .pic.yml instance")
            continue
        local, world, trans = inst.cell.dbbox(), inst.dbbox(), inst.dcplx_trans
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
        )
    gsch.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")  # stable diffs
    return problems


def xor_area(a: Path, b: Path) -> dict[str, float]:
    """XOR area (um^2) per layer between the top cells of two GDS files."""
    la, lb = kdb.Layout(), kdb.Layout()
    la.read(str(a))
    lb.read(str(b))
    ta, tb = la.top_cells()[0], lb.top_cells()[0]
    diffs = {}
    for info in {(i.layer, i.datatype) for lay in (la, lb) for i in lay.layer_infos()}:
        ra, rb = (
            kdb.Region(top.begin_shapes_rec(idx))
            if (idx := lay.find_layer(*info)) is not None
            else kdb.Region()
            for lay, top in ((la, ta), (lb, tb))
        )
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
