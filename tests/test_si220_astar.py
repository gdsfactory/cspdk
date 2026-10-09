"""The optional A* routing strategies of the si220 PDK."""

from __future__ import annotations

import subprocess
import sys

import gdsfactory as gf
import pytest

ASTAR_YAML = """
instances:
  l1: {component: straight}
  l2: {component: straight}
  r1: {component: straight}
  r2: {component: straight}
  block: {component: straight, settings: {length: 20, width: 40}}
placements:
  l1: {y: 50}
  l2: {y: 0}
  r1: {x: 200, y: 50}
  r2: {x: 200, y: 0}
  block: {x: 90, y: 25}
routes:
  bundle:
    links: {"r1-2,o1": "l1-2,o2"}
    routing_strategy: route_astar
    settings: {spacing: 2}
"""


def test_astar_strategy_routes_around_obstacle() -> None:
    """With doroutes installed, schematics can use route_astar."""
    pytest.importorskip("doroutes")
    from cspdk.si220 import PDK

    assert {"route_astar", "route_astar_metal"} <= PDK.routing_strategies.keys()
    PDK.activate()
    gf.clear_cache()  # other flavours cache cells with the same names
    c = gf.read.from_yaml(ASTAR_YAML, name="astar_obstacle_test")
    assert len(c.insts) > 5  # the 5 placed instances plus the routed bends


def test_astar_strategies_skipped_without_doroutes() -> None:
    """Without doroutes the A* strategies are absent and the reason is logged."""
    script = """
import sys
sys.modules["doroutes"] = None  # make `import doroutes` fail
import gdsfactory as gf  # configures the logger, so add the sink after
gf.logger.add(sys.stdout, level="INFO", format="{message}")
from cspdk.si220 import PDK
print(sorted(PDK.routing_strategies))
"""
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "doroutes is not installed" in out
    assert "'route_astar'" not in out.splitlines()[-1]
