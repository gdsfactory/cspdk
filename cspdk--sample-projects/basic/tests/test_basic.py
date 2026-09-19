"""Basic tests for cspdk.

These tests verify that cspdk imports correctly, the PDK can be activated,
and fundamental components can be created without requiring simulation or
API keys.
"""

import gdsfactory as gf
from mycspdk.samples.sample0 import sample0_routed_mzi

from cspdk.si220.cband import PDK, cells


def test_import():
    """Verify that cspdk and its submodules import without errors."""
    import cspdk
    import cspdk.si220.cband

    assert cspdk is not None
    assert cspdk.si220.cband is not None


def test_pdk_activation():
    """Verify that the si220 C-band PDK activates correctly."""
    PDK.activate()
    assert gf.get_active_pdk().name == "cspdk.si220.cband"


def test_cell_creation():
    """Create a basic straight waveguide and verify it is a valid component."""
    c = cells.straight(length=10)
    assert isinstance(c, gf.Component)
    assert len(c.ports) > 0


def test_multiple_cells():
    """Create several different cell types and verify each is a valid component."""
    coupler = cells.coupler()
    assert isinstance(coupler, gf.Component)
    assert len(coupler.ports) > 0

    mmi = cells.mmi1x2()
    assert isinstance(mmi, gf.Component)
    assert len(mmi.ports) > 0

    bend = cells.bend_euler()
    assert isinstance(bend, gf.Component)
    assert len(bend.ports) > 0


def test_routing():
    """Create a simple routed component connecting two straights."""
    c = sample0_routed_mzi()
    assert isinstance(c, gf.Component)
    assert {port.name for port in c.ports} == {"o1", "o2"}


def test_component_ports():
    """Verify that a straight waveguide has the expected input/output ports."""
    c = cells.straight(length=10)
    port_names = [p.name for p in c.ports]
    assert "o1" in port_names, f"Expected port 'o1', found {port_names}"
    assert "o2" in port_names, f"Expected port 'o2', found {port_names}"
