"""Tests for logical electrical pins on PCells with electrical ports."""


def _electrical_port_names(component):
    """Return names of electrical ports on a component."""
    return [p.name for p in component.ports if p.port_type == "electrical"]


# ── si220 cband ──────────────────────────────────────────────────────────────


def test_si220_cband_straight_metal_has_electrical_pins():
    """straight_metal exposes >=2 electrical ports."""
    from cspdk.si220.cband import PDK

    PDK.activate()
    from cspdk.si220.cband.cells import straight_metal

    c = straight_metal()
    ports = _electrical_port_names(c)
    assert len(ports) >= 2, f"Expected >=2 electrical ports, got {ports}"
    assert len(c.pins) > 0, f"No logical pins on {c.name}"


def test_si220_cband_taper_metal_has_electrical_pins():
    """taper_metal exposes >=2 electrical ports."""
    from cspdk.si220.cband import PDK

    PDK.activate()
    from cspdk.si220.cband.cells import taper_metal

    c = taper_metal()
    ports = _electrical_port_names(c)
    assert len(ports) >= 2, f"Expected >=2 electrical ports, got {ports}"
    assert len(c.pins) > 0, f"No logical pins on {c.name}"


def test_si220_cband_wire_corner_has_electrical_pins():
    """wire_corner exposes >=2 electrical ports."""
    from cspdk.si220.cband import PDK

    PDK.activate()
    from cspdk.si220.cband.cells import wire_corner

    c = wire_corner()
    ports = _electrical_port_names(c)
    assert len(ports) >= 2, f"Expected >=2 electrical ports, got {ports}"
    assert len(c.pins) > 0, f"No logical pins on {c.name}"


def test_si220_cband_bend_metal_has_electrical_pins():
    """bend_metal exposes >=2 electrical ports."""
    from cspdk.si220.cband import PDK

    PDK.activate()
    from cspdk.si220.cband.cells import bend_metal

    c = bend_metal()
    ports = _electrical_port_names(c)
    assert len(ports) >= 2, f"Expected >=2 electrical ports, got {ports}"
    assert len(c.pins) > 0, f"No logical pins on {c.name}"


# ── si220 oband ──────────────────────────────────────────────────────────────


def test_si220_oband_straight_metal_has_electrical_pins():
    """straight_metal (oband) exposes >=2 electrical ports."""
    from cspdk.si220.oband import PDK

    PDK.activate()
    from cspdk.si220.oband.cells import straight_metal

    c = straight_metal()
    ports = _electrical_port_names(c)
    assert len(ports) >= 2, f"Expected >=2 electrical ports, got {ports}"
    assert len(c.pins) > 0, f"No logical pins on {c.name}"


def test_si220_oband_taper_metal_has_electrical_pins():
    """taper_metal (oband) exposes >=2 electrical ports."""
    from cspdk.si220.oband import PDK

    PDK.activate()
    from cspdk.si220.oband.cells import taper_metal

    c = taper_metal()
    ports = _electrical_port_names(c)
    assert len(ports) >= 2, f"Expected >=2 electrical ports, got {ports}"
    assert len(c.pins) > 0, f"No logical pins on {c.name}"


# ── si500 ─────────────────────────────────────────────────────────────────────


def test_si500_wire_corner_has_electrical_pins():
    """wire_corner (si500) exposes >=2 electrical ports."""
    from cspdk.si500 import PDK

    PDK.activate()
    from cspdk.si500.cells import wire_corner

    c = wire_corner()
    ports = _electrical_port_names(c)
    assert len(ports) >= 2, f"Expected >=2 electrical ports, got {ports}"
    assert len(c.pins) > 0, f"No logical pins on {c.name}"


def test_si500_pad_has_electrical_pins():
    """Pad (si500) exposes >=1 electrical port."""
    from cspdk.si500 import PDK

    PDK.activate()
    from cspdk.si500.cells import pad

    c = pad()
    ports = _electrical_port_names(c)
    assert len(ports) >= 1, f"Expected >=1 electrical port, got {ports}"
    assert len(c.pins) > 0, f"No logical pins on {c.name}"


def test_si500_compass_has_electrical_pins():
    """Compass (si500) exposes >=1 electrical port."""
    from cspdk.si500 import PDK

    PDK.activate()
    from cspdk.si500.cells import compass

    c = compass()
    ports = _electrical_port_names(c)
    assert len(ports) >= 1, f"Expected >=1 electrical port, got {ports}"
    assert len(c.pins) > 0, f"No logical pins on {c.name}"


# ── sin300 ────────────────────────────────────────────────────────────────────


def test_sin300_wire_corner_has_electrical_pins():
    """wire_corner (sin300) exposes >=2 electrical ports."""
    from cspdk.sin300 import PDK

    PDK.activate()
    from cspdk.sin300.cells import wire_corner

    c = wire_corner()
    ports = _electrical_port_names(c)
    assert len(ports) >= 2, f"Expected >=2 electrical ports, got {ports}"
    assert len(c.pins) > 0, f"No logical pins on {c.name}"


def test_sin300_pad_has_electrical_pins():
    """Pad (sin300) exposes >=1 electrical port."""
    from cspdk.sin300 import PDK

    PDK.activate()
    from cspdk.sin300.cells import pad

    c = pad()
    ports = _electrical_port_names(c)
    assert len(ports) >= 1, f"Expected >=1 electrical port, got {ports}"
    assert len(c.pins) > 0, f"No logical pins on {c.name}"


# ── si340 ─────────────────────────────────────────────────────────────────────


def test_si340_wire_corner_has_electrical_pins():
    """wire_corner (si340) exposes >=2 electrical ports."""
    from cspdk.si340 import PDK

    PDK.activate()
    from cspdk.si340.cells import wire_corner

    c = wire_corner()
    ports = _electrical_port_names(c)
    assert len(ports) >= 2, f"Expected >=2 electrical ports, got {ports}"
    assert len(c.pins) > 0, f"No logical pins on {c.name}"


def test_si340_compass_has_electrical_pins():
    """Compass (si340) exposes >=1 electrical port."""
    from cspdk.si340 import PDK

    PDK.activate()
    from cspdk.si340.cells import compass

    c = compass()
    ports = _electrical_port_names(c)
    assert len(ports) >= 1, f"Expected >=1 electrical port, got {ports}"
    assert len(c.pins) > 0, f"No logical pins on {c.name}"


# ── sin200 ────────────────────────────────────────────────────────────────────


def test_sin200_wire_corner_has_electrical_pins():
    """wire_corner (sin200) exposes >=2 electrical ports."""
    from cspdk.sin200 import PDK

    PDK.activate()
    from cspdk.sin200.cells import wire_corner

    c = wire_corner()
    ports = _electrical_port_names(c)
    assert len(ports) >= 2, f"Expected >=2 electrical ports, got {ports}"
    assert len(c.pins) > 0, f"No logical pins on {c.name}"


def test_sin200_pad_has_electrical_pins():
    """Pad (sin200) exposes >=1 electrical port."""
    from cspdk.sin200 import PDK

    PDK.activate()
    from cspdk.sin200.cells import pad

    c = pad()
    ports = _electrical_port_names(c)
    assert len(ports) >= 1, f"Expected >=1 electrical port, got {ports}"
    assert len(c.pins) > 0, f"No logical pins on {c.name}"
