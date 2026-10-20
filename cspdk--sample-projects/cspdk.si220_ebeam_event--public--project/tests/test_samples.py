"""Tests for the e-beam event sample: the die design and the ring simulation."""

from __future__ import annotations

from pathlib import Path

import gdsfactory as gf
import jax.numpy as jnp
import numpy as np
import pytest
from cspdk.si220_ebeam_event import PDK, TECH, check
from cspdk.si220_ebeam_event.mask import layer_region
from cspdk.si220_ebeam_event.tech import LAYER

from mycspdk.samples.ring_die import ring_die, ring_transmission

SCHEMATIC = Path(__file__).resolve().parents[1] / "mycspdk" / "ring_die.pic.yml"


@pytest.fixture(autouse=True)
def activate_pdk() -> None:
    """Activate the PDK."""
    PDK.activate()


def test_ring_die_passes_event_rules() -> None:
    """The ring die fills the 6 x 3 mm design area, routes and passes DRC."""
    design = ring_die()
    assert (design.xsize, design.ysize) == pytest.approx(TECH.die_size)
    assert layer_region(design, LAYER.routing_error_marker).is_empty()
    assert check(design) == []


def test_schematic_passes_event_rules() -> None:
    """The ring_die schematic builds and passes DRC."""
    design = gf.read.from_yaml(SCHEMATIC)
    assert layer_region(design, LAYER.routing_error_marker).is_empty()
    assert check(design) == []


def test_ring_fsr_follows_group_index() -> None:
    """The free spectral range is wl^2 / (ng L) with the event's ng of 3.85."""
    radius = 25.0
    wl, t = ring_transmission(radius=radius)
    t = np.asarray(t)
    dips = np.flatnonzero((t[1:-1] < t[:-2]) & (t[1:-1] < t[2:])) + 1
    fsr = float(np.mean(np.diff(np.asarray(wl)[dips])))
    # ring_single defaults: 4 um coupling straight, 0.6 um vertical straights
    length = 2 * np.pi * radius + 2 * 4.0 + 2 * 0.6
    assert fsr == pytest.approx(1.55**2 / (3.85 * length), rel=0.01)
    assert float(jnp.max(t)) <= 1.0
