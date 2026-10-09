"""Tests for the SiN300 sample cells and their circuit simulation."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from mycspdk.samples.mzi_with_gratings import mzi_spectrum, mzi_with_gratings


@pytest.mark.parametrize("cross_section", ["xs_nc", "xs_no"])
def test_mzi_with_gratings_builds(cross_section: str) -> None:
    """The MZI is wired to one input and two output grating couplers."""
    c = mzi_with_gratings(cross_section=cross_section)
    assert {p.name for p in c.ports} == {"in", "out1", "out2"}


@pytest.mark.parametrize("cross_section", ["xs_nc", "xs_no"])
def test_mzi_outputs_interfere(cross_section: str) -> None:
    """The outputs are complementary: out1's share of the light swings from 0 to 1."""
    _, t = mzi_spectrum(cross_section=cross_section, points=201)
    share = t["out1"] / (t["out1"] + t["out2"])
    assert float(jnp.min(share)) < 0.01
    assert float(jnp.max(share)) > 0.99
